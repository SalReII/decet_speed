"""The fused training step against the unpatched model, on a small random 4-bit Qwen2 + LoRA:
loss, LoRA gradients, TRL's entropy and accuracy, and Soup's per-row rewind losses.

On a GPU:      python -m pytest solution -q
Without one:   TRITON_INTERPRET=1 python -m pytest solution -q
"""

import pytest
import torch

from solution._common import interpreting

if torch.cuda.is_available():
    DEV = "cuda"
elif interpreting():
    DEV = "cpu"
else:
    DEV = None

pytestmark = pytest.mark.skipif(DEV is None, reason="needs a CUDA GPU or TRITON_INTERPRET=1")


@pytest.fixture(scope="module")
def lora_model(tmp_path_factory):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Qwen2Config, \
        Qwen2ForCausalLM

    path = tmp_path_factory.mktemp("tiny_qwen2")
    cfg = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                proj.bias.normal_(0, 0.1)
            layer.input_layernorm.weight.uniform_(0.5, 1.5)
            layer.post_attention_layernorm.weight.uniform_(0.5, 1.5)
    model.to(torch.float16).save_pretrained(path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant,
                                                 device_map={"": DEV}, dtype=torch.float16,
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    model.train()
    return model


def batch(vocab=1000):
    torch.manual_seed(1)
    lengths = [24, 15, 6]
    seq = max(lengths)
    ids = torch.randint(0, vocab, (len(lengths), seq))
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for row, n in enumerate(lengths):
        mask[row, :n] = 1
        prompt = n // 3
        labels[row, prompt:n] = ids[row, prompt:n]
    ids[mask == 0] = 0
    return ids.to(DEV), mask.to(DEV), labels.to(DEV)


def run(model, fused):
    from solution.model import HeadStats
    from solution.trainer_patch import _FusedRequest, patch_forward

    patch_forward()
    ids, mask, labels = batch()
    n_items = (labels[:, 1:] != -100).sum()
    for param in model.parameters():
        param.grad = None
    torch.manual_seed(123)
    with torch.autocast(DEV, dtype=torch.float16):
        if fused:
            with _FusedRequest():
                out = model(input_ids=ids, attention_mask=mask, labels=labels,
                            num_items_in_batch=n_items)
        else:
            out = model(input_ids=ids, attention_mask=mask, labels=labels,
                        num_items_in_batch=n_items)
    (out.loss * 1024).backward()
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
             if p.grad is not None}
    if fused:
        stats = out.logits
        assert isinstance(stats, HeadStats)
        entropy = float(stats.entropy_sum / mask.sum())
        accuracy = float(stats.correct / stats.labelled)
        rows = stats.row_losses()
    else:
        from soup_cli.trainer.rewind_hf import row_losses
        from trl.trainer.utils import entropy_from_logits

        logits = out.logits.detach().float()
        entropy = float((entropy_from_logits(logits) * mask).sum() / mask.sum())
        pred = logits[:, :-1].argmax(-1)
        tgt = labels[:, 1:]
        accuracy = float(((pred == tgt) & (tgt != -100)).sum() / (tgt != -100).sum())
        rows = row_losses(logits, labels)
    return float(out.loss), grads, entropy, accuracy, rows


def test_fused_step_matches_unpatched_model(lora_model):
    ref = run(lora_model, fused=False)
    got = run(lora_model, fused=True)
    assert abs(got[0] - ref[0]) / ref[0] < 2e-3, (got[0], ref[0])
    assert set(got[1]) == set(ref[1]) and ref[1]
    for name, grad in ref[1].items():
        rel = (got[1][name] - grad).norm() / grad.norm().clamp(min=1e-12)
        assert rel < 2e-2, (name, float(rel))
    assert abs(got[2] - ref[2]) / ref[2] < 2e-3, (got[2], ref[2])
    assert abs(got[3] - ref[3]) < 1e-6 + 0.02, (got[3], ref[3])
    assert got[4][1] == ref[4][1]
    for a, b in zip(got[4][0], ref[4][0]):
        assert abs(a - b) / b < 2e-3, (got[4][0], ref[4][0])
