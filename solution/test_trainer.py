"""The trainer glue: TRL's SFTTrainer.compute_loss (patched) with Soup's rewind recorder on top
logs the same loss, entropy, num_tokens, accuracy and rewind rows as the unpatched one.

On a GPU:      python -m pytest solution -q
Without one:   TRITON_INTERPRET=1 python -m pytest solution -q
"""

import pytest
import torch

from solution.test_model import DEV, batch, lora_model  # noqa: F401 - fixture

pytestmark = pytest.mark.skipif(DEV is None, reason="needs a CUDA GPU or TRITON_INTERPRET=1")


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(1000)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t1"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="t0", eos_token="t2",
                                   unk_token="t1")


class Sink:
    def __init__(self):
        self.records = []

    def record_batch(self, **record):
        self.records.append(record)


def make_trainer(model, tmp_path):
    from datasets import Dataset
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from trl import SFTConfig, SFTTrainer

    data = Dataset.from_dict({"input_ids": [[3, 4, 5, 6]] * 4})
    args = SFTConfig(output_dir=str(tmp_path), per_device_train_batch_size=3, report_to=[],
                     use_cpu=DEV == "cpu", max_length=64, logging_steps=1, bf16=False,
                     fp16=False)
    trainer = make_rewind_trainer_class(SFTTrainer)(
        model=model, args=args, train_dataset=data, processing_class=tiny_tokenizer())
    sink = Sink()
    state = attach_rewind_state(trainer, sink)
    return trainer, state, sink


def one_step(trainer, state, sink, fused):
    from solution import trainer_patch

    trainer_patch.patch_forward()
    trainer_patch.patch_rewind()
    trainer_patch.patch_sft_compute_loss()
    ids, mask, labels = batch()
    for i in range(ids.shape[0]):
        state.on_index(i)
    trainer._metrics["train"].clear()
    trainer._total_train_tokens = 0
    inputs = {"input_ids": ids, "attention_mask": mask, "labels": labels}
    n_items = (labels[:, 1:] != -100).sum()
    compute = trainer.compute_loss
    if not fused:
        # the unpatched TRL compute_loss, with Soup's rewind subclass above it
        from trl.trainer.sft_trainer import SFTTrainer

        patched = SFTTrainer.compute_loss
        SFTTrainer.compute_loss = patched.__wrapped__
    try:
        torch.manual_seed(5)
        with torch.autocast(DEV, dtype=torch.float16):
            loss = compute(trainer.model, inputs, num_items_in_batch=n_items)
    finally:
        if not fused:
            SFTTrainer.compute_loss = patched
    metrics = {k: list(v) for k, v in trainer._metrics["train"].items()}
    return float(loss), metrics, sink.records[-1]


def test_trainer_metrics_and_rewind_match(lora_model, tmp_path):  # noqa: F811
    trainer, state, sink = make_trainer(lora_model, tmp_path)
    ref = one_step(trainer, state, sink, fused=False)
    got = one_step(trainer, state, sink, fused=True)
    assert not state.failed
    assert abs(got[0] - ref[0]) / ref[0] < 2e-3
    assert sorted(got[1]) == sorted(ref[1])
    assert got[1]["num_tokens"] == ref[1]["num_tokens"]
    for key in ("entropy", "mean_token_accuracy"):
        assert abs(got[1][key][0] - ref[1][key][0]) <= 2e-3 * max(abs(ref[1][key][0]), 1e-3)
    assert got[2]["rows"] == ref[2]["rows"] and got[2]["row_tokens"] == ref[2]["row_tokens"]
    for a, b in zip(got[2]["row_loss"], ref[2]["row_loss"]):
        assert abs(a - b) / b < 2e-3
