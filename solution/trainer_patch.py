"""The patches that route Soup's training step through model.fused_forward.

1. `Qwen2ForCausalLM.forward`: takes the fused path while our compute_loss asks for it and the
   model/inputs are ones it handles; otherwise the original forward, untouched.
2. `trl.SFTTrainer.compute_loss`: the same steps and metrics as TRL 0.29.1 (entropy, num_tokens,
   mean_token_accuracy, by TRL's formulas), read from HeadStats instead of the logits.
3. `soup_cli.trainer.rewind_hf.row_losses`: Soup's per-row losses for rewind.jsonl from
   HeadStats (the same mean token cross-entropy per row).
4. `SFTTrainerWrapper._setup_transformers`: starts compiling our kernels in the background
   before the model loads; afterwards stores the frozen tied embedding / lm_head in fp16. The
   fp32 table that `prepare_model_for_kbit_training` makes holds the checkpoint's fp16 values
   (loaded as fp16 on a T4), so the fp16 copy is exact; the embedding output is cast back to
   fp32 so every other path sees what it saw before.
Every patch falls through to the original code for anything else (other models, eval,
generation, the fake modules of Soup's regression tests).
"""

import threading

import torch

from . import warmup
from .model import HeadStats, can_run, fused_forward, model_plan

_local = threading.local()


class _FusedRequest:
    def __enter__(self):
        _local.depth = getattr(_local, "depth", 0) + 1

    def __exit__(self, *exc):
        _local.depth -= 1
        return False


def _requested():
    return getattr(_local, "depth", 0) > 0


def _causal_lm(model):
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def patch_forward():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls.forward, "_soup_kernel", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if _requested() and position_ids is None and past_key_values is None \
                and inputs_embeds is None and isinstance(logits_to_keep, int) \
                and logits_to_keep == 0 and not use_cache \
                and can_run(self, input_ids, attention_mask, labels, kwargs):
            out = fused_forward(self, input_ids, attention_mask, labels,
                                kwargs.get("num_items_in_batch"))
            if out is not None:
                return out
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    forward._soup_kernel = True
    forward.__wrapped__ = original
    cls.forward = forward


def _wants_fused(trainer, model, inputs):
    try:
        if getattr(trainer.args, "use_liger_kernel", False) or not trainer.model.training:
            return False
        if not isinstance(inputs, dict) or any(k not in inputs for k in
                                               ("input_ids", "attention_mask", "labels")):
            return False
        if "shift_labels" in inputs or "position_ids" in inputs:
            return False
        if getattr(trainer, "num_virtual_tokens", 0) or getattr(trainer, "aux_loss_enabled",
                                                                 False):
            return False
        if getattr(trainer, "compute_loss_func", None) is not None or \
                getattr(trainer, "label_smoother", None) is not None:
            return False
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

        lm = _causal_lm(trainer.accelerator.unwrap_model(model))
        return isinstance(lm, Qwen2ForCausalLM) and model_plan(lm) is not None
    except Exception:  # noqa: BLE001
        return False


def patch_sft_compute_loss():
    from trl.trainer import sft_trainer as trl_sft

    sft_cls = trl_sft.SFTTrainer
    if getattr(sft_cls.compute_loss, "_soup_kernel", False):
        return
    original = sft_cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _wants_fused(self, model, inputs):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        warmup.wait()
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        with _FusedRequest():
            loss, outputs = super(sft_cls, self).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        stats = getattr(outputs, "logits", None)
        fused = isinstance(stats, HeadStats)
        attention_mask = inputs["attention_mask"]
        gather = self.accelerator.gather_for_metrics

        # TRL: entropy over the positions with attention_mask == 1
        with torch.no_grad():
            if fused:
                entropy = stats.entropy_sum / attention_mask.sum()
            else:
                per_token_entropy = trl_sft.entropy_from_logits(outputs.logits)
                entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
            entropy = gather(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = gather(attention_mask.sum()).sum().item()
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # TRL: argmax accuracy over the labelled next-token positions
        with torch.no_grad():
            if fused:
                correct_tokens, total_tokens = stats.correct, stats.labelled
            else:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                correct_tokens = ((predictions == shift_labels) & mask).sum()
                total_tokens = mask.sum()
            correct_tokens = gather(correct_tokens)
            total_tokens = gather(total_tokens)
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_kernel = True
    compute_loss.__wrapped__ = original
    sft_cls.compute_loss = compute_loss


def patch_rewind():
    from soup_cli.trainer import rewind_hf

    if getattr(rewind_hf.row_losses, "_soup_kernel", False):
        return
    original = rewind_hf.row_losses

    def row_losses(logits, labels):
        if isinstance(logits, HeadStats):
            return logits.row_losses()
        return original(logits, labels)

    row_losses._soup_kernel = True
    row_losses.__wrapped__ = original
    rewind_hf.row_losses = row_losses


def _upcast_output(_module, _inputs, output):
    return output.float() if torch.is_tensor(output) and output.dtype == torch.float16 else output


def prepare_model(model):
    """After Soup built the 4-bit PEFT model: validate it for the fused path and keep the frozen
    tied embedding / lm_head table in fp16 (exact, see the module docstring)."""
    if not isinstance(model, torch.nn.Module):
        return
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    lm = _causal_lm(model)
    if not isinstance(lm, Qwen2ForCausalLM) or model_plan(lm) is None:
        return
    embed = lm.get_input_embeddings()
    weight = embed.weight
    if weight.dtype != torch.float32 or weight.requires_grad or not weight.is_cuda:
        return
    half = torch.empty_like(weight.data, dtype=torch.float16)
    for src, dst in zip(weight.data.split(8192), half.split(8192)):
        dst.copy_(src)
        if not torch.equal(dst.float(), src):
            return  # the table holds values fp16 cannot represent: keep it as it is
    weight.data = half
    embed.register_forward_hook(_upcast_output)
    torch.cuda.empty_cache()
    from .model import DENSE_WEIGHTS

    if DENSE_WEIGHTS:
        warmup.wait()
        for lp in model_plan(lm).layers:
            lp.unpack_dense()


def patch_setup():
    from soup_cli.trainer.sft import SFTTrainerWrapper

    if getattr(SFTTrainerWrapper._setup_transformers, "_soup_kernel", False):
        return
    original = SFTTrainerWrapper._setup_transformers

    def _setup_transformers(self, cfg, tcfg, *args, **kwargs):
        try:
            warmup.start(getattr(cfg, "base", None))
        except Exception:  # noqa: BLE001
            pass
        result = original(self, cfg, tcfg, *args, **kwargs)
        try:
            prepare_model(getattr(self, "model", None))
        except Exception as exc:  # noqa: BLE001
            print(f"[solution] model preparation skipped: {exc!r}", flush=True)
        return result

    _setup_transformers._soup_kernel = True
    _setup_transformers.__wrapped__ = original
    SFTTrainerWrapper._setup_transformers = _setup_transformers
