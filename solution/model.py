"""The training forward of Qwen2ForCausalLM + LoRA built from our kernels, on real tokens only.

`fused_forward` replaces `Qwen2ForCausalLM.forward` for one case only: a labelled training step
that our trainer patch asked for (see trainer_patch.py), on a model whose layers have the shape
of this task (bitsandbytes 4-bit projections, at most one vanilla LoRA adapter on q_proj/v_proj,
SDPA attention, fp16 autocast). Everything else goes to the original forward.

Per decoder layer, with s the fp32 residual stream of the M real tokens:
  s, y, y_pad = add_rmsnorm(s, d)               Triton (d: previous block output, fp16)
  qkv        = y @ [Wq;Wk;Wv]^T + b               Triton NF4 unpack + one cuBLAS GEMM
  lq, lv     = LoRA(dropout(y_pad))               PyTorch, as peft does it, same dropout draws
  q, k, v    = rope_pack(qkv, lq, lv)             Triton, [heads, tokens, 128], k/v repeated
  a          = SDPA(q, k, v) per row, is_causal   PyTorch (memory-efficient kernel)
  o          = a @ Wo^T                           Triton NF4 unpack + cuBLAS
  s, y2, _   = add_rmsnorm(s, o)                  Triton
  d          = mlp(y2)                            Triton NF4 + cuBLAS + Triton SwiGLU
and at the end the final norm and the fused head/loss (head.py).

Pad tokens never enter any of it: base Soup computes them and throws the result away (their
labels are -100, their keys are masked, TRL's entropy multiplies them by the attention mask).
Attention runs once per row on the row's real tokens with a causal mask, which is what base
Soup's causal + padding mask leaves of each row. Only right-padded batches (what TRL's collator
builds) take this path.
"""

import torch

from .head import IGNORE_INDEX, fused_head_loss
from .mlp import fused_mlp
from .nf4 import frozen_linear, unpack_concat

# Speed over memory: unpack every frozen 4-bit projection to fp16 once, when the plan is built
# (by our nf4_unpack_kernel, bit-identical to what bitsandbytes produces on every call), instead
# of twice per projection per step. Costs ~2.6 GB for Qwen2.5-1.5B; the 4-bit weights stay as they
# are in the model.
DENSE_WEIGHTS = True
from .rmsnorm import add_rmsnorm
from .rope import packed_causal_attention, rope_pack

_ALLOWED_KWARGS = {"num_items_in_batch", "output_attentions", "output_hidden_states",
                   "return_dict"}


class HeadStats:
    """What TRL's metrics and Soup's rewind log need from one labelled forward, in place of the
    [batch, seq, vocab] logits they would otherwise read. Lives in `outputs.logits`."""

    def __init__(self, entropy_sum, correct, labelled, row_loss_sum, row_tokens):
        self.entropy_sum = entropy_sum      # sum of token entropies over attention_mask == 1
        self.correct = correct              # labelled tokens whose argmax is the label
        self.labelled = labelled            # labelled tokens
        self.row_loss_sum = row_loss_sum    # [batch] summed token cross-entropy per row
        self.row_tokens = row_tokens        # [batch] labelled tokens per row

    def row_losses(self):
        """(row_loss, row_tokens) exactly as soup_cli.trainer.rewind_hf.row_losses returns."""
        tokens = self.row_tokens
        means = self.row_loss_sum / tokens.clamp(min=1).to(self.row_loss_sum.dtype)
        return [float(m) for m in means.tolist()], [int(t) for t in tokens.tolist()]


def _base_layer(module):
    get = getattr(module, "get_base_layer", None)
    return get() if callable(get) else module


def _is_4bit(module):
    try:
        import bitsandbytes as bnb
    except ImportError:  # pragma: no cover
        return False
    return isinstance(module, bnb.nn.Linear4bit) and \
        getattr(module.weight, "quant_state", None) is not None


def _lora_parts(module):
    """None for a plain layer; (dropout, A, B, scaling) for one active vanilla LoRA adapter;
    False for anything else (merged, disabled, several adapters, DoRA, ...)."""
    if not hasattr(module, "lora_A"):
        return None
    try:
        if getattr(module, "merged", False) or getattr(module, "disable_adapters", False):
            return False
        active = [a for a in module.active_adapters if a in module.lora_A.keys()]
        if not active:
            return None
        if len(active) != 1:
            return False
        name = active[0]
        if name in getattr(module, "lora_variant", {}) or \
                getattr(module, "use_dora", {}).get(name, False) or \
                getattr(module, "lora_bias", {}).get(name, False):
            return False
        dropout = module.lora_dropout[name]
        if not isinstance(dropout, (torch.nn.Dropout, torch.nn.Identity)):
            return False
        return dropout, module.lora_A[name], module.lora_B[name], float(module.scaling[name])
    except Exception:  # noqa: BLE001 - an unexpected peft layout: not ours to run
        return False


class LayerPlan:
    __slots__ = ("layer", "attn", "q", "k", "v", "o", "gate", "up", "down", "qkv_bias",
                 "o_bias", "lora_modules", "dense_qkv", "dense_o", "dense_mlp")

    def __init__(self, layer):
        self.layer = layer
        self.attn = layer.self_attn
        attn, mlp = layer.self_attn, layer.mlp
        self.lora_modules = (attn.q_proj, attn.v_proj)
        bases = [_base_layer(m) for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                                          mlp.gate_proj, mlp.up_proj, mlp.down_proj)]
        if not all(_is_4bit(b) for b in bases):
            raise ValueError("not a bitsandbytes 4-bit layer")
        q, k, v, o, gate, up, down = bases
        self.q, self.k, self.v, self.o = q.weight, k.weight, v.weight, o.weight
        self.gate, self.up, self.down = gate.weight, up.weight, down.weight
        if any(b.bias is not None for b in (gate, up, down)):
            raise ValueError("MLP bias")
        biases = [b.bias for b in (q, k, v)]
        if all(b is not None for b in biases):
            self.qkv_bias = torch.cat([b.detach().to(torch.float16) for b in biases])
        elif all(b is None for b in biases):
            self.qkv_bias = None
        else:
            raise ValueError("mixed q/k/v bias")
        self.o_bias = o.bias.detach().to(torch.float16) if o.bias is not None else None
        for module in (attn.k_proj, attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            if _lora_parts(module) is not None:
                raise ValueError("LoRA outside q_proj/v_proj")
        self.dense_qkv = self.dense_o = self.dense_mlp = None

    def unpack_dense(self):
        dev = self.q.device
        with torch.no_grad():
            self.dense_qkv = unpack_concat((self.q, self.k, self.v), dev)
            self.dense_o = unpack_concat((self.o,), dev)
            self.dense_mlp = (unpack_concat((self.gate, self.up), dev),
                              unpack_concat((self.down,), dev))


class ModelPlan:
    def __init__(self, causal_lm):
        cfg = causal_lm.config
        inner = causal_lm.model
        if getattr(cfg, "_attn_implementation", None) != "sdpa":
            raise ValueError("attention is not sdpa")
        if getattr(cfg, "attention_dropout", 0.0):
            raise ValueError("attention dropout")
        if getattr(inner, "has_sliding_layers", False) or \
                any(t != "full_attention" for t in getattr(cfg, "layer_types", [])):
            raise ValueError("sliding-window layers")
        if getattr(cfg, "hidden_act", "silu") != "silu":
            raise ValueError("activation is not silu")
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hidden = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", None) or self.hidden // self.n_heads
        if self.head_dim % 2 or self.n_heads % self.n_kv:
            raise ValueError("odd head_dim or uneven kv groups")
        self.layers = [LayerPlan(layer) for layer in inner.layers[:cfg.num_hidden_layers]]
        self.embed = inner.embed_tokens
        self.head = causal_lm.lm_head
        if self.embed.weight.requires_grad or self.head.weight.requires_grad or \
                getattr(self.head, "bias", None) is not None:
            raise ValueError("trainable or biased embedding/head")
        if self.head.weight.shape[0] != cfg.vocab_size:
            raise ValueError("vocab size")


def model_plan(causal_lm):
    """The validated plan of this model, built once; None if the model is not one we run."""
    plan = causal_lm.__dict__.get("_soup_kernel_plan")
    if plan is None:
        try:
            plan = ModelPlan(causal_lm)
        except Exception as exc:  # noqa: BLE001
            plan = False
            print(f"[solution] fused path off for this model: {exc}", flush=True)
        causal_lm.__dict__["_soup_kernel_plan"] = plan
    return plan or None


def can_run(causal_lm, input_ids, attention_mask, labels, extra):
    from ._common import triton_ok

    if not (causal_lm.training and torch.is_grad_enabled() and labels is not None):
        return False
    if input_ids is None or attention_mask is None or input_ids.dim() != 2 \
            or attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
        return False
    if not triton_ok(input_ids):
        return False
    for key, value in extra.items():
        if key not in _ALLOWED_KWARGS or (key != "num_items_in_batch" and key != "return_dict"
                                          and value):
            return False
    device_type = input_ids.device.type
    if not torch.is_autocast_enabled(device_type) or \
            torch.get_autocast_dtype(device_type) != torch.float16:
        return False
    plan = model_plan(causal_lm)
    if plan is None:
        return False
    for lp in plan.layers:
        if _lora_parts(lp.attn.k_proj) is not None:
            return False
        for module in lp.lora_modules:
            if _lora_parts(module) is False:
                return False
    return True


def _lora_branch(module, y_pad, batch, seq, pos_idx):
    """peft's `lora_B(lora_A(dropout(x))) * scaling` for the real tokens, where x is the
    padded fp32 norm output base Soup feeds it (same shape and dtype, so the same dropout
    draws); A and B then run on the real tokens only."""
    parts = _lora_parts(module)
    if parts is None:
        return None
    dropout, lora_a, lora_b, scaling = parts
    hidden = y_pad.shape[-1]
    dropped = dropout(y_pad.view(batch, seq, hidden)).view(batch * seq, hidden)
    rows = dropped.index_select(0, pos_idx)
    return lora_b(lora_a(rows)) * scaling


def fused_forward(causal_lm, input_ids, attention_mask, labels, num_items_in_batch):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    plan = model_plan(causal_lm)
    inner = causal_lm.model
    batch, seq = input_ids.shape
    dev = input_ids.device
    padded = batch * seq
    mask = attention_mask.bool()
    # Right padding (TRL's collator): every row is a prefix of real tokens. One sync for both.
    not_prefix = (mask[:, 1:] & ~mask[:, :-1]).any().view(1)
    info = torch.cat([not_prefix.long(), mask.sum(1)]).tolist()
    if info[0]:
        return None
    lengths = info[1:]
    pos_idx = mask.reshape(-1).nonzero().squeeze(1)
    if DENSE_WEIGHTS and plan.layers and plan.layers[0].dense_qkv is None:
        for lp in plan.layers:
            lp.unpack_dense()

    # Embeddings of the real tokens. The table is held in fp16 (see trainer_patch); its values
    # are the fp32 table's, so the fp32 rows below are the ones base Soup feeds the first layer.
    ids = input_ids.reshape(-1).index_select(0, pos_idx)
    s = torch.nn.functional.embedding(ids, plan.embed.weight).float()

    shape_ref = s.new_empty((1, 1, 1)).expand(batch, seq, plan.hidden)
    position_ids = torch.arange(seq, device=dev).unsqueeze(0)
    cos, sin = inner.rotary_emb(shape_ref, position_ids)
    cos, sin = cos[0].float(), sin[0].float()

    delta = None
    for lp in plan.layers:
        layer = lp.layer
        s, y, y_pad = add_rmsnorm(s, delta, layer.input_layernorm.weight,
                                  layer.input_layernorm.variance_epsilon, pos_idx, padded)
        qkv = frozen_linear(y, (lp.q, lp.k, lp.v), lp.qkv_bias, lp.dense_qkv)
        lora_q = _lora_branch(lp.attn.q_proj, y_pad, batch, seq, pos_idx)
        lora_v = _lora_branch(lp.attn.v_proj, y_pad, batch, seq, pos_idx)
        del y_pad
        q, k, v = rope_pack(qkv, lora_q, lora_v, cos, sin, pos_idx, seq,
                            plan.n_heads, plan.n_kv, plan.head_dim)
        del qkv, lora_q, lora_v
        attn = packed_causal_attention(q, k, v, lengths, lp.attn.scaling)
        del q, k, v
        o = frozen_linear(attn, (lp.o,), lp.o_bias, lp.dense_o)
        del attn
        s, y2, _ = add_rmsnorm(s, o, layer.post_attention_layernorm.weight,
                               layer.post_attention_layernorm.variance_epsilon)
        delta = fused_mlp(y2, lp.gate, lp.up, lp.down, lp.dense_mlp)
        del y2, o

    _, y, _ = add_rmsnorm(s, delta, inner.norm.weight, inner.norm.variance_epsilon)
    del s, delta

    # Causal-LM targets, as ForCausalLMLoss shifts them: position t is scored against t + 1.
    targets = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
    targets = targets.reshape(-1).index_select(0, pos_idx)
    labelled_mask = targets != IGNORE_INDEX
    denominator = num_items_in_batch if num_items_in_batch is not None else labelled_mask.sum()
    weight = plan.head.weight
    if weight.dtype != torch.float16:
        weight = weight.to(torch.float16)
    loss, entropy, token_loss, correct = fused_head_loss(y, weight, targets, denominator)

    with torch.no_grad():
        row = torch.div(pos_idx, seq, rounding_mode="floor")
        row_loss_sum = torch.zeros(batch, dtype=torch.float32, device=dev).index_add_(
            0, row, token_loss)
        row_tokens = torch.zeros(batch, dtype=torch.long, device=dev).index_add_(
            0, row, labelled_mask.long())
        stats = HeadStats(entropy_sum=entropy.sum(), correct=correct.sum(),
                          labelled=labelled_mask.sum(), row_loss_sum=row_loss_sum,
                          row_tokens=row_tokens)
    return CausalLMOutputWithPast(loss=loss, logits=stats)
