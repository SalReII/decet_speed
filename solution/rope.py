"""Q/K/V assembly for attention: LoRA add + rotary embedding + layout for SDPA, no padding.

Base Soup gets q, k and v from bitsandbytes as fp32 (the input was the fp32 norm output), adds
the LoRA outputs, transposes, applies the rotary embedding with seven PyTorch ops per tensor in
fp32 (two muls, a slice-negate-concat, an add), repeats k and v for grouped-query attention,
and SDPA's autocast casts all three to fp16 again; attention then runs on the padded batch with
an explicit [batch, 1, seq, seq] mask.

Here the fused q/k/v GEMM output (fp16, real tokens only) is read once per token by one Triton
program that adds the LoRA outputs in fp32, rotates q and k in fp32 (position = the token's
index in its row) and writes fp16 q, k, v as [heads, tokens, 128], with k and v already
repeated to the 12 query heads (each kv head written to its group of query heads, as
`repeat_kv` lays it out). Rows are contiguous along the token axis, so attention runs per row
with `is_causal` on the row's real length: no padding and no mask. Rounding happens once, where
base Soup's autocast rounds. Backward is the transposed rotation (summing the repeated k/v
gradients in fp32), one program per token, writing the fp16 gradient of the q/k/v GEMM output;
the LoRA branches get views of the same buffer (in base Soup both receive the same fp16 cast of
the fp32 gradient).
"""

import torch
import triton
import triton.language as tl

from ._common import next_pow2


@triton.jit(do_not_specialize=["M", "T", "has_lq", "has_lv"])
def rope_pack_fwd_kernel(qkv_ptr, lq_ptr, lv_ptr, cos_ptr, sin_ptr, pos_ptr, q_ptr, k_ptr, v_ptr,
                         M, T, has_lq, has_lv,
                         NH: tl.constexpr, NKV: tl.constexpr, HALF: tl.constexpr,
                         NH_P: tl.constexpr, NKV_P: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    t = tl.load(pos_ptr + tok).to(tl.int64) % T
    D = 2 * HALF
    W = (NH + 2 * NKV) * D
    half = tl.arange(0, HALF)[None, :]
    c1 = tl.load(cos_ptr + t * D + half)
    c2 = tl.load(cos_ptr + t * D + HALF + half)
    s1 = tl.load(sin_ptr + t * D + half)
    s2 = tl.load(sin_ptr + t * D + HALF + half)
    row = qkv_ptr + tok * W

    # queries (+ LoRA), rotated -> q[h, tok, :]
    hq = tl.arange(0, NH_P)[:, None]
    mq = (hq < NH) & (half < HALF)
    off = hq * D + half
    x1 = tl.load(row + off, mask=mq, other=0.0).to(tl.float32)
    x2 = tl.load(row + off + HALF, mask=mq, other=0.0).to(tl.float32)
    if has_lq:
        x1 = x1 + tl.load(lq_ptr + tok * (NH * D) + off, mask=mq, other=0.0).to(tl.float32)
        x2 = x2 + tl.load(lq_ptr + tok * (NH * D) + off + HALF, mask=mq, other=0.0).to(tl.float32)
    dst = q_ptr + (hq * M + tok) * D + half
    tl.store(dst, (x1 * c1 - x2 * s1).to(tl.float16), mask=mq)
    tl.store(dst + HALF, (x2 * c2 + x1 * s2).to(tl.float16), mask=mq)

    # keys, rotated, and values (+ LoRA): kv head j goes to query heads j*G .. j*G+G-1
    hk = tl.arange(0, NKV_P)[:, None]
    mk = (hk < NKV) & (half < HALF)
    off = NH * D + hk * D + half
    x1 = tl.load(row + off, mask=mk, other=0.0).to(tl.float32)
    x2 = tl.load(row + off + HALF, mask=mk, other=0.0).to(tl.float32)
    k1 = (x1 * c1 - x2 * s1).to(tl.float16)
    k2 = (x2 * c2 + x1 * s2).to(tl.float16)
    off = (NH + NKV) * D + hk * D + half
    v1 = tl.load(row + off, mask=mk, other=0.0).to(tl.float32)
    v2 = tl.load(row + off + HALF, mask=mk, other=0.0).to(tl.float32)
    if has_lv:
        lrow = lv_ptr + tok * (NKV * D) + hk * D + half
        v1 = v1 + tl.load(lrow, mask=mk, other=0.0).to(tl.float32)
        v2 = v2 + tl.load(lrow + HALF, mask=mk, other=0.0).to(tl.float32)
    v1 = v1.to(tl.float16)
    v2 = v2.to(tl.float16)
    for r in tl.static_range(NH // NKV):
        dst = ((hk * (NH // NKV) + r) * M + tok) * D + half
        tl.store(k_ptr + dst, k1, mask=mk)
        tl.store(k_ptr + dst + HALF, k2, mask=mk)
        tl.store(v_ptr + dst, v1, mask=mk)
        tl.store(v_ptr + dst + HALF, v2, mask=mk)


@triton.jit(do_not_specialize=["M", "T", "has_dq", "has_dk", "has_dv"])
def rope_pack_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, pos_ptr, dqkv_ptr,
                         M, T, has_dq, has_dk, has_dv,
                         NH: tl.constexpr, NKV: tl.constexpr, HALF: tl.constexpr,
                         NH_P: tl.constexpr, NKV_P: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    t = tl.load(pos_ptr + tok).to(tl.int64) % T
    D = 2 * HALF
    W = (NH + 2 * NKV) * D
    half = tl.arange(0, HALF)[None, :]
    c1 = tl.load(cos_ptr + t * D + half)
    c2 = tl.load(cos_ptr + t * D + HALF + half)
    s1 = tl.load(sin_ptr + t * D + half)
    s2 = tl.load(sin_ptr + t * D + HALF + half)
    row = dqkv_ptr + tok * W

    hq = tl.arange(0, NH_P)[:, None]
    mq = (hq < NH) & (half < HALF)
    g1 = tl.zeros([NH_P, HALF], dtype=tl.float32)
    g2 = tl.zeros([NH_P, HALF], dtype=tl.float32)
    if has_dq:
        src = dq_ptr + (hq * M + tok) * D + half
        g1 = tl.load(src, mask=mq, other=0.0).to(tl.float32)
        g2 = tl.load(src + HALF, mask=mq, other=0.0).to(tl.float32)
    off = hq * D + half
    tl.store(row + off, (g1 * c1 + g2 * s2).to(tl.float16), mask=mq)
    tl.store(row + off + HALF, (g2 * c2 - g1 * s1).to(tl.float16), mask=mq)

    hk = tl.arange(0, NKV_P)[:, None]
    mk = (hk < NKV) & (half < HALF)
    k1 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    k2 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    v1 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    v2 = tl.zeros([NKV_P, HALF], dtype=tl.float32)
    for r in tl.static_range(NH // NKV):
        src = ((hk * (NH // NKV) + r) * M + tok) * D + half
        if has_dk:
            k1 = k1 + tl.load(dk_ptr + src, mask=mk, other=0.0).to(tl.float32)
            k2 = k2 + tl.load(dk_ptr + src + HALF, mask=mk, other=0.0).to(tl.float32)
        if has_dv:
            v1 = v1 + tl.load(dv_ptr + src, mask=mk, other=0.0).to(tl.float32)
            v2 = v2 + tl.load(dv_ptr + src + HALF, mask=mk, other=0.0).to(tl.float32)
    off = NH * D + hk * D + half
    tl.store(row + off, (k1 * c1 + k2 * s2).to(tl.float16), mask=mk)
    tl.store(row + off + HALF, (k2 * c2 - k1 * s1).to(tl.float16), mask=mk)
    off = (NH + NKV) * D + hk * D + half
    tl.store(row + off, v1.to(tl.float16), mask=mk)
    tl.store(row + off + HALF, v2.to(tl.float16), mask=mk)


def _consts(n_heads, n_kv, head_dim):
    return dict(NH=n_heads, NKV=n_kv, HALF=head_dim // 2, NH_P=next_pow2(n_heads),
                NKV_P=next_pow2(n_kv))


class RopePack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, lora_q, lora_v, cos, sin, pos_idx, seq, n_heads, n_kv, head_dim):
        rows = qkv.shape[0]
        dev = qkv.device
        q = torch.empty((n_heads, rows, head_dim), dtype=torch.float16, device=dev)
        k = torch.empty((n_heads, rows, head_dim), dtype=torch.float16, device=dev)
        v = torch.empty((n_heads, rows, head_dim), dtype=torch.float16, device=dev)
        if rows:
            rope_pack_fwd_kernel[(rows,)](
                qkv, lora_q if lora_q is not None else qkv, lora_v if lora_v is not None else qkv,
                cos, sin, pos_idx, q, k, v, rows, seq,
                int(lora_q is not None), int(lora_v is not None),
                **_consts(n_heads, n_kv, head_dim))
        ctx.save_for_backward(cos, sin, pos_idx)
        ctx.dims = (rows, seq, n_heads, n_kv, head_dim)
        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        cos, sin, pos_idx = ctx.saved_tensors
        rows, seq, n_heads, n_kv, head_dim = ctx.dims
        need_qkv, need_lq, need_lv = ctx.needs_input_grad[:3]
        if not (need_qkv or need_lq or need_lv):
            return (None,) * 10
        width = (n_heads + 2 * n_kv) * head_dim
        dqkv = torch.empty((rows, width), dtype=torch.float16, device=pos_idx.device)
        grads = [g.contiguous() if g is not None else None for g in (grad_q, grad_k, grad_v)]
        if rows:
            rope_pack_bwd_kernel[(rows,)](
                grads[0] if grads[0] is not None else dqkv,
                grads[1] if grads[1] is not None else dqkv,
                grads[2] if grads[2] is not None else dqkv,
                cos, sin, pos_idx, dqkv, rows, seq,
                int(grads[0] is not None), int(grads[1] is not None), int(grads[2] is not None),
                **_consts(n_heads, n_kv, head_dim))
        q_cols = n_heads * head_dim
        v_start = (n_heads + n_kv) * head_dim
        return ((dqkv if need_qkv else None),
                (dqkv[:, :q_cols] if need_lq else None),
                (dqkv[:, v_start:] if need_lv else None),
                None, None, None, None, None, None, None)


def rope_pack(qkv, lora_q, lora_v, cos, sin, pos_idx, seq, n_heads, n_kv, head_dim):
    """(q, k, v), each [n_heads, tokens, head_dim] fp16; k and v repeated per query head.
    `pos_idx[i]` is token i's index in the padded [batch * seq] layout (its position is
    pos_idx[i] % seq)."""
    return RopePack.apply(qkv.contiguous(),
                          lora_q.contiguous() if lora_q is not None else None,
                          lora_v.contiguous() if lora_v is not None else None,
                          cos.contiguous(), sin.contiguous(), pos_idx, seq,
                          n_heads, n_kv, head_dim)


def packed_causal_attention(q, k, v, lengths, scale):
    """softmax(q k^T * scale + causal) v for each row on its own tokens.

    q, k, v: [heads, tokens, head_dim] with the rows' tokens consecutive (lengths sum to
    tokens). Returns [tokens, heads * head_dim]. PyTorch SDPA (memory-efficient kernel on a T4)
    per row with is_causal: the same attention base Soup computes with its padding mask, without
    the padded positions."""
    outs = []
    for qb, kb, vb in zip(q.split(lengths, dim=1), k.split(lengths, dim=1),
                          v.split(lengths, dim=1)):
        out = torch.nn.functional.scaled_dot_product_attention(
            qb.unsqueeze(0), kb.unsqueeze(0), vb.unsqueeze(0), dropout_p=0.0, is_causal=True,
            scale=scale)
        outs.append(out[0].transpose(0, 1))
    return torch.cat(outs, dim=0).reshape(q.shape[1], q.shape[0] * q.shape[2])
