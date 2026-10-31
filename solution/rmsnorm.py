"""Residual add + RMSNorm in one Triton pass, forward and backward.

Base Soup keeps the norms in fp32 (`prepare_model_for_kbit_training`) and runs Qwen2RMSNorm as
six PyTorch ops (cast, pow, mean, rsqrt, two muls) after a separate residual add, all on fp32
[batch, seq, 1536] tensors, and every consumer then casts the fp32 result to fp16 again.

Here one program per token row:
  s = x + delta            (the residual stream, fp32; delta is the fp16 output of the block)
  y = w * s * rsqrt(mean(s^2) + eps)
and writes y once in fp16 for the GEMMs and, when asked, once in fp32 into the padded
[batch * seq, hidden] layout that the LoRA dropout reads (so the dropout sees exactly the
tensor shape, dtype and values of base Soup and draws the same random mask).
Backward is one program per row as well; it sums the fp16 gradient of the GEMM path and the
fp32 gradient of the LoRA path in fp32 and adds the residual gradient.
"""

import torch
import triton
import triton.language as tl

from ._common import next_pow2


@triton.jit(do_not_specialize=["eps", "has_delta", "write_pad"])
def rmsnorm_fwd_kernel(x_ptr, delta_ptr, w_ptr, s_ptr, y_ptr, pad_ptr, pad_idx_ptr, rstd_ptr,
                       eps, has_delta, write_pad, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    base = row.to(tl.int64) * H
    s = tl.load(x_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
    if has_delta:
        s = s + tl.load(delta_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
        tl.store(s_ptr + base + cols, s, mask=inside)
    variance = tl.sum(s * s, axis=0) / H
    rstd = 1.0 / tl.sqrt(variance + eps)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    y = (s * rstd) * w
    tl.store(y_ptr + base + cols, y.to(tl.float16), mask=inside)
    if write_pad:
        dest = tl.load(pad_idx_ptr + row).to(tl.int64) * H
        tl.store(pad_ptr + dest + cols, y, mask=inside)
    tl.store(rstd_ptr + row, rstd)


@triton.jit(do_not_specialize=["has_dy", "has_padgrad", "has_ds", "write_ddelta"])
def rmsnorm_bwd_kernel(s_ptr, rstd_ptr, w_ptr, dy_ptr, dypad_ptr, pad_idx_ptr, ds_ptr, dx_ptr,
                       ddelta_ptr, has_dy, has_padgrad, has_ds, write_ddelta,
                       H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    base = row.to(tl.int64) * H
    g = tl.zeros([BLOCK], dtype=tl.float32)
    if has_dy:
        g = g + tl.load(dy_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
    if has_padgrad:
        src = tl.load(pad_idx_ptr + row).to(tl.int64) * H
        g = g + tl.load(dypad_ptr + src + cols, mask=inside, other=0.0).to(tl.float32)
    g = g * tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    normed = tl.load(s_ptr + base + cols, mask=inside, other=0.0).to(tl.float32) * rstd
    proj = tl.sum(g * normed, axis=0) / H
    dx = (g - normed * proj) * rstd
    if has_ds:
        dx = dx + tl.load(ds_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(dx_ptr + base + cols, dx, mask=inside)
    if write_ddelta:
        tl.store(ddelta_ptr + base + cols, dx.to(tl.float16), mask=inside)


def _index_dummy(device):
    return torch.zeros(1, dtype=torch.long, device=device)


def _launch_fwd(x, delta, weight, eps, pad_idx, padded_rows):
    rows, hidden = x.shape
    s = torch.empty_like(x) if delta is not None else x
    y = torch.empty((rows, hidden), dtype=torch.float16, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if pad_idx is not None:
        ypad = torch.zeros((padded_rows, hidden), dtype=torch.float32, device=x.device)
    else:
        ypad = torch.empty((0, hidden), dtype=torch.float32, device=x.device)
    if rows:
        # Unused pointers get dummies of the real dtype: one compiled variant for every norm.
        rmsnorm_fwd_kernel[(rows,)](
            x, delta if delta is not None else y, weight, s, y, ypad,
            pad_idx if pad_idx is not None else _index_dummy(x.device), rstd, float(eps),
            int(delta is not None), int(pad_idx is not None),
            H=hidden, BLOCK=next_pow2(hidden), num_warps=8)
    return s, y, ypad, rstd


def _launch_bwd(s, rstd, weight, pad_idx, grad_s, grad_y, grad_ypad, want_delta):
    rows, hidden = s.shape
    dx = torch.empty_like(s)
    half_dummy = torch.empty(1, dtype=torch.float16, device=s.device)
    ddelta = torch.empty((rows, hidden), dtype=torch.float16, device=s.device) \
        if want_delta else half_dummy
    has_pad = grad_ypad is not None and pad_idx is not None and grad_ypad.numel() > 0
    if rows:
        rmsnorm_bwd_kernel[(rows,)](
            s, rstd, weight,
            grad_y.contiguous() if grad_y is not None else half_dummy,
            grad_ypad.contiguous() if has_pad else s,
            pad_idx if has_pad else _index_dummy(s.device),
            grad_s.contiguous() if grad_s is not None else s,
            dx, ddelta,
            int(grad_y is not None), int(has_pad), int(grad_s is not None), int(want_delta),
            H=hidden, BLOCK=next_pow2(hidden), num_warps=8)
    return dx, (ddelta if want_delta else None)


class AddRMSNorm(torch.autograd.Function):
    """(s, y16, ypad32) = (x + delta, norm(x + delta) in fp16, the same in fp32 at padded rows)."""

    @staticmethod
    def forward(ctx, x, delta, weight, eps, pad_idx, padded_rows):
        s, y, ypad, rstd = _launch_fwd(x, delta, weight, eps, pad_idx, padded_rows)
        ctx.save_for_backward(s, rstd, weight, pad_idx)
        return s, y, ypad

    @staticmethod
    def backward(ctx, grad_s, grad_y, grad_ypad):
        s, rstd, weight, pad_idx = ctx.saved_tensors
        need_x, need_delta = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_x or need_delta):
            return None, None, None, None, None, None
        dx, ddelta = _launch_bwd(s, rstd, weight, pad_idx, grad_s, grad_y, grad_ypad, need_delta)
        return (dx if need_x else None), ddelta, None, None, None, None


class RMSNormNoAdd(torch.autograd.Function):
    """(y16, ypad32) = norm(x) without a residual add (the first layer's input norm)."""

    @staticmethod
    def forward(ctx, x, weight, eps, pad_idx, padded_rows):
        _s, y, ypad, rstd = _launch_fwd(x, None, weight, eps, pad_idx, padded_rows)
        ctx.save_for_backward(x, rstd, weight, pad_idx)
        return y, ypad

    @staticmethod
    def backward(ctx, grad_y, grad_ypad):
        x, rstd, weight, pad_idx = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        dx, _ = _launch_bwd(x, rstd, weight, pad_idx, None, grad_y, grad_ypad, False)
        return dx, None, None, None, None


def add_rmsnorm(x, delta, weight, eps, pad_idx=None, padded_rows=0):
    """Returns (s, y16, ypad32); with delta=None, s is x itself."""
    x = x.contiguous()
    if delta is None:
        y, ypad = RMSNormNoAdd.apply(x, weight, eps, pad_idx, padded_rows)
        return x, y, ypad
    return AddRMSNorm.apply(x, delta.contiguous(), weight, eps, pad_idx, padded_rows)
