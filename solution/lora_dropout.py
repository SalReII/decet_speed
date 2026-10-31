"""The LoRA input, dropout(y) in fp16 for lora_A, from the fp32 norm output of the real tokens.

peft runs `lora_A(dropout(x))` on the fp32 norm output of the padded batch: PyTorch's fused
dropout reads and writes the padded fp32 tensor, autocast casts the result to fp16, and backward
runs the dropout mask over a padded fp32 gradient. With the padding removed that also needs a
row gather and a scatter back.

Here PyTorch still draws the dropout mask, with the same call on a tensor of the same shape and
dtype as base Soup's input, so the generator produces the same mask in the same order (q, then v,
in every layer). One Triton program per real token then writes fp16(y * mask * 1/(1-p)), read
from the token's padded row of the mask, which is what peft + autocast feed lora_A; backward
writes the fp32 gradient mask * 1/(1-p) * g for the token, which the norm backward adds in.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from ._common import next_pow2


@triton.jit(do_not_specialize=["scale"])
def lora_dropout_fwd_kernel(y_ptr, keep_ptr, pos_ptr, out_ptr, scale,
                            H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    src = tl.load(pos_ptr + row).to(tl.int64) * H
    keep = tl.load(keep_ptr + src + cols, mask=inside, other=0)
    y = tl.load(y_ptr + row * H + cols, mask=inside, other=0.0)
    out = tl.where(keep != 0, y * scale, 0.0)
    tl.store(out_ptr + row * H + cols, out.to(tl.float16), mask=inside)


@triton.jit(do_not_specialize=["scale"])
def lora_dropout_bwd_kernel(grad_ptr, keep_ptr, pos_ptr, dy_ptr, scale,
                            H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    src = tl.load(pos_ptr + row).to(tl.int64) * H
    keep = tl.load(keep_ptr + src + cols, mask=inside, other=0)
    g = tl.load(grad_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(dy_ptr + row * H + cols, tl.where(keep != 0, g * scale, 0.0), mask=inside)


def dropout_scale(p):
    """The fp32 factor PyTorch's fused dropout multiplies kept values by: 1 / (keep prob)."""
    return float(np.float32(1.0 / np.float64(np.float32(1.0 - p))))


def draw_mask(shape, p, device):
    """The keep mask `F.dropout(x, p, training=True)` would draw for an fp32 `x` of `shape`,
    as uint8, consuming the generator exactly as that call does."""
    if device.type == "cuda":
        x = torch.empty(shape, dtype=torch.float32, device=device)  # values are never read
        return torch.ops.aten.native_dropout.default(x, p, True)[1].view(torch.uint8)
    return (torch.nn.functional.dropout(torch.ones(shape, dtype=torch.float32, device=device),
                                        p, True) != 0).view(torch.uint8)


class LoraDropoutInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, keep, pos_idx, scale):
        rows, hidden = y.shape
        out = torch.empty((rows, hidden), dtype=torch.float16, device=y.device)
        if rows:
            lora_dropout_fwd_kernel[(rows,)](y, keep, pos_idx, out, scale, H=hidden,
                                             BLOCK=next_pow2(hidden), num_warps=8)
        ctx.save_for_backward(keep, pos_idx)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        keep, pos_idx = ctx.saved_tensors
        rows, hidden = grad.shape
        dy = torch.empty((rows, hidden), dtype=torch.float32, device=grad.device)
        if rows:
            lora_dropout_bwd_kernel[(rows,)](grad.contiguous(), keep, pos_idx, dy, ctx.scale,
                                             H=hidden, BLOCK=next_pow2(hidden), num_warps=8)
        return dy, None, None, None


def lora_dropout_input(y32, keep, pos_idx, p):
    """fp16 dropout(y32) for the real tokens; `keep` is the [batch * seq, hidden] padded mask."""
    return LoraDropoutInput.apply(y32.contiguous(), keep, pos_idx, dropout_scale(p))
