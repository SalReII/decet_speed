"""The Qwen2 MLP, down(silu(gate(x)) * up(x)), with frozen NF4 weights, as one autograd node.

What base Soup does on a T4: bitsandbytes returns each projection in the input's dtype, and the
input is the fp32 norm output, so gate and up come back as fp32 [tokens, 8960] tensors; silu,
the product and the autograd graph then keep three fp32 [tokens, 8960] tensors per layer
(about 220 MB per layer at 2048 tokens, 6 GB over 28 layers) and run every element-wise op in
fp32 over them.

Here:
- gate and up are one GEMM against both weights unpacked into one [17920, 1536] buffer;
- one Triton kernel reads the fp16 [tokens, 2 * 8960] result and writes silu(g) * u in fp16
  (computed in fp32: gate/up were fp16 GEMM outputs in base Soup too, so the fp32 copies held
  no extra information, and the product is rounded once to fp16 exactly where base Soup rounds
  it for down_proj);
- nothing of size [tokens, 8960] is saved: backward recomputes gate/up from the saved fp16
  input (one GEMM) and a second Triton kernel turns the incoming gradient into the gate/up
  gradients in place, in the same buffer, so the input gradient is again a single GEMM.
"""

import torch
import triton
import triton.language as tl

from ._common import triton_ok
from .nf4 import unpack_concat

RECOMPUTE_GATE_UP = True  # keep only the [tokens, 1536] input; recompute gate/up in backward
COLS_PER_PROGRAM = 1024


@triton.jit
def gated_silu_fwd_kernel(gu_ptr, act_ptr, I: tl.constexpr, COLS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    keep = col < I
    pair = gu_ptr + row * (2 * I)
    gate = tl.load(pair + col, mask=keep, other=0.0).to(tl.float32)
    up = tl.load(pair + I + col, mask=keep, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(act_ptr + row * I + col, (silu * up).to(tl.float16), mask=keep)


@triton.jit
def gated_silu_bwd_inplace_kernel(dact_ptr, gu_ptr, I: tl.constexpr, COLS: tl.constexpr):
    """Overwrites [gate | up] with [d gate | d up] for d act = dact."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * COLS + tl.arange(0, COLS)
    keep = col < I
    pair = gu_ptr + row * (2 * I)
    dact = tl.load(dact_ptr + row * I + col, mask=keep, other=0.0).to(tl.float32)
    gate = tl.load(pair + col, mask=keep, other=0.0).to(tl.float32)
    up = tl.load(pair + I + col, mask=keep, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    d_silu = dact * up
    d_gate = d_silu * sig * (1.0 + gate * (1.0 - sig))
    d_up = dact * (gate * sig)
    tl.store(pair + col, d_gate.to(tl.float16), mask=keep)
    tl.store(pair + I + col, d_up.to(tl.float16), mask=keep)


def gated_silu(gu):
    rows, two_i = gu.shape
    inter = two_i // 2
    act = torch.empty((rows, inter), dtype=torch.float16, device=gu.device)
    if rows:
        gated_silu_fwd_kernel[(rows, triton.cdiv(inter, COLS_PER_PROGRAM))](
            gu, act, I=inter, COLS=COLS_PER_PROGRAM)
    return act


def gated_silu_backward_(dact, gu):
    rows, two_i = gu.shape
    inter = two_i // 2
    if rows:
        gated_silu_bwd_inplace_kernel[(rows, triton.cdiv(inter, COLS_PER_PROGRAM))](
            dact.contiguous(), gu, I=inter, COLS=COLS_PER_PROGRAM)
    return gu


class FusedNF4MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_w, up_w, down_w):
        gate_up = unpack_concat((gate_w, up_w), x.device)
        gu = torch.matmul(x, gate_up.t())
        del gate_up
        act = gated_silu(gu)
        if RECOMPUTE_GATE_UP:
            del gu
            ctx.save_for_backward(x)
        else:
            ctx.save_for_backward(x, gu)
        down = unpack_concat((down_w,), x.device)
        out = torch.matmul(act, down.t())
        ctx.weights = (gate_w, up_w, down_w)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        gate_w, up_w, down_w = ctx.weights
        saved = ctx.saved_tensors
        x = saved[0]
        grad_out = grad_out.to(torch.float16)
        down = unpack_concat((down_w,), grad_out.device)
        dact = torch.matmul(grad_out, down)
        del down
        gate_up = unpack_concat((gate_w, up_w), grad_out.device)
        gu = torch.matmul(x, gate_up.t()) if RECOMPUTE_GATE_UP else saved[1]
        gated_silu_backward_(dact, gu)
        del dact
        grad_x = torch.matmul(gu, gate_up)
        return grad_x, None, None, None


class FusedDenseMLP(torch.autograd.Function):
    """The same MLP on fp16 weights unpacked once ([gate; up] stacked, and down). Speed path:
    the fp16 [tokens, 2 * 8960] gate/up GEMM output is kept for backward instead of being
    recomputed, so backward is two GEMMs and one Triton kernel."""

    @staticmethod
    def forward(ctx, x, gate_up, down):
        gu = torch.matmul(x, gate_up.t())
        act = gated_silu(gu)
        out = torch.matmul(act, down.t())
        ctx.save_for_backward(gu)
        ctx.weights = (gate_up, down)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        (gu,) = ctx.saved_tensors
        gate_up, down = ctx.weights
        dact = torch.matmul(grad_out.to(torch.float16), down)
        gated_silu_backward_(dact, gu)  # gu now holds [d gate | d up]
        del dact
        return torch.matmul(gu, gate_up), None, None


def fused_mlp(x, gate_w, up_w, down_w, dense=None):
    assert triton_ok(x)
    if dense is not None:
        return FusedDenseMLP.apply(x.contiguous(), dense[0], dense[1])
    return FusedNF4MLP.apply(x.contiguous(), gate_w, up_w, down_w)
