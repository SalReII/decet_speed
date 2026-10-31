"""NF4 (bitsandbytes 4-bit, double quant) weight unpacking in Triton, and the frozen 4-bit linear
that uses it.

bitsandbytes stores every frozen projection as NF4: two 4-bit codes per byte (high nibble first),
one absmax per 64 values, and with double quant ("nested") the absmax itself is an 8-bit code
into a 256-entry table, scaled per 256 absmax blocks and shifted by one offset:

    absmax[b] = code2[absmax_u8[b]] * absmax2[b // 256] + offset
    w[i]      = nf4[q[i]] * absmax[i // 64]          (fp32, rounded once to fp16)

bitsandbytes does this in two CUDA launches (absmax first, then the 4-bit values) and then runs
`F.linear`. Here one Triton program unpacks 32 quantization blocks (1024 bytes) into 2048 fp16 values,
looking each block's absmax up once, and several projections are unpacked into one buffer so that, for example,
q/k/v or gate/up run as a single GEMM. The arithmetic is the same fp32 product, rounded once to
fp16, so the values are the ones bitsandbytes produces.
"""

import torch
import triton
import triton.language as tl

from ._common import triton_ok

BLOCKS_PER_PROGRAM = 32  # absmax blocks per program: 32 x 64 values = 2048 fp16 outputs


@triton.jit(do_not_specialize=["n_blocks"])
def nf4_unpack_kernel(packed_ptr, absmax_u8_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                      out_ptr, n_blocks,
                      QBLOCK: tl.constexpr, QBLOCK2: tl.constexpr, NBLK: tl.constexpr):
    """A [NBLK, QBLOCK // 2] tile of packed bytes per program: the absmax of each row (one
    quantization block) is looked up once and broadcast, so per byte only the two 16-entry NF4
    lookups remain."""
    pid = tl.program_id(0)
    blk = pid * NBLK + tl.arange(0, NBLK)
    keep = blk < n_blocks
    code = tl.load(absmax_u8_ptr + blk, mask=keep, other=0).to(tl.int32)
    scale = tl.load(code2_ptr + code, mask=keep, other=0.0) \
        * tl.load(absmax2_ptr + blk // QBLOCK2, mask=keep, other=0.0)
    scale = scale + tl.load(offset_ptr)
    byte_idx = blk.to(tl.int64)[:, None] * (QBLOCK // 2) + tl.arange(0, QBLOCK // 2)[None, :]
    packed = tl.load(packed_ptr + byte_idx, mask=keep[:, None], other=0).to(tl.int32)
    first = tl.load(nf4_ptr + (packed >> 4)) * scale[:, None]
    second = tl.load(nf4_ptr + (packed & 15)) * scale[:, None]
    values = tl.interleave(first, second)
    out_idx = blk.to(tl.int64)[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
    tl.store(out_ptr + out_idx, values.to(tl.float16), mask=keep[:, None])


def _quant_state(weight):
    return getattr(weight, "quant_state", None)


def supports_fast_unpack(weight):
    """True for the layout the kernel handles: NF4, nested absmax with 256-blocks, even sizes."""
    state = _quant_state(weight)
    if state is None or getattr(state, "quant_type", None) != "nf4" or not state.nested:
        return False
    if state.state2 is None or state.state2.blocksize != 256 or state.blocksize % 2:
        return False
    shape = tuple(state.shape)
    if len(shape) != 2 or (shape[0] * shape[1]) % state.blocksize:
        return False
    return weight.data.dtype == torch.uint8 and weight.data.is_contiguous()


def _on_device(weight, device):
    state = _quant_state(weight)
    tensors = (weight.data, state.absmax, state.code, state.state2.code, state.state2.absmax)
    return all(t.device == device for t in tensors) and \
        (not torch.is_tensor(state.offset) or state.offset.device == device)


def unpack_into(weight, out):
    """Write the fp16 values of one bitsandbytes 4-bit weight ([N, K]) into `out` ([N, K])."""
    state = _quant_state(weight)
    n_bytes = weight.data.numel()
    if triton_ok(out) and supports_fast_unpack(weight) and _on_device(weight, out.device):
        n_blocks = 2 * n_bytes // state.blocksize
        grid = (triton.cdiv(n_blocks, BLOCKS_PER_PROGRAM),)
        nf4_unpack_kernel[grid](
            weight.data, state.absmax, state.state2.code, state.state2.absmax,
            state.offset if torch.is_tensor(state.offset)
            else torch.tensor([float(state.offset)], device=out.device),
            state.code, out, n_blocks,
            QBLOCK=state.blocksize, QBLOCK2=state.state2.blocksize, NBLK=BLOCKS_PER_PROGRAM,
            enable_fp_fusion=False)  # mul, add, mul rounded separately, as bitsandbytes does
        return out
    import bitsandbytes.functional as bnb_functional

    out.copy_(bnb_functional.dequantize_4bit(weight.data, state).to(out.dtype).view_as(out))
    return out


def weight_shape(weight):
    return tuple(_quant_state(weight).shape)


def unpack_concat(weights, device):
    """fp16 [sum N_i, K]: the weights stacked by rows, each unpacked in place."""
    rows = [weight_shape(w)[0] for w in weights]
    cols = weight_shape(weights[0])[1]
    out = torch.empty((sum(rows), cols), dtype=torch.float16, device=device)
    start = 0
    for weight, n in zip(weights, rows):
        unpack_into(weight, out[start:start + n])
        start += n
    return out


class FrozenNF4Linear(torch.autograd.Function):
    """y = x @ W^T (+ b) for frozen 4-bit weights stacked by rows, fp16 in and out.

    Nothing large is saved: the weights are unpacked again in backward (as bitsandbytes does),
    and no input is kept because W does not train."""

    @staticmethod
    def forward(ctx, x, bias, weights):
        weight = unpack_concat(weights, x.device)
        out = torch.nn.functional.linear(x, weight, bias)
        ctx.weights = weights
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight = unpack_concat(ctx.weights, grad_out.device)
            grad_x = grad_out.to(torch.float16) @ weight
        return grad_x, None, None


class FrozenDenseLinear(torch.autograd.Function):
    """y = x @ W^T (+ b) with W an fp16 copy unpacked once by nf4_unpack_kernel (the same values
    every per-call unpack produces). Saves no activation: W is frozen."""

    @staticmethod
    def forward(ctx, x, bias, weight):
        ctx.weight = weight
        return torch.nn.functional.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = grad_out.to(torch.float16) @ ctx.weight if ctx.needs_input_grad[0] else None
        return grad_x, None, None


def frozen_linear(x, weights, bias=None, dense=None):
    if dense is not None:
        return FrozenDenseLinear.apply(x, bias, dense)
    return FrozenNF4Linear.apply(x, bias, tuple(weights))
