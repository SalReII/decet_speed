"""The output head and everything Soup computes from its logits, without full logits.

In base Soup the head writes fp16 logits [batch, seq, 151936] (622 MB at 2048 tokens); accelerate
converts them to fp32 (1.2 GB), the loss upcasts them again and keeps log-softmax for backward
(1.2 GB), TRL makes a shifted fp32 copy for the token accuracy and runs a chunked log-softmax for
the entropy, Soup's rewind recorder runs one more cross-entropy per row, and backward builds
another fp32 and fp16 gradient of the same shape. That is about 4 GB at the peak and seven
passes over the logits.

Here the head runs over the real (non-pad) tokens in chunks of 256 rows, labelled rows first:
1. cuBLAS: logits = h @ W^T for the chunk, fp16, exactly the GEMM base Soup runs under autocast;
2. Triton, one program per row, two sweeps over the 151,936 logits:
   sweep 1 (online, fp32): running max, sum of exp, sum of exp * z and the first argmax, which
   give logsumexp, entropy = lse - sum(p z), the token loss lse - z[label] and whether the
   argmax is the label;
   sweep 2 (labelled rows only): overwrites the logits in place with the gradient
   (softmax - onehot) * scale in fp16;
3. cuBLAS: dh = dlogits @ W for the labelled rows of the chunk.
The chunk's logits are dropped right away, so the head costs one [256, 151936] fp16 buffer.
The gradient is taken in forward (the head is frozen) and scaled by the loss scale in backward.
"""

import math

import torch
import triton
import triton.language as tl

CHUNK_ROWS = 1024  # speed version: fewer, larger head GEMMs (a 311 MB fp16 chunk)
VOCAB_BLOCK = 4096
IGNORE_INDEX = -100


@triton.jit(do_not_specialize=["grad_scale"])
def head_stats_grad_kernel(logits_ptr, target_ptr, entropy_ptr, loss_ptr, correct_ptr, grad_scale,
                           V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * V
    target = tl.load(target_ptr + row).to(tl.int32)
    offs = tl.arange(0, BLOCK)

    run_max = -float("inf")
    run_sum = 0.0
    run_dot = 0.0
    best = -float("inf")
    best_idx = 0
    for start in range(0, V, BLOCK):
        cols = start + offs
        inside = cols < V
        z = tl.load(base + cols, mask=inside, other=-float("inf")).to(tl.float32)
        block_max = tl.max(z, axis=0)
        block_arg = tl.argmax(z, axis=0, tie_break_left=True)
        take = block_max > best
        best_idx = tl.where(take, start + block_arg, best_idx)
        best = tl.where(take, block_max, best)
        new_max = tl.maximum(run_max, block_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.exp(z - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(tl.where(inside, e * z, 0.0), axis=0)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    tl.store(entropy_ptr + row, lse - run_dot / run_sum)
    labelled = target != -100
    safe_target = tl.where(labelled, target, 0)
    z_target = tl.load(base + safe_target).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - z_target, 0.0))
    tl.store(correct_ptr + row, (labelled & (best_idx == target)).to(tl.int32))
    if labelled:
        for start in range(0, V, BLOCK):
            cols = start + offs
            inside = cols < V
            z = tl.load(base + cols, mask=inside, other=0.0).to(tl.float32)
            grad = tl.exp(z - lse) - (cols == target).to(tl.float32)
            tl.store(base + cols, (grad * grad_scale).to(tl.float16), mask=inside)


def _grad_unit(denominator):
    """A power of two G with G / denominator about 64: the fp16 head gradient is held at the
    magnitude base Soup's GradScaler gives it (65536 / tokens), and rescaling by a power of two
    is exact."""
    exponent = int(round(math.log2(max(float(denominator), 1e-6)))) + 6
    return float(2 ** max(0, min(16, exponent)))


class FusedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, denominator):
        rows, width = hidden.shape
        vocab = weight.shape[0]
        dev = hidden.device
        labelled = targets != IGNORE_INDEX
        order = torch.argsort((~labelled).to(torch.int8), stable=True)
        n_labelled, denom = _host_counts(labelled, denominator, dev)
        h = hidden.index_select(0, order)
        tgt = targets.index_select(0, order).contiguous()
        entropy = torch.empty(rows, dtype=torch.float32, device=dev)
        token_loss = torch.empty(rows, dtype=torch.float32, device=dev)
        correct = torch.empty(rows, dtype=torch.int32, device=dev)
        want_grad = ctx.needs_input_grad[0]
        unit = _grad_unit(denom)
        grad_sorted = torch.empty((n_labelled, width), dtype=torch.float16, device=dev) \
            if want_grad else None
        for start in range(0, rows, CHUNK_ROWS):
            stop = min(rows, start + CHUNK_ROWS)
            logits = torch.matmul(h[start:stop], weight.t())
            head_stats_grad_kernel[(stop - start,)](
                logits, tgt[start:stop], entropy[start:], token_loss[start:], correct[start:],
                unit / max(denom, 1e-6), V=vocab, BLOCK=VOCAB_BLOCK, num_warps=8)
            lab_stop = min(stop, n_labelled)
            if want_grad and lab_stop > start:
                torch.matmul(logits[:lab_stop - start], weight, out=grad_sorted[start:lab_stop])
            del logits
        total = token_loss.sum()
        if torch.is_tensor(denominator):
            loss = total / denominator.to(dev)
        else:
            loss = total / denominator
        if want_grad:
            grad = torch.zeros((rows, width), dtype=torch.float16, device=dev)
            grad.index_copy_(0, order[:n_labelled], grad_sorted)
            ctx.save_for_backward(grad)
        ctx.unit = unit
        unsort = torch.empty_like(order)
        unsort[order] = torch.arange(rows, device=dev)
        stats = (entropy[unsort], token_loss[unsort], correct[unsort].bool())
        ctx.mark_non_differentiable(*stats)
        return (loss, *stats)

    @staticmethod
    def backward(ctx, grad_loss, *_unused):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        (grad,) = ctx.saved_tensors
        factor = grad_loss.to(torch.float32) / ctx.unit
        return (grad.to(torch.float32) * factor).to(torch.float16), None, None, None


def _host_counts(labelled, denominator, device):
    """(labelled rows, loss denominator) on the host, with one synchronisation."""
    if torch.is_tensor(denominator):
        pair = torch.stack([labelled.sum().to(torch.float64),
                            denominator.to(device=device, dtype=torch.float64).reshape(())])
        n_labelled, denom = pair.tolist()
    else:
        n_labelled, denom = labelled.sum().item(), float(denominator)
    return int(n_labelled), float(denom)


def fused_head_loss(hidden16, weight16, targets, denominator):
    """(loss, token entropy [M], token loss [M], token correct [M]) for rows of `hidden16`.

    `targets[i]` is the label row i predicts (IGNORE_INDEX for none); `denominator` is what
    the summed loss is divided by (num_items_in_batch, or the labelled count)."""
    return FusedHeadLoss.apply(hidden16.contiguous(), weight16, targets.contiguous(), denominator)
