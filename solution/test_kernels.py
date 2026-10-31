"""Our kernels against plain PyTorch, forward and backward.

On a GPU:      python -m pytest solution -q
Without one:   TRITON_INTERPRET=1 python -m pytest solution -q   (Triton's CPU interpreter)
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
SMALL = DEV == "cpu"  # the interpreter is slow: smaller shapes


def close(actual, expected, tol):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol)


def rel_err(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


def quantized(n_out, n_in, seed=0):
    import bitsandbytes.functional as bnbf

    torch.manual_seed(seed)
    w = (torch.randn(n_out, n_in, device=DEV) * 0.05).to(torch.float16)
    packed, state = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=True,
                                       quant_type="nf4")

    class P:  # what Params4bit exposes: .data and .quant_state
        pass

    p = P()
    p.data, p.quant_state = packed, state
    return p, bnbf.dequantize_4bit(packed, state).to(torch.float16)


@pytest.mark.parametrize("shape", [(128, 192), (96, 640)])
def test_nf4_unpack_matches_bitsandbytes(shape):
    from solution.nf4 import supports_fast_unpack, unpack_concat

    a, ref_a = quantized(*shape, seed=1)
    b, ref_b = quantized(shape[0] // 2, shape[1], seed=2)
    assert supports_fast_unpack(a)
    out = unpack_concat((a, b), torch.device(DEV))
    assert torch.equal(out, torch.cat([ref_a, ref_b]))


@pytest.mark.parametrize("with_delta,with_pad", [(True, True), (True, False), (False, True)])
def test_add_rmsnorm(with_delta, with_pad):
    from solution.rmsnorm import add_rmsnorm

    torch.manual_seed(0)
    rows, hidden, padded = (7, 96, 10) if SMALL else (300, 1536, 512)
    x = torch.randn(rows, hidden, device=DEV, requires_grad=True)
    delta = torch.randn(rows, hidden, device=DEV, dtype=torch.float16, requires_grad=True) \
        if with_delta else None
    w = torch.rand(hidden, device=DEV) + 0.5
    pad_idx = torch.randperm(padded, device=DEV)[:rows] if with_pad else None
    s, y, ypad = add_rmsnorm(x, delta, w, 1e-6, pad_idx, padded)

    xr = x.detach().clone().requires_grad_(True)
    dr = delta.detach().clone().requires_grad_(True) if with_delta else None
    sr = xr + dr.float() if with_delta else xr
    yr = w * (sr * torch.rsqrt(sr.pow(2).mean(-1, keepdim=True) + 1e-6))
    close(y, yr, 2e-3)
    if with_pad:
        close(ypad[pad_idx], yr, 1e-5)
    gy = torch.randn(rows, hidden, device=DEV, dtype=torch.float16)
    gpad = torch.randn(padded, hidden, device=DEV) if with_pad else None
    gs = torch.randn(rows, hidden, device=DEV) if with_delta else None
    outs = [y] + ([ypad] if with_pad else []) + ([s] if with_delta else [])
    grads = [gy] + ([gpad] if with_pad else []) + ([gs] if with_delta else [])
    torch.autograd.backward(outs, grads)
    total = gy.float() + (gpad[pad_idx] if with_pad else 0)
    ref_outs = [yr] + ([sr] if with_delta else [])
    ref_grads = [total] + ([gs] if with_delta else [])
    torch.autograd.backward(ref_outs, ref_grads)
    close(x.grad, xr.grad, 1e-4)
    if with_delta:
        close(delta.grad, dr.grad, 2e-3)


def test_gated_silu():
    from solution.mlp import gated_silu, gated_silu_backward_

    torch.manual_seed(0)
    rows, inter = (5, 1500) if SMALL else (333, 8960)
    gu = (torch.randn(rows, 2 * inter, device=DEV) * 3).to(torch.float16)
    g, u = gu[:, :inter].float().requires_grad_(True), gu[:, inter:].float().requires_grad_(True)
    ref = torch.nn.functional.silu(g) * u
    close(gated_silu(gu), ref, 2e-3)
    dact = torch.randn(rows, inter, device=DEV, dtype=torch.float16)
    ref.backward(dact.float())
    out = gated_silu_backward_(dact, gu.clone())
    close(out[:, :inter], g.grad, 5e-3)
    close(out[:, inter:], u.grad, 5e-3)


def test_fused_mlp_matches_dequantized_reference():
    from solution.mlp import fused_mlp

    hidden, inter, rows = (64, 128, 9) if SMALL else (1536, 8960, 200)
    gate, gate_ref = quantized(inter, hidden, 3)
    up, up_ref = quantized(inter, hidden, 4)
    down, down_ref = quantized(hidden, inter, 5)
    torch.manual_seed(0)
    x = torch.randn(rows, hidden, device=DEV, dtype=torch.float16, requires_grad=True)
    out = fused_mlp(x, gate, up, down)
    xr = x.detach().float().requires_grad_(True)
    ref = (torch.nn.functional.silu(xr @ gate_ref.float().t()) * (xr @ up_ref.float().t())) \
        @ down_ref.float().t()
    # fp16 GEMMs over 1536 / 8960 terms vs an fp32 reference: compare relative error norms
    assert rel_err(out, ref) < 5e-3
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    assert rel_err(x.grad, xr.grad) < 1e-2


def test_rope_pack_and_packed_attention():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    from solution.rope import packed_causal_attention, rope_pack

    torch.manual_seed(0)
    batch, seq, nh, nkv, d = (2, 5, 4, 2, 8) if SMALL else (4, 77, 12, 2, 128)
    lengths = [seq, seq - 2] + ([seq - 30, 9] if not SMALL else [])
    batch = len(lengths)
    pos_idx = torch.cat([b * seq + torch.arange(n) for b, n in enumerate(lengths)]).to(DEV)
    rows = pos_idx.numel()
    width = (nh + 2 * nkv) * d
    qkv = torch.randn(rows, width, device=DEV, dtype=torch.float16, requires_grad=True)
    lq = torch.randn(rows, nh * d, device=DEV, dtype=torch.float16, requires_grad=True)
    lv = torch.randn(rows, nkv * d, device=DEV, dtype=torch.float16, requires_grad=True)
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=DEV).float() / d))
    freqs = torch.arange(seq, device=DEV).float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos(), emb.sin()
    q, k, v = rope_pack(qkv, lq, lv, cos, sin, pos_idx, seq, nh, nkv, d)
    out = packed_causal_attention(q, k, v, lengths, d ** -0.5)

    # reference: base Soup's way, padded [B, H, T, D], repeat_kv, causal + padding mask
    leaves = [t.detach().float().requires_grad_(True) for t in (qkv, lq, lv)]
    fq = leaves[0][:, :nh * d] + leaves[1]
    fk = leaves[0][:, nh * d:(nh + nkv) * d]
    fv = leaves[0][:, (nh + nkv) * d:] + leaves[2]

    def padded(flat, heads):
        full = torch.zeros(batch * seq, heads * d, device=DEV).index_copy(0, pos_idx, flat)
        return full.view(batch, seq, heads, d).transpose(1, 2)

    rq, rk = apply_rotary_pos_emb(padded(fq, nh), padded(fk, nkv), cos[None], sin[None])
    rv = padded(fv, nkv)
    to_packed = lambda t: t.transpose(0, 1).reshape(nh, batch * seq, d)[:, pos_idx]  # noqa: E731
    close(q, to_packed(rq), 2e-3)
    close(k, to_packed(repeat_kv(rk, nh // nkv)), 2e-3)
    close(v, to_packed(repeat_kv(rv, nh // nkv)), 2e-3)
    valid = torch.zeros(batch, seq, dtype=torch.bool, device=DEV).view(-1)
    valid[pos_idx] = True
    valid = valid.view(batch, seq)
    causal = torch.ones(seq, seq, dtype=torch.bool, device=DEV).tril()
    attn_mask = causal[None, None] & valid[:, None, None, :]
    ref = torch.nn.functional.scaled_dot_product_attention(
        rq, repeat_kv(rk, nh // nkv), repeat_kv(rv, nh // nkv), attn_mask=attn_mask,
        scale=d ** -0.5)
    ref = ref.transpose(1, 2).reshape(batch * seq, nh * d)[pos_idx]
    close(out, ref, 5e-3)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    for got, leaf in zip((qkv.grad, lq.grad, lv.grad), leaves):
        assert rel_err(got, leaf.grad) < 5e-3


def test_fused_head_loss():
    from solution.head import fused_head_loss

    torch.manual_seed(0)
    rows, hidden, vocab = (40, 32, 5003) if SMALL else (700, 1536, 151936)
    h = torch.randn(rows, hidden, device=DEV, dtype=torch.float16, requires_grad=True)
    w = (torch.randn(vocab, hidden, device=DEV) * 0.2).to(torch.float16)
    targets = torch.randint(0, vocab, (rows,), device=DEV)
    targets[torch.rand(rows, device=DEV) < 0.4] = -100
    targets[3] = int(torch.argmax(h[3].float() @ w.float().t()))  # one exact hit
    denom = torch.tensor(int((targets != -100).sum()) + 3, device=DEV)
    loss, entropy, token_loss, correct = fused_head_loss(h, w, targets, denom)

    hr = h.detach().clone().requires_grad_(True)
    logits = (hr @ w.t()).float()
    ref_loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100,
                                                 reduction="sum") / denom
    logp = torch.log_softmax(logits, -1)
    ref_entropy = -(logp.exp() * logp).sum(-1)
    lab = targets != -100
    ref_tok = torch.zeros(rows, device=DEV)
    ref_tok[lab] = -logp[lab].gather(1, targets[lab, None]).squeeze(1)
    ref_correct = (logits.argmax(-1) == targets) & lab
    close(loss, ref_loss, 1e-4)
    close(entropy, ref_entropy, 1e-3)
    close(token_loss, ref_tok, 1e-3)
    assert torch.equal(correct, ref_correct)
    scale = torch.tensor(1024.0, device=DEV)
    (loss * scale).backward()
    (ref_loss * scale).backward()
    close(h.grad, hr.grad, 2e-2)
