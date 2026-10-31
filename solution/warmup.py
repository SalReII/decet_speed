"""Compile our kernels on a background thread while Soup loads the model.

Every measured run starts with an empty Triton cache, so the first launch of each kernel pays
for its compilation. The kernels' compile-time constants come from the model config, which is
already on disk when `soup train` starts (the harness downloads the model first), so the
compilation can run while the main thread reads and quantizes the weights. The launches here
use tiny tensors with the same dtypes, constants and alignment as the training launches, so
training finds every kernel compiled. Nothing here is required: if it fails or is late, the
kernels compile on first use as usual.
"""

import threading
from types import SimpleNamespace

import torch

_thread = None


def _fake_nf4(n_out, n_in, device):
    n_bytes = n_out * n_in // 2
    state2 = SimpleNamespace(code=torch.zeros(256, device=device),
                             absmax=torch.zeros(1, device=device), blocksize=256)
    state = SimpleNamespace(quant_type="nf4", nested=True, state2=state2, blocksize=64,
                            shape=(n_out, n_in),
                            absmax=torch.zeros(max(1, n_out * n_in // 64), dtype=torch.uint8,
                                               device=device),
                            offset=torch.zeros((), device=device),
                            code=torch.zeros(16, device=device))
    return SimpleNamespace(data=torch.zeros((n_bytes, 1), dtype=torch.uint8, device=device),
                           quant_state=state)


def compile_kernels(hidden, intermediate, n_heads, n_kv, head_dim, vocab, device="cuda"):
    from . import head, mlp, nf4, rmsnorm, rope

    dev = torch.device(device)
    with torch.no_grad():
        weight = _fake_nf4(4, 512, dev)
        nf4.unpack_into(weight, torch.empty((4, 512), dtype=torch.float16, device=dev))

        x = torch.zeros((2, hidden), device=dev)
        delta = torch.zeros((2, hidden), dtype=torch.float16, device=dev)
        w = torch.ones(hidden, device=dev)
        pad_idx = torch.arange(2, device=dev)
        s, y, ypad, rstd = rmsnorm._launch_fwd(x, delta, w, 1e-6, pad_idx, 3)
        rmsnorm._launch_bwd(s, rstd, w, pad_idx, x, y, ypad, True)
        rmsnorm._launch_fwd(x, None, w, 1e-6, pad_idx, 3)       # same compiled variants as
        rmsnorm._launch_fwd(x, delta, w, 1e-6, None, 0)         # above (dummies share dtypes);
        rmsnorm._launch_bwd(s, rstd, w, None, None, y, None, True)  # kept as a cheap check

        gu = torch.zeros((1, 2 * intermediate), dtype=torch.float16, device=dev)
        act = mlp.gated_silu(gu)
        mlp.gated_silu_backward_(act, gu)

        width = (n_heads + 2 * n_kv) * head_dim
        qkv = torch.zeros((1, width), dtype=torch.float16, device=dev)
        lq = torch.zeros((1, n_heads * head_dim), dtype=torch.float16, device=dev)
        lv = torch.zeros((1, n_kv * head_dim), dtype=torch.float16, device=dev)
        cos = torch.zeros((2, head_dim), device=dev)
        q, k, v = (torch.zeros((n_heads, 1, head_dim), dtype=torch.float16, device=dev)
                   for _ in range(3))
        pos = torch.zeros(1, dtype=torch.long, device=dev)
        consts = rope._consts(n_heads, n_kv, head_dim)
        rope.rope_pack_fwd_kernel[(1,)](qkv, lq, lv, cos, cos, pos, q, k, v, 1, 2, 1, 1, **consts)
        rope.rope_pack_bwd_kernel[(1,)](q, k, v, cos, cos, pos, qkv, 1, 2, 1, 1, 1, **consts)

        logits = torch.zeros((1, vocab), dtype=torch.float16, device=dev)
        target = torch.zeros(1, dtype=torch.long, device=dev)
        out = torch.zeros(1, device=dev)
        head.head_stats_grad_kernel[(1,)](logits, target, out, out.clone(),
                                          torch.zeros(1, dtype=torch.int32, device=dev), 1.0,
                                          V=vocab, BLOCK=head.VOCAB_BLOCK, num_warps=8)


def start(base_model):
    """Start compiling for `base_model` if its config is already on disk (no network)."""
    global _thread
    if _thread is not None or not torch.cuda.is_available():
        return
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(base_model, local_files_only=True)
        if getattr(cfg, "model_type", None) != "qwen2":
            return
        dims = dict(hidden=cfg.hidden_size, intermediate=cfg.intermediate_size,
                    n_heads=cfg.num_attention_heads, n_kv=cfg.num_key_value_heads,
                    head_dim=getattr(cfg, "head_dim", None)
                    or cfg.hidden_size // cfg.num_attention_heads,
                    vocab=cfg.vocab_size)
    except Exception:  # noqa: BLE001 - not cached, not a Qwen2: compile on first use
        return

    def run():
        try:
            compile_kernels(**dims)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            print(f"[solution] background kernel compilation skipped: {exc!r}", flush=True)

    _thread = threading.Thread(target=run, name="soup-kernel-warmup", daemon=True)
    _thread.start()


def wait(timeout=180):
    """Let the background compilation finish before the first step (it normally has)."""
    if _thread is not None:
        _thread.join(timeout)
