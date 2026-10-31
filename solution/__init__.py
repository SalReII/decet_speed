"""Soup Kernel Challenge solution: Triton kernels for QLoRA training of Qwen2 on a T4.

`apply()` installs four narrow patches (see trainer_patch.py); the training step of the task then
runs through model.fused_forward:

- head.py     fused head + loss + entropy + accuracy + per-row loss, no full logits (Triton)
- mlp.py      gate/up GEMM + SwiGLU in fp16 + recompute in backward (Triton)
- rmsnorm.py  residual add + RMSNorm, forward and backward (Triton)
- rope.py     LoRA add + rotary + scatter of q/k/v to SDPA's layout (Triton)
- nf4.py      NF4 double-quant unpacking for every frozen projection (Triton)
- model.py    the Qwen2 training forward on real (non-pad) tokens built from the above
- warmup.py   compiles the kernels on a background thread while the model loads
"""


def apply():
    from . import trainer_patch

    for patch in (trainer_patch.patch_forward, trainer_patch.patch_sft_compute_loss,
                  trainer_patch.patch_rewind, trainer_patch.patch_setup):
        try:
            patch()
        except Exception as exc:  # noqa: BLE001 - a missing package must not stop `soup train`
            print(f"[solution] {patch.__name__} not installed: {exc!r}", flush=True)
