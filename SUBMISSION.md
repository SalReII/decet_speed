# Submission: <team name>

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  README.md): https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `head_stats_grad_kernel` | `solution/head.py` | full-vocab logits path: loss, entropy, accuracy, rewind row loss, gradient | 0.6% |
| `gated_silu_fwd_kernel` | `solution/mlp.py` | fp32 silu(gate) * up | 1.0% |
| `gated_silu_bwd_inplace_kernel` | `solution/mlp.py` | its backward, in place into the gate/up buffer | 1.7% |
| `rmsnorm_fwd_kernel` | `solution/rmsnorm.py` | residual add + fp32 Qwen2RMSNorm (+ fp16/fp32 outputs) | 0.9% |
| `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | its backward (GEMM + LoRA + residual gradients) | 1.0% |
| `rope_pack_fwd_kernel` | `solution/rope.py` | LoRA add + RoPE + transpose + casts of q/k/v | 0.3% |
| `rope_pack_bwd_kernel` | `solution/rope.py` | its backward | 0.2% |
| `nf4_unpack_kernel` | `solution/nf4.py` | bitsandbytes NF4 double-quant dequantization | 8.9% |
| **total** | | | **14.8%** |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- chunked head with the gradient taken in forward: Liger Kernel FusedLinearCrossEntropy
  (https://github.com/linkedin/Liger-Kernel), Cut Cross-Entropy (arXiv:2411.09009);
- online softmax: arXiv:1805.02867;
- row-wise norm forward/backward: Triton tutorial "Layer Normalization";
- patching the model classes: Unsloth (https://github.com/unslothai/unsloth);
- NF4 layout: QLoRA (arXiv:2305.14314) and the bitsandbytes 0.50.2 sources.

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
(`python tools/results_table.py` prints these rows.)

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 88.03 | 56.29 | speedup x1.564 |
| Peak GPU memory, MiB | 14345 | 3299 | base/ours x4.348 |
| Eval loss (public rows) | 2.04963 | 2.04970 | +0.003 % vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (fp32 activations from 4-bit layers, the 152k-vocab logits
path, padding), how the kernels work (head with online softmax and in-place gradient, fused
SwiGLU with recompute, add+RMSNorm, RoPE+pack, NF4 unpack), results against the base (speed,
memory, quality), limits and next steps.

## Limitations
Attention stays PyTorch SDPA on the padded layout (now the largest activation term); GEMMs are
cuBLAS on unpacked fp16 weights; the fast path is specific to Qwen2 + bitsandbytes NF4 + vanilla
LoRA + fp16 autocast (anything else runs the original code). Next: a Triton GQA varlen
flash-attention and a fused NF4 GEMM for sm75.
