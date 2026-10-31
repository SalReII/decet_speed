"""Small helpers shared by the kernels."""

import os

import torch


def interpreting():
    """Triton's CPU interpreter (TRITON_INTERPRET=1), used by the CPU tests in this folder."""
    return os.environ.get("TRITON_INTERPRET") == "1"


def triton_ok(tensor):
    """Our kernels run on CUDA tensors (and on CPU tensors under the interpreter)."""
    return torch.is_tensor(tensor) and (tensor.is_cuda or interpreting())


def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()
