"""
Code modified from different sources:

* https://github.com/ELS-RD/kernl/tree/main/experimental/llama-v2
"""

from typing import List, Optional

import triton
import triton.language as tl
import torch

torch.manual_seed(123)


def find_last_one_index(lst: List[int]) -> Optional[int]:
    """
    Find the index of the last 1 in a list of integers.
    """
    index = len(lst) - 1
    while index >= 0:
        if lst[index] == 1:
            return index
        else:
            index -= 1
    return None


def f8_to_f16(x, dtypes=tl.float8e5) -> torch.Tensor:
    """
    Convert a torch.int8 tensor to torch.float16.
    """
    assert x.dtype == torch.int8, f"torch.int8 expected but got {x.dtype}"
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty_like(x, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = ret.untyped_storage().size() // ret.element_size()  # manage cases where tensor is not contiguous, like ::2
    kernel[grid](ret, triton.reinterpret(x, dtypes), numel, BLOCK_SIZE=1024)
    return ret


def f16_to_f8(x: torch.Tensor, dtypes=tl.float8e5) -> torch.Tensor:
    """
    Convert a torch.float16 tensor to torch.int8.
    """
    assert x.dtype in [torch.float16, torch.float32]
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty_like(x, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    numel = x.untyped_storage().size() // x.element_size()  # manage cases where tensor is not contiguous, like ::2
    kernel[grid](triton.reinterpret(ret, dtypes), x, numel, BLOCK_SIZE=1024)
    return ret

# Test
for _ in range(20):
    a = torch.randn((16, 128), dtype=torch.float16, device="cuda")
    b = f16_to_f8(a, dtypes=tl.float8e5)
    c = f8_to_f16(b, dtypes=tl.float8e5) + 1e-4

    assert (a/c).abs().mean().item()-1 < 1e-1, f"{(a/c).abs().mean()}"
