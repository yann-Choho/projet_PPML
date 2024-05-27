#%%
import torch

import triton
import triton.language as tl

import itertools

# %%
# Triple mul. kernel

@torch.jit.script
def naive_triple_mul(A,B,C):
    return A*B*C

def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE":2**i},num_stages=j,num_warps=k) for i,j,k in itertools.product([7,8,9,10],[3,4,5],[2,4,8])
    ]

# Inspired by Triton tutorials
@triton.autotune(
    configs=get_autotune_config(),
    key = ["n_elements"]
)
@triton.jit
def triple_mul_kernel(A_ptr,
               B_ptr,
               C_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    A = tl.load(A_ptr + offsets, mask=mask)
    B = tl.load(B_ptr + offsets, mask=mask)
    C = tl.load(C_ptr + offsets, mask=mask)
    output = A*B*C
    tl.store(output_ptr + offsets, output, mask=mask)

def triple_mul(A,
               B,
               C):
    output = torch.empty_like(A)
    assert A.is_cuda and B.is_cuda and C.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    triple_mul_kernel[grid](A, B, C, output, n_elements)
    return output


# %%
# Correctness

torch.manual_seed(0)
size = 314
A = torch.rand((size,size), device='cuda')
B = torch.rand((size,size), device='cuda')
C = torch.rand((size,size), device='cuda')
output_torch = A*B*C
output_triton = triple_mul(A, B, C)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(6, 13, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    A = torch.rand((size,size), device='cuda')
    B = torch.rand((size,size), device='cuda')
    C = torch.rand((size,size), device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_triple_mul(A,B,C), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triple_mul(A,B,C), quantiles=quantiles)
    gbps = lambda ms: 12 * size**2 / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)

# %%
# GLU kernel

# @torch.jit.script
# def naive_glu(X, G, U, D):
#     Z = X @ G.T
#     return (Z * torch.sigmoid(Z) * (X @ U.T)) @ D.T


# @torch.jit
# def glu_kernel(X, G, L, U):  # other kwargs
#     pass

def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 2**i, "BLOCK_SIZE_N": 2**i, "BLOCK_SIZE_K": 64}, num_stages=j, num_warps=k)
        for i, j, k in itertools.product([4, 5, 6], [2, 3, 4], [2, 4, 8])
    ]

# % MLP Fused kernel
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.jit
def ff_llama(
    a_ptr, w1_ptr, w3_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(w1_ptrs)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True)
            b = b.to(tl.float32)
            b = b.to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True)
            c = c.to(tl.float32)
            c = c.to(tl.float16)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    acc1 = acc1
    acc2 = acc2
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert w1.dtype == w3.dtype
    assert w1.dtype in [torch.int8, torch.float16]
    assert w1.shape == w3.shape

    w1_t = w1.t()
    w3_t = w3.t()

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim

    N = w1_t.shape[1]
    assert K == w1_t.shape[0]
    assert w1_t.shape == w3_t.shape
    x_reshape = x.reshape(M, K)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)
    ff_llama[grid](
        x_reshape, w1_t, w3_t, out,
        M, N, K,
        *x_reshape.stride(),
        *w1_t.stride(),
        *w3_t.stride(),
        *out.stride(),
        USE_FP8=w1_t.dtype != torch.float16,
        EPS=1e-6,
        #BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, #already defined in autotune
        #num_stages=2, num_warps=4
    )
    out = out.view(batch, seq_len, -1)
    return out


# %%
# MLP Fused kernel WITH RMS NORM

def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 2**i, "BLOCK_SIZE_N": 2**i, "BLOCK_SIZE_K": 64}, num_stages=j, num_warps=k)
        for i, j, k in itertools.product([4, 5, 6], [2, 3, 4], [2, 4, 8])
    ]

# % MLP Fused kernel
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K']
)
@triton.jit
def ff_llama_with_rmsnorm(
    a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    A Triton kernel for performing feed-forward operations in a LLaMA model.
    
    This kernel computes the feed-forward transformation using the following operations:
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    
    Args:
        a_ptr: Pointer to the input tensor.
        w1_ptr: Pointer to the first weight tensor.
        w3_ptr: Pointer to the third weight tensor.
        out_ptr: Pointer to the output tensor.
        rms_w_ptr: Pointer to the RMS normalization weights.
        M: Number of rows in the input tensor.
        N: Number of columns in the weight tensors.
        K: Number of columns in the input tensor.
        stride_am: Stride of the input tensor in the first dimension.
        stride_ak: Stride of the input tensor in the second dimension.
        stride_w1k: Stride of the first weight tensor in the first dimension.
        stride_w1n: Stride of the first weight tensor in the second dimension.
        stride_w3k: Stride of the third weight tensor in the first dimension.
        stride_w3n: Stride of the third weight tensor in the second dimension.
        stride_outm: Stride of the output tensor in the first dimension.
        stride_outn: Stride of the output tensor in the second dimension.
        stride_rms_w: Stride of the RMS normalization weights.
        USE_FP8: Constant specifying whether to use FP8 precision.
        EPS: Constant epsilon value for numerical stability in RMS normalization.
        BLOCK_SIZE_M: Constant block size in the M dimension.
        BLOCK_SIZE_N: Constant block size in the N dimension.
        BLOCK_SIZE_K: Constant block size in the K dimension.
    """
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    a_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        a_sum += tl.math.pow(a.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        a = a * rms_w
        b = tl.load(w1_ptrs)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True)
            b = b.to(tl.float32)
            b = b.to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True)
            c = c.to(tl.float32)
            c = c.to(tl.float16)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w

    a_mean = tl.sum(a_sum, axis=1) / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def kernel_ff_with_rmsnorm(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    """
    A wrapper function to execute the Triton kernel for feed-forward operations.

    Args:
        x: Input tensor of shape (batch, seq_len, dim) with dtype torch.float16.
        w1: First weight tensor with dtype torch.float16 or torch.int8.
        w3: Third weight tensor with dtype torch.float16 or torch.int8.
        rms_w: RMS normalization weight tensor with dtype torch.float16.

    Returns:
        Output tensor after applying the feed-forward transformation.
    """
    assert x.dtype == torch.float16
    assert w1.dtype == w3.dtype == rms_w.dtype
    assert w1.dtype in [torch.int8, torch.float16]
    assert w1.shape == w3.shape

    w1_t = w1.t()
    w3_t = w3.t()

    batch, seq_len, dim = x.shape
    M, K = batch * seq_len, dim

    N = w1_t.shape[1]
    assert K == w1_t.shape[0]
    assert w1_t.shape == w3_t.shape
    x_reshape = x.reshape(M, K)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)
    ff_llama_with_rmsnorm[grid](
        x_reshape, w1_t, w3_t, out, rms_w,
        M, N, K,
        *x_reshape.stride(),
        *w1_t.stride(),
        *w3_t.stride(),
        *out.stride(),
        *rms_w.stride(),
        USE_FP8=w1_t.dtype != torch.float16,
        EPS=1e-6,
        #BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
        #num_stages=2, num_warps=4
    )
    out = out.view(batch, seq_len, -1)
    return out

