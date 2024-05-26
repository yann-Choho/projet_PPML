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
