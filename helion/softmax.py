import torch
import triton
import triton.language as tl
from triton.runtime import driver
import time
from contextlib import contextmanager
import helion
import helion.language as hl


DEVICE = driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = driver.active.get_current_target()

print(f"NUM_SM: {NUM_SM}, NUM_REGS: {NUM_REGS}, SIZE_SMEM: {SIZE_SMEM}, WARP_SIZE: {WARP_SIZE}, target: {target}")

# PyTorch softmax
# In this case, we have 3 intermediate tensors: x_max, exp_x, and sum_exp
def softmax_pytorch(x):
    x_max = x.max(dim=-1, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    sum_exp = exp_x.sum(dim=-1, keepdim=True)[0]
    return exp_x / sum_exp


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x):
    # Things we need to tune:
    # - BLOCK_SIZE: the size of the block to process in each loop iteration
    # - num_stages: the number of software pipelining stages
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


@helion.kernel()
def softmax_helion(x: torch.Tensor) -> torch.Tensor:
    """
    Helion kernel implementing softmax by decomposing into max, exp, and normalization steps.
    This avoids using PyTorch's built-in softmax decomposition.
    Args:
        x (torch.Tensor): Input tensor of shape [n, m].
    Returns:
        torch.Tensor: Softmax output tensor of the same shape.
    """
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


@contextmanager
def memory_tracker(name=""):
    """Context manager to track GPU memory allocations.
    
    Note: allocated_delta shows net memory change (end - start).
    If intermediate tensors are garbage collected, they won't show up
    in allocated_delta, but peak_memory_delta will capture them.
    """
    if torch.cuda.is_available():
        import gc
        torch.cuda.synchronize()
        gc.collect()  # Force GC before measuring to get baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()
        start_time = time.perf_counter()
        
        yield
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        end_allocated = torch.cuda.memory_allocated()
        end_reserved = torch.cuda.memory_reserved()
        peak_allocated = torch.cuda.max_memory_allocated()
        
        allocated_delta = end_allocated - start_allocated
        peak_delta = peak_allocated - start_allocated
        reserved_delta = end_reserved - start_reserved
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"Memory Tracking: {name}")
        print(f"{'='*60}")
        print(f"Time: {elapsed_time*1000:.3f} ms")
        print(f"Allocated delta: {allocated_delta / 1024**2:.2f} MB (net change)")
        print(f"Peak memory delta: {peak_delta / 1024**2:.2f} MB (includes GC'd tensors)")
        if peak_delta > allocated_delta:
            print(f"  ⚠️  Peak > Allocated: {(peak_delta - allocated_delta) / 1024**2:.2f} MB were GC'd")
        print(f"Reserved delta: {reserved_delta / 1024**2:.2f} MB")
        print(f"Peak allocated: {peak_allocated / 1024**2:.2f} MB")
    else:
        yield


if __name__ == "__main__":
    # Create input tensor
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    
    print(f"Input tensor shape: {x.shape}, dtype: {x.dtype}")
    print(f"Input tensor size: {x.numel() * x.element_size() / 1024**2:.2f} MB")
    
    # Track the entire softmax operation
    with memory_tracker("softmax_pytorch"):
        result = softmax_pytorch(x)
    
    with memory_tracker("softmax_triton"):
        result_triton = softmax_triton(x)
    
    # with memory_tracker("softmax_helion"):
    result_helion = softmax_helion(x)
    
    # Compare with PyTorch's built-in softmax
    print(f"\n{'='*60}")
    print("Comparison with torch.nn.functional.softmax")
    print(f"{'='*60}")
    
    with memory_tracker("torch.nn.functional.softmax"):
        result_builtin = torch.nn.functional.softmax(x, dim=-1)
    
    print(f"\nMax difference: {(result - result_builtin).abs().max().item():.2e}")