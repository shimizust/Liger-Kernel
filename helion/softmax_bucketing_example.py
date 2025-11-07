"""
Example: Using Different Bucketing Strategies with Helion Kernels

This file demonstrates how to create Helion kernels with different bucketing strategies
for handling multiple input shapes efficiently.

Key Concepts:
1. static_shapes=True (default): Creates a new kernel specialization for each exact shape
2. static_shapes=False: Reuses kernels across shapes (buckets by dtype/device only)
3. Custom key functions: Define your own bucketing logic (e.g., power-of-two)
4. Multiple candidate configs: Helion autoselects the best config for each bucket
"""

import torch
import helion
import helion.language as hl
from typing import Tuple


# ==============================================================================
# Strategy 1: Static Shapes (Default - Exact Shape Matching)
# ==============================================================================
# This is the default behavior. Each unique shape gets its own compiled kernel.
# Best for: When you know all shapes ahead of time and want optimal per-shape performance

@helion.kernel()  # Equivalent to: @helion.kernel(static_shapes=True)
def softmax_static_shapes(x: torch.Tensor) -> torch.Tensor:
    """
    Default Helion behavior: exact shape matching.
    
    - Input (1024, 256) compiles a kernel for that exact shape
    - Input (2048, 256) triggers a new compilation
    - Input (1024, 256) reuses the first kernel (cache hit)
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


# ==============================================================================
# Strategy 2: Dynamic Shapes (Bucket by Dtype/Device Only)
# ==============================================================================
# Reuses the same kernel across ALL shapes as long as dtype and device match.
# Best for: Highly variable input shapes where compilation time matters

@helion.kernel(static_shapes=False)
def softmax_dynamic_shapes(x: torch.Tensor) -> torch.Tensor:
    """
    Dynamic shape bucketing: one kernel handles all shapes.
    
    - Input (1024, 256) compiles a kernel
    - Input (2048, 256) reuses the same kernel (no recompilation!)
    - Input (512, 8192) still reuses the same kernel
    
    Bucketing dimensions {0, 1, ≥2}:
    - Shapes are only re-specialized if a dimension crosses these buckets
    - E.g., going from shape (1, 256) to (2, 256) might trigger recompilation
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


# ==============================================================================
# Strategy 3: Custom Key Functions (Power-of-Two Bucketing)
# ==============================================================================
# Group shapes by custom criteria, e.g., power-of-two buckets
# Best for: When you want more control over shape grouping

@helion.kernel(
    key=lambda x: helion.next_power_of_2(x.numel()),
    static_shapes=False
)
def softmax_power_of_two_bucketing(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bucketing by power-of-two sizes.
    
    - Shapes with similar total elements share kernels
    - (1024, 256) and (512, 512) both have 262144 elements
    - Next power of 2 is 262144, so they share a kernel
    - (2048, 256) has 524288 elements → different bucket
    
    This balances between exact shape matching and full dynamic shapes.
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


# ==============================================================================
# Strategy 4: Multiple Candidate Configs with Autotuning
# ==============================================================================
# Provide multiple pre-tuned configs; Helion picks the best for each bucket
# Best for: When you've pre-tuned for different workload sizes

def softmax_with_candidate_configs():
    """
    Example of using multiple candidate configs.
    
    This requires pre-generated config files from autotuning runs.
    Helion will benchmark each config on first use and select the fastest.
    """
    
    # First, you would need to generate configs (usually via autotuning):
    # See softmax.py's run_autotune_softmax() function
    
    # Then load and use them:
    candidate_configs = [
        helion.Config.load("configs/softmax_small.json"),   # Optimized for small shapes
        helion.Config.load("configs/softmax_medium.json"),  # Optimized for medium shapes
        helion.Config.load("configs/softmax_large.json"),   # Optimized for large shapes
    ]
    
    @helion.kernel(configs=candidate_configs, static_shapes=False)
    def softmax_multi_config(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            values = x[tile_n, :]
            amax = torch.amax(values, dim=1, keepdim=True)
            exp = torch.exp(values - amax)
            sum_exp = torch.sum(exp, dim=1, keepdim=True)
            out[tile_n, :] = exp / sum_exp
        return out


# ==============================================================================
# Demonstration and Comparison
# ==============================================================================

def demonstrate_bucketing():
    """
    Demonstrate different bucketing strategies with example shapes.
    """
    print("=" * 80)
    print("HELION BUCKETING STRATEGIES DEMONSTRATION")
    print("=" * 80)
    
    # Test shapes with varying sizes
    test_shapes = [
        (1024, 256),
        (2048, 256),
        (1024, 512),
        (4096, 4096),
    ]
    
    for strategy_name, kernel_fn in [
        ("Static Shapes (Default)", softmax_static_shapes),
        ("Dynamic Shapes", softmax_dynamic_shapes),
        ("Power-of-Two Bucketing", softmax_power_of_two_bucketing),
    ]:
        print(f"\n{'=' * 80}")
        print(f"Testing: {strategy_name}")
        print("=" * 80)
        
        for i, (rows, cols) in enumerate(test_shapes, 1):
            x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
            
            # First call may compile
            y = kernel_fn(x)
            
            # Verify correctness
            expected = torch.nn.functional.softmax(x, dim=-1)
            max_diff = (y - expected).abs().max().item()
            
            print(f"  Shape {i}: ({rows:5d}, {cols:5d}) | "
                  f"numel: {x.numel():8d} | "
                  f"next_pow2: {helion.next_power_of_2(x.numel()):8d} | "
                  f"max_diff: {max_diff:.2e}")
    
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print("1. Static Shapes: May recompile for each new shape")
    print("2. Dynamic Shapes: Compiles once, reuses for all shapes")
    print("3. Power-of-Two: Groups similar-sized tensors together")
    print("\nChoose based on your workload:")
    print("  - Known shapes → Static for best per-shape performance")
    print("  - Variable shapes → Dynamic or Power-of-Two for fewer compilations")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_bucketing()
    
    print("\n\nTo use these strategies in benchmarks:")
    print("  1. Copy the desired kernel definition to softmax.py")
    print("  2. Replace the @helion.kernel() decorator on softmax_helion()")
    print("  3. Run: python benchmark_softmax.py --bucketed")
    print("\nThis will test how well the bucketing strategy generalizes to unseen shapes!")

