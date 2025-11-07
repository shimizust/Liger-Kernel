import torch
import triton
from triton.testing import do_bench
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from softmax import softmax_pytorch, softmax_triton, softmax_liger_triton, softmax_helion

# Create benchmarks directory
BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
BENCHMARK_DIR.mkdir(exist_ok=True)


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

# Benchmark parameters
BENCHMARK_CONFIG = {
    'warmup': 25,
    'rep': 300,
}

# Implementation configurations
# Each implementation needs: key, fn, label, label_short, color, description
IMPLEMENTATIONS = [
    {
        'key': 'pytorch_native',
        'fn': lambda x: torch.nn.functional.softmax(x, dim=-1),
        'label': 'Native PyTorch',
        'label_short': 'Native\nPyTorch',
        'color': '#1f77b4',
        'description': 'torch.nn.functional.softmax',
        'compile': False,
    },
    {
        'key': 'pytorch_custom',
        'fn': softmax_pytorch,
        'label': 'Custom PyTorch',
        'label_short': 'Custom\nPyTorch',
        'color': '#17becf',
        'description': 'Custom PyTorch implementation',
        'compile': False,
    },
    {
        'key': 'pytorch_compiled',
        'fn': softmax_pytorch,
        'label': 'Custom PyTorch Compiled',
        'label_short': 'Custom\nCompiled',
        'color': '#9467bd',
        'description': 'torch.compile(Custom PyTorch)',
        'compile': True,
    },
    {
        'key': 'triton_tutorial',
        'fn': softmax_triton,
        'label': 'Triton Tutorial',
        'label_short': 'Triton\nTutorial',
        'color': '#ff7f0e',
        'description': 'Triton tutorial implementation',
        'compile': False,
    },
    {
        'key': 'liger_triton',
        'fn': softmax_liger_triton,
        'label': 'Liger Triton',
        'label_short': 'Liger\nTriton',
        'color': '#2ca02c',
        'description': 'Liger Triton implementation',
        'compile': False,
    },
    {
        'key': 'helion',
        'fn': softmax_helion,
        'label': 'Helion',
        'label_short': 'Helion',
        'color': '#d62728',
        'description': 'Helion implementation',
        'compile': False,
    },
]

# Baseline implementation for speedup calculations
BASELINE_KEY = 'pytorch_native'


def benchmark_softmax(n_rows, n_cols, dtype=torch.float32):
    """
    Benchmark different softmax implementations using triton.testing.do_bench.
    
    Args:
        n_rows: Number of rows in input tensor
        n_cols: Number of columns in input tensor
        dtype: Data type of input tensor
    
    Returns:
        Dictionary with benchmark results
    """
    # Create input tensor
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
    
    print(f"\nBenchmarking shape: ({n_rows}, {n_cols}), dtype: {dtype}")
    print(f"Input size: {x.numel() * x.element_size() / 1024**2:.2f} MB")
    print("-" * 80)
    
    results = {}
    warmup = BENCHMARK_CONFIG['warmup']
    rep = BENCHMARK_CONFIG['rep']
    
    # Benchmark each implementation
    for impl in IMPLEMENTATIONS:
        key = impl['key']
        fn = impl['fn']
        label = impl['label']
        description = impl['description']
        compile = impl['compile']
        
        print(f"Benchmarking {label} ({description})...")
        
        if compile:
            # Compile the function and warm it up
            compiled_fn = torch.compile(fn)
            _ = compiled_fn(x)
            bench_time = do_bench(lambda: compiled_fn(x), warmup=warmup, rep=rep)
        else:
            bench_time = do_bench(lambda: fn(x), warmup=warmup, rep=rep)
        
        results[key] = bench_time
        print(f"{label}: {bench_time:.4f} ms")
    
    # Calculate speedups relative to baseline
    baseline_time = results[BASELINE_KEY]
    baseline_label = next(impl['label'] for impl in IMPLEMENTATIONS if impl['key'] == BASELINE_KEY)
    
    print("\n" + "=" * 80)
    print(f"Speedup relative to {baseline_label}:")
    print("=" * 80)
    for impl in IMPLEMENTATIONS:
        key = impl['key']
        if key != BASELINE_KEY:
            speedup = baseline_time / results[key]
            print(f"{impl['label']:30s}: {speedup:.2f}x")
    
    # Find fastest implementation
    fastest_key = min(results, key=results.get)
    fastest_label = next(impl['label'] for impl in IMPLEMENTATIONS if impl['key'] == fastest_key)
    print(f"\nFastest: {fastest_label} ({results[fastest_key]:.4f} ms)")
    
    return results


def plot_timing_comparison(df, output_file='timing_comparison.png'):
    """
    Create a bar chart comparing execution times across different implementations.
    """
    output_path = BENCHMARK_DIR / output_file
    
    n_impls = len(IMPLEMENTATIONS)
    plt.figure(figsize=(18, 8))
    
    x = np.arange(len(df))
    width = 0.8 / n_impls
    
    for i, impl in enumerate(IMPLEMENTATIONS):
        offset = (i - n_impls/2 + 0.5) * width
        plt.bar(x + offset, df[impl['key']], width, 
                label=impl['label'], alpha=0.8, color=impl['color'])
    
    plt.xlabel('Tensor Shape', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    plt.title('Softmax Implementation Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, df['shape'], rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved timing comparison plot to: {output_path}")
    plt.close()


def plot_speedup_comparison(df, output_file='speedup_comparison.png'):
    """
    Create a bar chart comparing speedups relative to baseline.
    """
    output_path = BENCHMARK_DIR / output_file
    
    # Get non-baseline implementations
    non_baseline_impls = [impl for impl in IMPLEMENTATIONS if impl['key'] != BASELINE_KEY]
    baseline_impl = next(impl for impl in IMPLEMENTATIONS if impl['key'] == BASELINE_KEY)
    
    n_impls = len(non_baseline_impls)
    plt.figure(figsize=(18, 8))
    
    x = np.arange(len(df))
    width = 0.8 / n_impls
    
    for i, impl in enumerate(non_baseline_impls):
        speedup_key = f"{impl['key']}_speedup"
        offset = (i - n_impls/2 + 0.5) * width
        plt.bar(x + offset, df[speedup_key], width, 
                label=impl['label'], alpha=0.8, color=impl['color'])
    
    # Add horizontal line at y=1 for baseline
    plt.axhline(y=1, color=baseline_impl['color'], linestyle='--', linewidth=2, 
                label=f'{baseline_impl["label"]} (Baseline)', alpha=0.7)
    
    plt.xlabel('Tensor Shape', fontsize=12, fontweight='bold')
    plt.ylabel(f'Speedup vs {baseline_impl["label"]}', fontsize=12, fontweight='bold')
    plt.title('Softmax Speedup Comparison (Higher is Better)', fontsize=14, fontweight='bold')
    plt.xticks(x, df['shape'], rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved speedup comparison plot to: {output_path}")
    plt.close()


def plot_timing_log_scale(df, output_file='timing_comparison_log.png'):
    """
    Create a bar chart with log scale for better visibility of differences.
    """
    output_path = BENCHMARK_DIR / output_file
    
    n_impls = len(IMPLEMENTATIONS)
    plt.figure(figsize=(18, 8))
    
    x = np.arange(len(df))
    width = 0.8 / n_impls
    
    for i, impl in enumerate(IMPLEMENTATIONS):
        offset = (i - n_impls/2 + 0.5) * width
        plt.bar(x + offset, df[impl['key']], width, 
                label=impl['label'], alpha=0.8, color=impl['color'])
    
    plt.xlabel('Tensor Shape', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (ms, log scale)', fontsize=12, fontweight='bold')
    plt.title('Softmax Performance Comparison (Log Scale)', fontsize=14, fontweight='bold')
    plt.xticks(x, df['shape'], rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved log-scale timing plot to: {output_path}")
    plt.close()


def plot_summary_statistics(df, output_file='summary_statistics.png'):
    """
    Create a summary bar chart showing average speedups.
    """
    output_path = BENCHMARK_DIR / output_file
    baseline_impl = next(impl for impl in IMPLEMENTATIONS if impl['key'] == BASELINE_KEY)
    non_baseline_impls = [impl for impl in IMPLEMENTATIONS if impl['key'] != BASELINE_KEY]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Average speedup plot (non-baseline implementations only)
    impl_names_short = [impl['label_short'] for impl in non_baseline_impls]
    avg_speedups = [df[f"{impl['key']}_speedup"].mean() for impl in non_baseline_impls]
    colors = [impl['color'] for impl in non_baseline_impls]
    
    bars = ax1.bar(impl_names_short, avg_speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, 
                label=f'{baseline_impl["label"]} Baseline')
    ax1.set_ylabel(f'Average Speedup vs {baseline_impl["label"]}', fontsize=12, fontweight='bold')
    ax1.set_title('Average Speedup Across All Configurations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Add value labels on bars
    for bar, speedup in zip(bars, avg_speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Average execution time plot (all implementations)
    impl_names_short_all = [impl['label_short'] for impl in IMPLEMENTATIONS]
    avg_times = [df[impl['key']].mean() for impl in IMPLEMENTATIONS]
    colors_time = [impl['color'] for impl in IMPLEMENTATIONS]
    
    bars2 = ax2.bar(impl_names_short_all, avg_times, color=colors_time, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Execution Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Execution Time Across All Configurations', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars2, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}ms',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary statistics plot to: {output_path}")
    plt.close()


def run_comprehensive_benchmark():
    """
    Run benchmarks across various tensor shapes.
    """
    # Test configurations: (n_rows, n_cols)
    configs = [
        (1024, 128),     # Small
        (4096, 256),     # Medium
        (16384, 512),    # Medium-Large
        (64000, 256),    # Large (many rows)
        (4096, 4096),    # Square
        (1024, 8192),    # Wide (few rows, many cols)
    ]
    
    all_results = []
    
    for n_rows, n_cols in configs:
        results = benchmark_softmax(n_rows, n_cols)
        results['shape'] = f"{n_rows}x{n_cols}"
        results['n_rows'] = n_rows
        results['n_cols'] = n_cols
        all_results.append(results)
    
    # Create DataFrame for easy viewing
    df = pd.DataFrame(all_results)
    
    # Reorder columns dynamically based on IMPLEMENTATIONS
    impl_keys = [impl['key'] for impl in IMPLEMENTATIONS]
    cols = ['shape', 'n_rows', 'n_cols'] + impl_keys
    df = df[cols]
    
    # Calculate speedups relative to baseline
    baseline_impl = next(impl for impl in IMPLEMENTATIONS if impl['key'] == BASELINE_KEY)
    for impl in IMPLEMENTATIONS:
        if impl['key'] != BASELINE_KEY:
            df[f"{impl['key']}_speedup"] = df[BASELINE_KEY] / df[impl['key']]
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 120)
    print("\nTiming Results (ms):")
    print(df[['shape'] + impl_keys].to_string(index=False))
    
    print("\n" + "-" * 120)
    print(f"\nSpeedup vs {baseline_impl['label']}:")
    speedup_cols = [f"{impl['key']}_speedup" for impl in IMPLEMENTATIONS if impl['key'] != BASELINE_KEY]
    print(df[['shape'] + speedup_cols].to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    for impl in IMPLEMENTATIONS:
        if impl['key'] != BASELINE_KEY:
            speedup_col = f"{impl['key']}_speedup"
            print(f"Average {impl['label']} speedup: {df[speedup_col].mean():.2f}x")
    
    # Save results to CSV
    csv_path = BENCHMARK_DIR / 'softmax_benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots
    print("\n" + "=" * 120)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 120)
    plot_timing_comparison(df)
    plot_speedup_comparison(df)
    plot_timing_log_scale(df)
    plot_summary_statistics(df)
    print("\n✓ All plots generated successfully!")
    
    return df


def run_quick_benchmark():
    """
    Run a quick benchmark on a smaller set of shapes (useful for testing).
    """
    
    configs = [
        (4096, 256),     # Medium
        (4096, 4096),    # Square
        (1024, 8192),    # Wide
    ]
    
    all_results = []
    
    for n_rows, n_cols in configs:
        results = benchmark_softmax(n_rows, n_cols)
        results['shape'] = f"{n_rows}x{n_cols}"
        results['n_rows'] = n_rows
        results['n_cols'] = n_cols
        all_results.append(results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns dynamically based on IMPLEMENTATIONS
    impl_keys = [impl['key'] for impl in IMPLEMENTATIONS]
    cols = ['shape', 'n_rows', 'n_cols'] + impl_keys
    df = df[cols]
    
    # Calculate speedups relative to baseline
    baseline_impl = next(impl for impl in IMPLEMENTATIONS if impl['key'] == BASELINE_KEY)
    for impl in IMPLEMENTATIONS:
        if impl['key'] != BASELINE_KEY:
            df[f"{impl['key']}_speedup"] = df[BASELINE_KEY] / df[impl['key']]
    
    print("\n" + "=" * 120)
    print("QUICK BENCHMARK RESULTS")
    print("=" * 120)
    print("\nTiming Results (ms):")
    print(df[['shape'] + impl_keys].to_string(index=False))
    
    print("\n" + "-" * 120)
    print(f"\nSpeedup vs {baseline_impl['label']}:")
    speedup_cols = [f"{impl['key']}_speedup" for impl in IMPLEMENTATIONS if impl['key'] != BASELINE_KEY]
    print(df[['shape'] + speedup_cols].to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    for impl in IMPLEMENTATIONS:
        if impl['key'] != BASELINE_KEY:
            speedup_col = f"{impl['key']}_speedup"
            print(f"Average {impl['label']} speedup: {df[speedup_col].mean():.2f}x")
    
    # Save results
    csv_path = BENCHMARK_DIR / 'softmax_benchmark_quick.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots with "quick" prefix
    print("\n" + "=" * 120)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 120)
    plot_timing_comparison(df, 'quick_timing_comparison.png')
    plot_speedup_comparison(df, 'quick_speedup_comparison.png')
    plot_timing_log_scale(df, 'quick_timing_comparison_log.png')
    plot_summary_statistics(df, 'quick_summary_statistics.png')
    print("\n✓ All plots generated successfully!")
    
    return df


def run_bucketed_benchmark(training_shapes=None, test_shapes=None, bucketing_strategy='dynamic'):
    """
    Test Helion's bucketed config performance on unseen shapes.
    
    This benchmark:
    1. Warms up Helion on a small set of training shapes (compiling kernels)
    2. Benchmarks all implementations on a different set of test shapes
    3. Analyzes how well Helion's bucketed configs generalize to unseen shapes
    
    Args:
        training_shapes: List of (n_rows, n_cols) tuples for initial compilation
        test_shapes: List of (n_rows, n_cols) tuples for testing on unseen shapes
        bucketing_strategy: Strategy for shape bucketing:
            - 'default': static_shapes=True (exact shape matching, default Helion behavior)
            - 'dynamic': static_shapes=False (bucket by dtype/device, reuse across shapes)
            - 'power_of_two': Custom bucketing by power-of-two sizes
    
    Note:
        By default, Helion uses static_shapes=True which creates a new specialization
        for each exact shape. To test true bucketing behavior, you would need to modify
        the @helion.kernel() decorator in softmax.py to use:
        
        - static_shapes=False for dynamic shape bucketing
        - Or custom key functions like:
          @helion.kernel(key=lambda x: helion.next_power_of_2(x.numel()), static_shapes=False)
    """
    
    if training_shapes is None:
        # Small set of diverse shapes to compile on
        training_shapes = [
            (4096, 256),      # Medium - many rows, few cols
            (1024, 8192),     # Wide - few rows, many cols
            (4096, 4096),     # Square - balanced
        ]
    
    if test_shapes is None:
        # Completely different shapes to test generalization
        test_shapes = [
            (2048, 512),      # Different medium size
            (8192, 1024),     # Different wide size
            (512, 16384),     # Very wide
            (32768, 128),     # Very many rows
            (2048, 2048),     # Different square
            (16384, 4096),    # Large asymmetric
        ]
    
    print("=" * 120)
    print("BUCKETED CONFIG BENCHMARK - Testing Generalization to Unseen Shapes")
    print("=" * 120)
    print(f"\nBucketing Strategy: {bucketing_strategy}")
    if bucketing_strategy == 'default':
        print("  → Using static_shapes=True (exact shape matching)")
        print("  → Each new shape will trigger recompilation/autotuning")
    elif bucketing_strategy == 'dynamic':
        print("  → Using static_shapes=False (dynamic shape bucketing)")
        print("  → Shapes reuse kernels as long as dtype/device match")
    else:
        print(f"  → Custom bucketing strategy: {bucketing_strategy}")
    
    print("\nPhase 1: Warming up Helion on Training Shapes")
    print("-" * 120)
    print(f"Training shapes: {training_shapes}")
    print()
    
    # Phase 1: Warm up Helion on training shapes
    for i, (n_rows, n_cols) in enumerate(training_shapes, 1):
        print(f"Training {i}/{len(training_shapes)}: Compiling on shape ({n_rows}, {n_cols})")
        x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
        # Run Helion to compile and cache the kernel
        _ = softmax_helion(x)
        print(f"  ✓ Compilation complete for ({n_rows}, {n_cols})")
    
    print("\n" + "=" * 120)
    print("Phase 2: Benchmarking on Unseen Test Shapes")
    print("-" * 120)
    print(f"Test shapes: {test_shapes}")
    print()
    
    # Phase 2: Benchmark on test shapes
    all_results = []
    
    for n_rows, n_cols in test_shapes:
        results = benchmark_softmax(n_rows, n_cols)
        results['shape'] = f"{n_rows}x{n_cols}"
        results['n_rows'] = n_rows
        results['n_cols'] = n_cols
        results['shape_type'] = 'unseen'
        all_results.append(results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Reorder columns dynamically based on IMPLEMENTATIONS
    impl_keys = [impl['key'] for impl in IMPLEMENTATIONS]
    cols = ['shape', 'n_rows', 'n_cols', 'shape_type'] + impl_keys
    df = df[cols]
    
    # Calculate speedups relative to baseline
    baseline_impl = next(impl for impl in IMPLEMENTATIONS if impl['key'] == BASELINE_KEY)
    for impl in IMPLEMENTATIONS:
        if impl['key'] != BASELINE_KEY:
            df[f"{impl['key']}_speedup"] = df[BASELINE_KEY] / df[impl['key']]
    
    print("\n" + "=" * 120)
    print("BUCKETED BENCHMARK RESULTS - Unseen Shapes")
    print("=" * 120)
    print("\nTiming Results (ms):")
    print(df[['shape'] + impl_keys].to_string(index=False))
    
    print("\n" + "-" * 120)
    print(f"\nSpeedup vs {baseline_impl['label']} on Unseen Shapes:")
    speedup_cols = [f"{impl['key']}_speedup" for impl in IMPLEMENTATIONS if impl['key'] != BASELINE_KEY]
    print(df[['shape'] + speedup_cols].to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS - Unseen Shape Performance")
    print("=" * 120)
    for impl in IMPLEMENTATIONS:
        if impl['key'] != BASELINE_KEY:
            speedup_col = f"{impl['key']}_speedup"
            avg_speedup = df[speedup_col].mean()
            min_speedup = df[speedup_col].min()
            max_speedup = df[speedup_col].max()
            print(f"{impl['label']:30s}: avg={avg_speedup:.2f}x, min={min_speedup:.2f}x, max={max_speedup:.2f}x")
    
    # Special analysis for Helion
    if 'helion' in df.columns:
        print("\n" + "-" * 120)
        print("HELION GENERALIZATION ANALYSIS")
        print("-" * 120)
        helion_speedup = df['helion_speedup']
        print(f"Helion performance on unseen shapes:")
        print(f"  Average speedup: {helion_speedup.mean():.2f}x")
        print(f"  Median speedup:  {helion_speedup.median():.2f}x")
        print(f"  Std deviation:   {helion_speedup.std():.2f}x")
        print(f"  Min speedup:     {helion_speedup.min():.2f}x (shape: {df.loc[helion_speedup.idxmin(), 'shape']})")
        print(f"  Max speedup:     {helion_speedup.max():.2f}x (shape: {df.loc[helion_speedup.idxmax(), 'shape']})")
        
        # Compare to other custom kernels
        other_kernels = ['triton_tutorial', 'liger_triton']
        for kernel in other_kernels:
            if f'{kernel}_speedup' in df.columns:
                other_speedup = df[f'{kernel}_speedup'].mean()
                helion_avg = helion_speedup.mean()
                comparison = helion_avg / other_speedup
                if comparison > 1:
                    print(f"\n  Helion is {comparison:.2f}x faster than {kernel} on average")
                else:
                    print(f"\n  Helion is {1/comparison:.2f}x slower than {kernel} on average")
    
    # Save results
    csv_path = BENCHMARK_DIR / 'softmax_benchmark_bucketed.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to: {csv_path}")
    
    # Save training shapes metadata
    metadata_path = BENCHMARK_DIR / 'softmax_benchmark_bucketed_metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write("BUCKETED CONFIG BENCHMARK METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Bucketing Strategy: {bucketing_strategy}\n\n")
        f.write("Training Shapes (used for compilation):\n")
        for shape in training_shapes:
            f.write(f"  {shape[0]} x {shape[1]}\n")
        f.write("\nTest Shapes (unseen during compilation):\n")
        for shape in test_shapes:
            f.write(f"  {shape[0]} x {shape[1]}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("\nNOTE:\n")
        f.write("By default, Helion uses static_shapes=True (exact shape matching).\n")
        f.write("To test true bucketing behavior, modify the @helion.kernel() decorator\n")
        f.write("in softmax.py. Examples:\n\n")
        f.write("1. Dynamic shape bucketing:\n")
        f.write("   @helion.kernel(static_shapes=False)\n\n")
        f.write("2. Power-of-two bucketing:\n")
        f.write("   @helion.kernel(\n")
        f.write("       key=lambda x: helion.next_power_of_2(x.numel()),\n")
        f.write("       static_shapes=False\n")
        f.write("   )\n\n")
        f.write("3. Multiple candidate configs:\n")
        f.write("   candidate_configs = [\n")
        f.write("       helion.Config.load('config_small.json'),\n")
        f.write("       helion.Config.load('config_large.json'),\n")
        f.write("   ]\n")
        f.write("   @helion.kernel(configs=candidate_configs, static_shapes=False)\n")
    print(f"Metadata saved to: {metadata_path}")
    
    # Generate plots with "bucketed" prefix
    print("\n" + "=" * 120)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 120)
    plot_timing_comparison(df, 'bucketed_timing_comparison.png')
    plot_speedup_comparison(df, 'bucketed_speedup_comparison.png')
    plot_timing_log_scale(df, 'bucketed_timing_comparison_log.png')
    plot_summary_statistics(df, 'bucketed_summary_statistics.png')
    print("\n✓ All plots generated successfully!")
    
    return df


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Run quick benchmark
        print("=" * 120)
        print("RUNNING QUICK BENCHMARK (with plots)")
        print("=" * 120)
        df = run_quick_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Run comprehensive benchmark
        print("=" * 120)
        print("STARTING COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 120)
        print("Note: This will take a while due to Helion autotuning.")
        print("=" * 120)
        df = run_comprehensive_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == '--bucketed':
        # Run bucketed config benchmark
        print("=" * 120)
        print("RUNNING BUCKETED CONFIG BENCHMARK")
        print("=" * 120)
        print("Testing how well Helion's bucketed configs generalize to unseen shapes.")
        print("=" * 120)
        df = run_bucketed_benchmark()
    else:
        # Run single benchmark
        print("=" * 120)
        print("SINGLE BENCHMARK")
        print("=" * 120)
        print("Run with '--quick' for quick benchmark with plots")
        print("Run with '--full' for comprehensive benchmark")
        print("Run with '--bucketed' to test bucketed config generalization")
        print("=" * 120)
        benchmark_softmax(4096, 4096)

