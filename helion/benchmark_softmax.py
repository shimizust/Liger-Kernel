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
    
    # Benchmark native PyTorch softmax
    print("Benchmarking Native PyTorch softmax (torch.nn.functional.softmax)...")
    native_time = do_bench(lambda: torch.nn.functional.softmax(x, dim=-1), warmup=25, rep=100)
    results['pytorch_native'] = native_time
    print(f"Native PyTorch: {native_time:.4f} ms")
    
    # Benchmark custom PyTorch implementation
    print("Benchmarking Custom PyTorch softmax...")
    pytorch_time = do_bench(lambda: softmax_pytorch(x), warmup=25, rep=100)
    results['pytorch_custom'] = pytorch_time
    print(f"Custom PyTorch: {pytorch_time:.4f} ms")
    
    # Benchmark Triton (persistent kernel)
    print("Benchmarking Triton softmax (persistent)...")
    triton_time = do_bench(lambda: softmax_triton(x), warmup=25, rep=100)
    results['triton_persistent'] = triton_time
    print(f"Triton (persistent): {triton_time:.4f} ms")
    
    # Benchmark Liger Triton
    print("Benchmarking Liger Triton softmax...")
    liger_time = do_bench(lambda: softmax_liger_triton(x), warmup=25, rep=100)
    results['liger_triton'] = liger_time
    print(f"Liger Triton: {liger_time:.4f} ms")
    
    # Benchmark Helion
    print("Benchmarking Helion softmax...")
    helion_time = do_bench(lambda: softmax_helion(x), warmup=25, rep=100)
    results['helion'] = helion_time
    print(f"Helion: {helion_time:.4f} ms")
    
    # Calculate speedups relative to native PyTorch
    print("\n" + "=" * 80)
    print("Speedup relative to Native PyTorch:")
    print("=" * 80)
    for name, time in results.items():
        if name != 'pytorch_native':
            speedup = native_time / time
            print(f"{name:20s}: {speedup:.2f}x")
    
    # Find fastest implementation
    fastest = min(results.items(), key=lambda x: x[1])
    print(f"\nFastest: {fastest[0]} ({fastest[1]:.4f} ms)")
    
    return results


def plot_timing_comparison(df, output_file='timing_comparison.png'):
    """
    Create a bar chart comparing execution times across different implementations.
    """
    output_path = BENCHMARK_DIR / output_file
    
    plt.figure(figsize=(16, 8))
    
    x = np.arange(len(df))
    width = 0.15
    
    plt.bar(x - 2.5*width, df['pytorch_native'], width, label='Native PyTorch', alpha=0.8, color='#1f77b4')
    plt.bar(x - 1.5*width, df['pytorch_custom'], width, label='Custom PyTorch', alpha=0.8, color='#17becf')
    plt.bar(x - 0.5*width, df['triton_persistent'], width, label='Triton (Persistent)', alpha=0.8, color='#ff7f0e')
    plt.bar(x + 0.5*width, df['liger_triton'], width, label='Liger Triton', alpha=0.8, color='#2ca02c')
    plt.bar(x + 1.5*width, df['helion'], width, label='Helion', alpha=0.8, color='#d62728')
    
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
    Create a bar chart comparing speedups relative to Native PyTorch.
    """
    output_path = BENCHMARK_DIR / output_file
    
    plt.figure(figsize=(16, 8))
    
    x = np.arange(len(df))
    width = 0.18
    
    plt.bar(x - 1.5*width, df['custom_speedup'], width, label='Custom PyTorch', alpha=0.8, color='#17becf')
    plt.bar(x - 0.5*width, df['triton_speedup'], width, label='Triton (Persistent)', alpha=0.8, color='#ff7f0e')
    plt.bar(x + 0.5*width, df['liger_speedup'], width, label='Liger Triton', alpha=0.8, color='#2ca02c')
    plt.bar(x + 1.5*width, df['helion_speedup'], width, label='Helion', alpha=0.8, color='#d62728')
    
    # Add horizontal line at y=1 for baseline
    plt.axhline(y=1, color='#1f77b4', linestyle='--', linewidth=2, label='Native PyTorch (Baseline)', alpha=0.7)
    
    plt.xlabel('Tensor Shape', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup vs Native PyTorch', fontsize=12, fontweight='bold')
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
    
    plt.figure(figsize=(16, 8))
    
    x = np.arange(len(df))
    width = 0.15
    
    plt.bar(x - 2.5*width, df['pytorch_native'], width, label='Native PyTorch', alpha=0.8, color='#1f77b4')
    plt.bar(x - 1.5*width, df['pytorch_custom'], width, label='Custom PyTorch', alpha=0.8, color='#17becf')
    plt.bar(x - 0.5*width, df['triton_persistent'], width, label='Triton (Persistent)', alpha=0.8, color='#ff7f0e')
    plt.bar(x + 0.5*width, df['liger_triton'], width, label='Liger Triton', alpha=0.8, color='#2ca02c')
    plt.bar(x + 1.5*width, df['helion'], width, label='Helion', alpha=0.8, color='#d62728')
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Average speedup plot
    implementations = ['Custom\nPyTorch', 'Triton\n(Persistent)', 'Liger\nTriton', 'Helion']
    avg_speedups = [
        df['custom_speedup'].mean(),
        df['triton_speedup'].mean(),
        df['liger_speedup'].mean(),
        df['helion_speedup'].mean()
    ]
    colors = ['#17becf', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax1.bar(implementations, avg_speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Native PyTorch Baseline')
    ax1.set_ylabel('Average Speedup vs Native PyTorch', fontsize=12, fontweight='bold')
    ax1.set_title('Average Speedup Across All Configurations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # Add value labels on bars
    for bar, speedup in zip(bars, avg_speedups):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Average execution time plot
    avg_times = [
        df['pytorch_native'].mean(),
        df['pytorch_custom'].mean(),
        df['triton_persistent'].mean(),
        df['liger_triton'].mean(),
        df['helion'].mean()
    ]
    impl_names = ['Native\nPyTorch', 'Custom\nPyTorch', 'Triton\n(Persistent)', 'Liger\nTriton', 'Helion']
    colors_time = ['#1f77b4', '#17becf', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars2 = ax2.bar(impl_names, avg_times, color=colors_time, alpha=0.8, edgecolor='black', linewidth=1.5)
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
    
    # Reorder columns
    cols = ['shape', 'n_rows', 'n_cols', 'pytorch_native', 'pytorch_custom', 'triton_persistent', 'liger_triton', 'helion']
    df = df[cols]
    
    # Calculate speedups relative to native PyTorch
    df['custom_speedup'] = df['pytorch_native'] / df['pytorch_custom']
    df['triton_speedup'] = df['pytorch_native'] / df['triton_persistent']
    df['liger_speedup'] = df['pytorch_native'] / df['liger_triton']
    df['helion_speedup'] = df['pytorch_native'] / df['helion']
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 120)
    print("\nTiming Results (ms):")
    print(df[['shape', 'pytorch_native', 'pytorch_custom', 'triton_persistent', 'liger_triton', 'helion']].to_string(index=False))
    
    print("\n" + "-" * 120)
    print("\nSpeedup vs Native PyTorch:")
    print(df[['shape', 'custom_speedup', 'triton_speedup', 'liger_speedup', 'helion_speedup']].to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(f"Average Custom PyTorch speedup: {df['custom_speedup'].mean():.2f}x")
    print(f"Average Triton Persistent speedup: {df['triton_speedup'].mean():.2f}x")
    print(f"Average Liger Triton speedup: {df['liger_speedup'].mean():.2f}x")
    print(f"Average Helion speedup: {df['helion_speedup'].mean():.2f}x")
    
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
    cols = ['shape', 'n_rows', 'n_cols', 'pytorch_native', 'pytorch_custom', 'triton_persistent', 'liger_triton', 'helion']
    df = df[cols]
    
    # Calculate speedups relative to native PyTorch
    df['custom_speedup'] = df['pytorch_native'] / df['pytorch_custom']
    df['triton_speedup'] = df['pytorch_native'] / df['triton_persistent']
    df['liger_speedup'] = df['pytorch_native'] / df['liger_triton']
    df['helion_speedup'] = df['pytorch_native'] / df['helion']
    
    print("\n" + "=" * 120)
    print("QUICK BENCHMARK RESULTS")
    print("=" * 120)
    print("\nTiming Results (ms):")
    print(df[['shape', 'pytorch_native', 'pytorch_custom', 'triton_persistent', 'liger_triton', 'helion']].to_string(index=False))
    
    print("\n" + "-" * 120)
    print("\nSpeedup vs Native PyTorch:")
    print(df[['shape', 'custom_speedup', 'triton_speedup', 'liger_speedup', 'helion_speedup']].to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(f"Average Custom PyTorch speedup: {df['custom_speedup'].mean():.2f}x")
    print(f"Average Triton Persistent speedup: {df['triton_speedup'].mean():.2f}x")
    print(f"Average Liger Triton speedup: {df['liger_speedup'].mean():.2f}x")
    print(f"Average Helion speedup: {df['helion_speedup'].mean():.2f}x")
    
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
    else:
        # Run single benchmark
        print("=" * 120)
        print("SINGLE BENCHMARK")
        print("=" * 120)
        print("Run with '--quick' for quick benchmark with plots")
        print("Run with '--full' for comprehensive benchmark")
        print("=" * 120)
        benchmark_softmax(4096, 4096)

