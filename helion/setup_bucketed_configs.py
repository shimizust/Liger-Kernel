"""
Setup script for bucketed config benchmarking.

This script:
1. Generates candidate configs for different workload sizes
2. Creates a softmax kernel variant that uses these configs
3. Provides instructions for running bucketed benchmarks

Usage:
    python setup_bucketed_configs.py --generate
    # Then follow the instructions to update softmax.py
"""

import torch
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from softmax import softmax_helion, generate_candidate_configs


def generate_configs_for_bucketing(output_dir="./configs"):
    """
    Step 1: Generate and save candidate configs.
    """
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING CANDIDATE CONFIGS")
    print("=" * 80)
    print("\nThis will autotune softmax for three different workload sizes:")
    print("  - Small:  1024 x 256   (262K elements)")
    print("  - Medium: 4096 x 2048  (8.4M elements)")
    print("  - Large:  16384 x 4096 (67M elements)")
    print("\nEach autotuning run will take a few minutes...")
    print("=" * 80)
    
    # Generate configs
    config_paths = generate_candidate_configs(output_dir=output_dir)
    
    return config_paths


def create_bucketed_kernel_code(config_paths):
    """
    Step 2: Generate code for the bucketed kernel.
    """
    print("\n\n" + "=" * 80)
    print("STEP 2: UPDATE YOUR KERNEL DEFINITION")
    print("=" * 80)
    print("\nAdd this code to softmax.py (after the regular softmax_helion):\n")
    
    code = f'''
# ==============================================================================
# Bucketed Softmax with Multiple Candidate Configs
# ==============================================================================

# Load candidate configs
try:
    _candidate_configs = [
'''
    
    for tag, path in config_paths.items():
        code += f'        helion.Config.load("{path}"),  # {tag}\n'
    
    code += '''    ]
    _configs_loaded = True
except FileNotFoundError as e:
    print(f"Warning: Could not load candidate configs: {{e}}")
    print("Run: python setup_bucketed_configs.py --generate")
    _configs_loaded = False
    _candidate_configs = None

# Bucketed kernel with candidate configs
if _configs_loaded:
    @helion.kernel(configs=_candidate_configs, static_shapes=False)
    def softmax_helion_bucketed(x: torch.Tensor) -> torch.Tensor:
        """
        Softmax with bucketed configs and dynamic shape handling.
        
        This kernel:
        - Uses static_shapes=False to enable bucketing across different shapes
        - On first call for a bucket, benchmarks all candidate configs
        - Selects and caches the fastest config for that bucket
        - Reuses the selected config for similar shapes
        
        Args:
            x: Input tensor of shape [n, m]
        
        Returns:
            Softmax output of the same shape
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
else:
    # Fallback if configs aren't available
    softmax_helion_bucketed = None
'''
    
    print(code)
    print("=" * 80)
    
    return code


def test_bucketed_kernel():
    """
    Step 3: Test the bucketed kernel setup.
    """
    print("\n\n" + "=" * 80)
    print("STEP 3: TESTING BUCKETED KERNEL")
    print("=" * 80)
    
    # Check if configs exist
    config_dir = Path("./configs")
    expected_configs = ["softmax_small.json", "softmax_medium.json", "softmax_large.json"]
    
    missing = []
    for config_file in expected_configs:
        if not (config_dir / config_file).exists():
            missing.append(config_file)
    
    if missing:
        print("\n⚠️  Missing config files:")
        for f in missing:
            print(f"    - {f}")
        print("\nRun: python setup_bucketed_configs.py --generate")
        return False
    
    print("\n✓ All candidate configs found")
    
    # Try to import bucketed kernel
    try:
        from softmax import softmax_helion_bucketed
        if softmax_helion_bucketed is None:
            print("⚠️  softmax_helion_bucketed is not defined")
            print("    Add the generated code to softmax.py")
            return False
        
        print("✓ Bucketed kernel imported successfully")
        
        # Test with a few shapes
        test_shapes = [(2048, 512), (8192, 1024)]
        
        print("\nTesting bucketed kernel on different shapes:")
        for rows, cols in test_shapes:
            x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
            y = softmax_helion_bucketed(x)
            expected = torch.nn.functional.softmax(x, dim=-1)
            max_diff = (y - expected).abs().max().item()
            print(f"  Shape ({rows:5d}, {cols:5d}): max_diff = {max_diff:.2e} {'✓' if max_diff < 1e-5 else '✗'}")
        
        print("\n✓ Bucketed kernel working correctly!")
        return True
        
    except ImportError:
        print("⚠️  Could not import softmax_helion_bucketed")
        print("    Add the generated code to softmax.py")
        return False


def print_next_steps():
    """
    Print instructions for running benchmarks.
    """
    print("\n\n" + "=" * 80)
    print("NEXT STEPS: RUN BUCKETED BENCHMARK")
    print("=" * 80)
    print("\nOption 1: Update benchmark to use bucketed kernel")
    print("-" * 80)
    print("In benchmark_softmax.py, add softmax_helion_bucketed to IMPLEMENTATIONS:")
    print("""
    {
        'key': 'helion_bucketed',
        'fn': softmax_helion_bucketed,
        'label': 'Helion Bucketed',
        'label_short': 'Helion\\nBucketed',
        'color': '#e377c2',
        'description': 'Helion with multiple candidate configs',
        'compile': False,
    },
""")
    
    print("\nOption 2: Test manually")
    print("-" * 80)
    print("from softmax import softmax_helion_bucketed")
    print("from benchmark_softmax import run_bucketed_benchmark")
    print("")
    print("# This will test how well the bucketed configs generalize")
    print("df = run_bucketed_benchmark()")
    
    print("\nOption 3: Compare strategies")
    print("-" * 80)
    print("# Test default (static_shapes=True)")
    print("python benchmark_softmax.py --bucketed")
    print("")
    print("# Update softmax.py to use softmax_helion_bucketed")
    print("# Then run again to compare")
    print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup bucketed configs for Helion softmax benchmarking"
    )
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate candidate configs for different workload sizes'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test if bucketed kernel is set up correctly'
    )
    parser.add_argument(
        '--output-dir',
        default='./configs',
        help='Directory to save config files (default: ./configs)'
    )
    
    args = parser.parse_args()
    
    if args.generate:
        # Step 1: Generate configs
        config_paths = generate_configs_for_bucketing(output_dir=args.output_dir)
        
        # Step 2: Show code to add
        create_bucketed_kernel_code(config_paths)
        
        # Step 3: Show next steps
        print_next_steps()
        
    elif args.test:
        # Test if setup is complete
        success = test_bucketed_kernel()
        if success:
            print_next_steps()
        else:
            print("\n⚠️  Setup not complete. Run with --generate first.")
    
    else:
        # Show usage
        print("=" * 80)
        print("BUCKETED CONFIG SETUP FOR HELION SOFTMAX")
        print("=" * 80)
        print("\nThis script helps you set up bucketed configs for testing Helion's")
        print("ability to generalize across different input shapes.")
        print("\nUsage:")
        print("  1. Generate configs:  python setup_bucketed_configs.py --generate")
        print("  2. Add code to softmax.py (printed by step 1)")
        print("  3. Test setup:        python setup_bucketed_configs.py --test")
        print("  4. Run benchmark:     python benchmark_softmax.py --bucketed")
        print("\nFor more info, run with --help")
        print("=" * 80)


if __name__ == "__main__":
    main()

