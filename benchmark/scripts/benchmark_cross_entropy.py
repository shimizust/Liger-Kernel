import os

import torch
import triton
from torch.nn import CrossEntropyLoss
from utils import _test_memory, get_current_file_directory, get_gpu_name, update_benchmark_data_csv, SingleBenchmarkRunInput, BenchmarkData
from typing import List, Union, Dict, Any, Callable, Optional
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
import json
from dataclasses import dataclass
import time

# @triton.testing.perf_report(
#     [
#         triton.testing.Benchmark(
#             x_names=["V"],
#             x_vals=[2**i for i in range(12, 18)],
#             xlabel="vocab size",
#             line_arg="provider",
#             line_vals=["liger", "huggingface"],
#             line_names=[
#                 "Liger",
#                 "Hugging Face",
#             ],
#             styles=[
#                 ("blue", "solid"),
#                 ("orange", "solid"),
#             ],
#             ylabel="time (ms)",
#             plot_name="cross-entropy-fwd-speed-benchmark",
#             args={"B": 8, "T": 2048, "mode": "forward", "dtype": torch.bfloat16},
#         ),
#         triton.testing.Benchmark(
#             x_names=["V"],
#             x_vals=[2**i for i in range(12, 18)],
#             xlabel="vocab size",
#             line_arg="provider",
#             line_vals=["liger", "huggingface"],
#             line_names=["Liger", "Hugging Face"],
#             styles=[
#                 ("blue", "solid"),
#                 ("orange", "solid"),
#             ],
#             ylabel="time (ms)",
#             plot_name="cross-entropy-full-speed-benchmark",
#             args={"B": 8, "T": 2048, "mode": "full", "dtype": torch.bfloat16},
#         ),
#     ]
# )
# def bench_speed_cross_entropy(B, T, V, provider, mode, dtype, device="cuda"):
#     torch_ce = CrossEntropyLoss()
#     liger_ce = LigerCrossEntropyLoss()

#     _input = torch.randn(B * T, V, requires_grad=True, device="cuda")
#     target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

#     def fwd():
#         if provider == "liger":
#             return liger_ce(_input, target)
#         else:
#             return torch_ce(_input, target)

#     quantiles = [0.5, 0.2, 0.8]

#     if mode == "forward":
#         ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=100)
#     elif mode == "backward":
#         y = fwd()

#         ms, min_ms, max_ms = triton.testing.do_bench(
#             lambda: y.backward(retain_graph=True),
#             quantiles=quantiles,
#             grad_to_none=[_input],
#             rep=100,
#         )
#     elif mode == "full":

#         def full():
#             y = fwd()
#             y.backward()

#         ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=quantiles, rep=100)

    
#     return ms, min_ms, max_ms




def bench_speed_cross_entropy(input: SingleBenchmarkRunInput):
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    V = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    print(mode)
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda")
    target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    quantiles = [0.5, 0.2, 0.8]

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(fwd, quantiles=quantiles, rep=100)
    elif mode == "backward":
        y = fwd()

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[_input],
            rep=100,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms, min_ms, max_ms = triton.testing.do_bench(full, quantiles=quantiles, rep=100)

    return ms
    # return ms, min_ms, max_ms

#             x_names=["V"],
#             x_vals=[2**i for i in range(12, 18)],
#             xlabel="vocab size",
#             line_arg="provider",
#             line_vals=["liger", "huggingface"],
#             line_names=[
#                 "Liger",
#                 "Hugging Face",
#             ],
#             styles=[
#                 ("blue", "solid"),
#                 ("orange", "solid"),
#             ],
#             ylabel="time (ms)",
#             plot_name="cross-entropy-fwd-speed-benchmark",
#             args={"B": 8, "T": 2048, "mode": "forward", "dtype": torch.bfloat16},
#         ),

# bench_test_fn = bench_speed_cross_entropy
# providers = ["liger", "huggingface"]
# x_values = [2**i for i in range(12, 18)]
# modes = ["forward", "full"]
# extra_args = {"B": 8, "T": 2048}



def run_benchmarks(
    bench_test_fn: Callable,
    kernel_name: str,
    metric_name: str,
    metric_unit: str,
    x_name: str,
    x_label: str,
    x_values: List[Union[float, int]],
    kernel_providers: List[str],
    kernel_operation_modes: Optional[List[str]] = [None],
    extra_benchmark_config: Optional[Dict[str, Any]] = None
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput
    a set of x_values to run, provider (e.g. liger, huggingface), optional mode (e.g. forward, backward, full)
    to run the kernel, and extra_args for running the benchmark functions. Extra_args should 
    """

    assert len(kernel_operation_modes) >= 1
    assert len(kernel_providers) >= 1

    gpu_name = get_gpu_name()
    benchmark_data_list = []
    for kernel_operation_mode in kernel_operation_modes:
        for kernel_provider in kernel_providers:
            y_values = []
            for x in x_values:
                single_benchmark_run_input = SingleBenchmarkRunInput(
                    x=x,
                    kernel_provider=kernel_provider,
                    kernel_operation_mode=kernel_operation_mode,
                    extra_benchmark_config=extra_benchmark_config)
                benchmark_result = bench_test_fn(single_benchmark_run_input)
                y_values.append(benchmark_result)
            
            benchmark_run_data = BenchmarkData(
                kernel_name=kernel_name,
                kernel_operation_mode=kernel_operation_mode,
                kernel_provider=kernel_provider,
                metric_name=metric_name,
                metric_unit=metric_unit,
                gpu_name=gpu_name,
                x_name=x_name,
                x_label=x_label,
                x_values=x_values,
                y_values=y_values,
                extra_benchmark_config=json.dumps(extra_benchmark_config),
                timestamp=time.time()
            )

            benchmark_data_list.append(benchmark_run_data)

    update_benchmark_data_csv(benchmark_data_list=benchmark_data_list)
            


# def benchmark_speed_cross_entropy_wrapper():
#     curr_dir = get_current_file_directory()
#     dir_name = "cross_entropy_speed"
#     output_dir = os.path.join(curr_dir, dir_name)
#     os.makedirs(output_dir, exist_ok=True)
#     dfs = bench_speed_cross_entropy.run(
#         save_path=output_dir, print_data=False, return_df=True
#     )
#     for df in dfs:
#         print(df.head())
#         print(df.info())


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["V"],
            x_vals=[2**i for i in range(12, 18)],
            xlabel="vocab size",
            line_arg="provider",
            line_vals=["liger", "huggingface"],
            line_names=[
                "Liger",
                "Hugging Face",
            ],
            styles=[
                ("blue", "solid"),
                ("orange", "solid"),
            ],
            ylabel="GPU memory usage (MB)",
            plot_name="cross-entropy-memory-benchmark",
            args={"B": 8, "T": 2048, "dtype": torch.bfloat16},
        )
    ]
)
def bench_memory_cross_entropy(B, T, V, provider, dtype, device="cuda"):
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda")
    target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem = _test_memory(full)
    return mem / 2**20


# def benchmark_memory_cross_entropy_wrapper():
#     curr_dir = get_current_file_directory()
#     dir_name = "cross_entropy_memory"
#     output_dir = os.path.join(curr_dir, dir_name)
#     os.makedirs(output_dir, exist_ok=True)
#     bench_memory_cross_entropy.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    run_benchmarks(
        bench_test_fn=bench_speed_cross_entropy,
        kernel_name="cross_entropy",
        kernel_operation_modes=["forward"],
        metric_name="speed",
        metric_unit="ms",
        x_name="V",
        x_label="vocab size",
        x_values=[2**i for i in range(12, 18)],
        kernel_providers=["liger", "huggingface"],
        extra_benchmark_config={"B": 8, "T": 2048}
    )
    # benchmark_speed_cross_entropy_wrapper()
    # benchmark_memory_cross_entropy_wrapper()
