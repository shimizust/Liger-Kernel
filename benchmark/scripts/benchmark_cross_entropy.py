import os

import torch
import triton
from torch.nn import CrossEntropyLoss
from utils import _test_memory, get_current_file_directory, get_gpu_name, update_benchmark_data_csv, SingleBenchmarkRunInput, SingleBenchmarkRunOuput, BenchmarkData, get_mean_std_stats, LIGER_KERNEL_VERSION, run_benchmarks
from typing import List, Union, Dict, Any, Callable, Optional
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
import json
from dataclasses import dataclass
import time


def bench_memory_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOuput:
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    V = input.x
    provider = input.kernel_provider
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]

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

    mem_mean, mem_std = _test_memory(full)
    return SingleBenchmarkRunOuput(
        y_mean=mem_mean,
        y_std=mem_std,
    )


def bench_speed_cross_entropy(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOuput:
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    V = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode
    B = input.extra_benchmark_config["B"]
    T = input.extra_benchmark_config["T"]

    _input = torch.randn(B * T, V, requires_grad=True, device="cuda")
    target = torch.randint(V, (B * T, 1), device="cuda").squeeze(1)

    def fwd():
        if provider == "liger":
            return liger_ce(_input, target)
        else:
            return torch_ce(_input, target)

    if mode == "forward":
        y_mean = triton.testing.do_bench(fwd, rep=100, return_mode="mean")
    elif mode == "backward":
        y = fwd()

        y_mean = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[_input],
            rep=100,
            return_mode="mean",
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        y_mean = triton.testing.do_bench(full, rep=100, return_mode="mean")

    return SingleBenchmarkRunOuput(
        y_mean=y_mean,
    )


if __name__ == "__main__":
    common_configs = {
        "kernel_name": "cross_entropy",
        "x_name": "V",
        "x_label": "vocab size",
        "x_values": [2**i for i in range(12, 18)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [{"B": 8, "T": 2048}],
        "overwrite": True
    }

    run_benchmarks(
        bench_test_fn=bench_speed_cross_entropy,
        kernel_operation_modes=["forward", "full"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_cross_entropy,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )
