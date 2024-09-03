import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

LIGER_KERNEL_VERSION = version("liger-kernel")


@dataclass
class SingleBenchmarkRunInput:
    x: Union[int, float]
    kernel_provider: str
    kernel_operation_mode: Optional[str] = ""
    extra_benchmark_config: Optional[Dict[str, Any]] = None


@dataclass
class SingleBenchmarkRunOutput:
    y_mean: float
    # Triton 3.0.0 doesn't yet support returning all values from do_bench(), so make this optional
    # This change adds support for return_mode='all': https://github.com/triton-lang/triton/pull/4493
    y_std: Optional[float] = None


@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a a completed benchmark
    run on all x-values for a given kernel/kernel operation mode/metric/extra_benchmark_config
    """

    kernel_name: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_values: List[float]
    y_values_mean: List[float]
    timestamp: str
    kernel_operation_mode: Optional[str] = None
    y_values_std: Optional[List[float]] = None
    extra_benchmark_config_str: Optional[str] = None
    liger_version: str = LIGER_KERNEL_VERSION


@dataclass
class BenchmarkDataCSVRow:
    # The ordering of field names here will be the order of columns in the CSV
    kernel_name: str
    kernel_provider: str
    kernel_operation_mode: Union[str, None]
    metric_name: str
    metric_unit: str
    x_name: str
    x_label: str
    x_value: float
    y_value_mean: float
    y_value_std: Union[str, None]
    extra_benchmark_config_str: Union[str, None]
    gpu_name: str
    timestamp: str
    liger_version: str


def get_mean_std_stats(values: List[float]) -> Tuple[float, float]:
    """
    Calculate the mean and standard deviation of the values.

    Args:
        values: The list of values.

    Returns:
        Tuple[float, float]: The mean and standard deviation of the values.
    """
    return np.mean(values), np.std(values)


def _test_memory(func: Callable, _iter: int = 10) -> Tuple[float, float]:
    """
    Run the function `func` `_iter` times and return the mean and standard deviation of the
    peak allocated memory in MB.

    Args:
        func: The function to run.
        _iter: The number of times to run the function.

    Returns:
        Tuple[float, float]: The mean and standard deviation of the peak allocated memory in MB.
    """
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        # Memory returned in bytes
        mem = torch.cuda.max_memory_allocated()
        total_mem.append(mem)

    mean, std = get_mean_std_stats(total_mem)
    # Convert bytes to MB
    return mean / 2**20, std / 2**20


def get_current_file_directory() -> str:
    """
    Returns the directory path of the current Python file.
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    return os.path.dirname(current_file_path)


def sleep(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            time.sleep(seconds)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def _print_benchmarking_banner(metric_name: str, kernel_name: str):
    print("**************************************")
    print(f"     BENCHMARKING {metric_name.upper()} for {kernel_name.upper()}")
    print("**************************************")


def get_formatted_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_gpu_name():
    """
    Returns the current GPU name, formatted to serve as a directory name
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return gpu_name
    else:
        raise Exception("Benchmarks can only be run on GPU.")


def update_benchmark_data_csv(
    benchmark_data_list: List[BenchmarkData],
    filename: str = "all_benchmark_data.csv",
    overwrite: bool = True,
):
    """
    Update the CSV file with the new benchmark data. If the file does not exist, create it.
    If an entry already exists for the benchmark, then overwrite it if `overwrite` is True.
    """

    def create_unique_key(row):
        # This unique key is used to determine if a benchmark run already exists in the CSV
        # If the key is the same, then the benchmark run already exists and will optionally
        # be overwritten. Otherwise, it is considered a new benchmark run and appended.
        return (
            row["kernel_name"],
            row["kernel_provider"],
            row["kernel_operation_mode"] if row["kernel_operation_mode"] else "",
            row["metric_name"],
            row["x_name"],
            str(row["x_value"]),
            (
                row["extra_benchmark_config_str"]
                if row["extra_benchmark_config_str"]
                else ""
            ),
            row["gpu_name"],
        )

    fieldnames = BenchmarkDataCSVRow.__annotations__.keys()

    # Make filename path relative to current file
    filename_abs_path = os.path.join(get_current_file_directory(), "../data", filename)
    file_exists = os.path.isfile(filename_abs_path)

    # Read existing data into a list of dicts
    existing_data = []
    if file_exists:
        with open(filename_abs_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_data.append(row)

    existing_data_dict = OrderedDict(
        (create_unique_key(row), row) for row in existing_data
    )

    for benchmark_data in benchmark_data_list:
        benchmark_data_dict = asdict(benchmark_data)
        x_values = benchmark_data_dict.pop("x_values")
        y_values_mean = benchmark_data_dict.pop("y_values_mean")
        y_values_std = benchmark_data_dict.pop("y_values_std", [])

        # Need to convert benchmark_data into multiple rows based on x_values and y_values
        for x_value, y_value_mean, y_value_std in zip_longest(
            x_values, y_values_mean, y_values_std
        ):
            row = BenchmarkDataCSVRow(
                x_value=x_value,
                y_value_mean=y_value_mean,
                y_value_std=y_value_std,
                **benchmark_data_dict,
            )
            row_dict = asdict(row)

            row_key = create_unique_key(row_dict)

            if row_key in existing_data_dict:
                if overwrite:
                    # If overwriting, update the row
                    existing_data_dict[row_key] = row_dict
                else:
                    # If not overwriting, skip this row
                    pass
            else:
                existing_data_dict[row_key] = row_dict

    with open(filename_abs_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_data_dict.values():
            writer.writerow(row)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(self, obj)


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
    extra_benchmark_configs: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = False,
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput as input and
    saves data to the CSV file.

    Args:
        - bench_test_fn: The benchmark test function to run. This function should take in a
            SingleBenchmarkRunInput as input and return a SingleBenchmarkRunOutput.
        - kernel_name: The name of the kernel being benchmarked (e.g. "swiglu")
        - metric_name: The name of the metric being benchmarked (e.g. "speed" or "memory")
        - metric_unit: The unit of the metric being benchmarked (e.g. "ms" or "MB")
        - x_name: The name of the x-axis (e.g. "T" for sequence length)
        - x_label: The label of the x-axis (e.g. "sequence length")
        - x_values: The list of x-values to run the benchmark on (e.g. [2**i for i in range(10, 14)])
        - kernel_providers: The list of kernel providers to run the benchmark on (e.g. ["liger", "huggingface"])
        - kernel_operation_modes: The list of kernel operation modes to run the benchmark on (e.g. ["full", "backward"])
        - extra_benchmark_configs: The list of extra benchmark configurations to run the benchmark on.
        - overwrite: Whether to overwrite the existing benchmark data entry if it already exists.
    """

    assert len(kernel_operation_modes) >= 1
    assert len(kernel_providers) >= 1

    _print_benchmarking_banner(metric_name=metric_name, kernel_name=kernel_name)

    gpu_name = get_gpu_name()
    benchmark_data_list = []
    for extra_benchmark_config in extra_benchmark_configs:
        for kernel_operation_mode in kernel_operation_modes:
            for kernel_provider in kernel_providers:
                y_values_mean = []
                y_values_std = []

                for x in x_values:
                    single_benchmark_run_input = SingleBenchmarkRunInput(
                        x=x,
                        kernel_provider=kernel_provider,
                        kernel_operation_mode=kernel_operation_mode,
                        extra_benchmark_config=extra_benchmark_config,
                    )
                    benchmark_result: SingleBenchmarkRunOutput = bench_test_fn(
                        single_benchmark_run_input
                    )
                    y_values_mean.append(benchmark_result.y_mean)
                    y_values_std.append(benchmark_result.y_std)

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
                    y_values_mean=y_values_mean,
                    y_values_std=y_values_std,
                    extra_benchmark_config_str=json.dumps(
                        extra_benchmark_config, cls=CustomEncoder
                    ),
                    timestamp=get_formatted_time(),
                    liger_version=LIGER_KERNEL_VERSION,
                )

                benchmark_data_list.append(benchmark_run_data)

    update_benchmark_data_csv(
        benchmark_data_list=benchmark_data_list, overwrite=overwrite
    )


def parse_benchmark_script_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for Liger-Kernel")

    # Add an optional --overwrite flag
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to overwrite existing benchmark data with current run.",
    )

    args = parser.parse_args()
    return args
