import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Any, Dict, Optional
import csv
import torch


@dataclass
class SingleBenchmarkRunInput:
    x: int
    kernel_provider: str
    kernel_operation_mode: str
    extra_benchmark_config: Dict[str, Any]


@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a single benchmark run.
    For example, the data collected after running rms_norm speed benchmark from a single provider.
    """
    kernel_name: str
    kernel_operation_mode: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_values: List[Any]
    y_values: List[Any]
    extra_benchmark_config: str
    timestamp: str


@dataclass
class BenchmarkDataCSVRow:
    kernel_name: str
    kernel_operation_mode: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_value: float
    y_value: float
    extra_benchmark_config: str
    timestamp: str


def _test_memory(func: Callable, _iter: int = 10) -> float:
    total_mem = []

    for _ in range(_iter):
        torch.cuda.memory.reset_peak_memory_stats()
        func()
        mem = torch.cuda.max_memory_allocated()
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)


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


def _print_memory_banner():
    print("**************************************")
    print("*     BENCHMARKING GPU MEMORY        *")
    print("**************************************")


def _print_speed_banner():
    print("**************************************")
    print("*        BENCHMARKING SPEED          *")
    print("**************************************")


def get_gpu_name():
    """
    Returns the current GPU name, formatted to serve as a directory name
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        return gpu_name
        # return gpu_name.lower().replace(" ", "_")
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

    fieldnames = BenchmarkDataCSVRow.__annotations__.keys()

    # Make filename path relative to current file
    filename_abs_path = os.path.join(get_current_file_directory(), "../data", filename)
    file_exists = os.path.isfile(filename_abs_path)

    with open(filename_abs_path, mode="a") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for benchmark_data in benchmark_data_list:
            benchmark_data_dict = asdict(benchmark_data)
            x_values = benchmark_data_dict.pop("x_values")
            y_values = benchmark_data_dict.pop("y_values")

            # Need to convert benchmark_data into multiple rows based on x_values and y_values
            for x_value, y_value in zip(x_values, y_values):
                row = BenchmarkDataCSVRow(
                    x_value=x_value, y_value=y_value, **benchmark_data_dict
                )
                writer.writerow(asdict(row))
