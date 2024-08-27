import os

import torch
import triton
from torch.nn import CrossEntropyLoss
from utils import _test_memory, get_current_file_directory

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss


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

@dataclass
class SingleBenchmarkRunInput:
    x: int
    provider: str
    mode: str
    extra_args: Dict[str, Any]

@dataclass

def bench_speed_cross_entropy(input: SingleBenchmarkRunInput):
    torch_ce = CrossEntropyLoss()
    liger_ce = LigerCrossEntropyLoss()

    V = input.x
    provider = input.provider
    mode = input.mode
    B = input.extra_args["B"]
    T = input.extra_args["T"]

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

    
    return ms, min_ms, max_ms

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

bench_test_fn = bench_speed_cross_entropy
providers = ["liger", "huggingface"]
x_values = [2**i for i in range(12, 18)]
modes = [None]
extra_args = {"B": 8, "T": 2048}

run_benchmarks(bench_test_fn, x_values, providers, modes, extra_args)

def run_benchmarks(
    bench_test_fn: Callable,
    x_values: List[Any],
    providers: List[str],
    modes: Optional[List[str]] = [None],
    extra_args: Optional[Dict[str, Any]] = None
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput
    a set of x_values to run, provider (e.g. liger, huggingface), optional mode (e.g. forward, backward, full)
    to run the kernel, and extra_args for running the benchmark functions. Extra_args should 
    """

    assert len(modes) >= 1
    assert len(providers) >= 1

    benchmark_runs = [
        SingleBenchmarkRunInput(x=2**i, provider="liger", mode="forward", extra_args={"B": 8, "T": 2048}),
        SingleBenchmarkRunInput(x=2**i, provider="huggingface", mode="forward", extra_args={"B": 8, "T": 2048}),
        SingleBenchmarkRunInput(x=2**i, provider="liger", mode="full", extra_args={"B": 8, "T": 2048}),
        SingleBenchmarkRunInput(x=2**i, provider="huggingface", mode="full", extra_args={"B": 8, "T": 2048}),
    ]

    results = []
    for benchmark_run in benchmark_runs:
        results.append(bench_speed_cross_entropy(benchmark_run))
    return results

def benchmark_speed_cross_entropy_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "cross_entropy_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    dfs = bench_speed_cross_entropy.run(
        save_path=output_dir, print_data=False, return_df=True
    )
    for df in dfs:
        print(df.head())
        print(df.info())


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


def benchmark_memory_cross_entropy_wrapper():
    curr_dir = get_current_file_directory()
    dir_name = "cross_entropy_memory"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    bench_memory_cross_entropy.run(save_path=output_dir, print_data=True)


if __name__ == "__main__":
    benchmark_speed_cross_entropy_wrapper()
    benchmark_memory_cross_entropy_wrapper()
