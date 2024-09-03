import torch
import triton
import torch.nn as nn
from utils import (
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)
from liger_kernel.transformers.rms_norm import LigerRMSNorm


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def bench_speed_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    triton_rms = LigerRMSNorm(hidden_size=N, eps=eps).to("cuda")
    llama_rms = LlamaRMSNorm(hidden_size=N, eps=eps).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions

    def y_fwd():
        if provider == "liger":
            return triton_rms(x)

        if provider == "huggingface":
            return llama_rms(x)

    if mode == "forward":
        time_mean = triton.testing.do_bench(
            y_fwd, grad_to_none=[x], rep=500, return_mode="mean"
        )
    elif mode == "backward":
        y = y_fwd()
        time_mean = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            grad_to_none=[x],
            rep=500,
            return_mode="mean"
        )
    elif mode == "full":

        def full():
            y = y_fwd()
            y.backward(dy, retain_graph=True)

        time_mean = triton.testing.do_bench(
            full, grad_to_none=[x], rep=500, return_mode="mean"
        )

    return SingleBenchmarkRunOutput(
        y_mean=time_mean,
    )


def bench_memory_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    triton_rms = LigerRMSNorm(hidden_size=N, eps=eps).to("cuda")
    llama_rms = LlamaRMSNorm(hidden_size=N, eps=eps).to("cuda")

    x = torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = torch.randn_like(x)
    x.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger":
            return triton_rms(x)
        if provider == "huggingface":
            return llama_rms(x)

    def full():
        y = y_fwd()
        y.backward(dy, retain_graph=True)

    mem_mean, mem_std = _test_memory(full)

    return SingleBenchmarkRunOutput(
        y_mean=mem_mean,
        y_std=mem_std,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "rms_norm",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [{"M": 2048, "dtype": torch.bfloat16, "eps": 1e-6}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_rms_norm,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_rms_norm,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )
