import torch
import triton
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from torch.nn import CrossEntropyLoss
from utils import (
    SingleBenchmarkRunInput,
    SingleBenchmarkRunOutput,
    _test_memory,
    parse_benchmark_script_args,
    run_benchmarks,
)

from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


SLEEP_SECONDS = 0.1

def bench_speed_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    bsz = extra_benchmark_config["B"]
    hidden_size = extra_benchmark_config["hidden_size"]
    dtype = extra_benchmark_config["dtype"]
    intermediate_size = extra_benchmark_config["intermediate_size"]
    hidden_act = extra_benchmark_config["hidden_act"]

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )

    x_shape = (bsz, seq_len, hidden_size)
    device = 'cuda'

    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for SwiGLU")

    def fwd():
        return layer(x)

    if mode == "forward":
        time_mean = triton.testing.do_bench(
            fwd, grad_to_none=[x], rep=10, return_mode="mean",
        )
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        time_mean = triton.testing.do_bench(
            lambda: y.backward(do, retain_graph=True),
            grad_to_none=[x],
            rep=10,
            return_mode="mean",
        )
    else:

        def full():
            y = fwd()
            y.backward(torch.randn_like(y), retain_graph=True)

        time_mean = triton.testing.do_bench(
            full, grad_to_none=[x], rep=10, return_mode="mean"
        )

    return SingleBenchmarkRunOutput(
        y_mean=time_mean
    )



def bench_memory_swiglu(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    seq_len = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    bsz = extra_benchmark_config["B"]
    hidden_size = extra_benchmark_config["hidden_size"]
    dtype = extra_benchmark_config["dtype"]
    intermediate_size = extra_benchmark_config["intermediate_size"]
    hidden_act = extra_benchmark_config["hidden_act"]

    llama_config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
    )

    x_shape = (bsz, seq_len, hidden_size)
    device = 'cuda'
    
    # initialize input
    x = torch.randn(*x_shape, device=device, dtype=dtype, requires_grad=True)

    if provider == "liger":
        layer = LigerSwiGLUMLP(config=llama_config).to(device).to(dtype)
    elif provider == "huggingface":
        layer = LlamaMLP(config=llama_config).to(device).to(dtype)
    else:
        raise ValueError(f"Invalid provider: {provider} for SwiGLU")

    def fwd():
        return layer(x)

    def full():
        y = fwd()
        y.backward(torch.randn_like(y), retain_graph=True)

    if mode == "forward":
        mem_mean, mem_std = _test_memory(fwd)
    elif mode == "backward":
        do = torch.randn_like(x)
        y = fwd()
        mem_mean, mem_std = _test_memory(lambda: y.backward(do, retain_graph=True))
    else:
        mem_mean, mem_std = _test_memory(full)

    return SingleBenchmarkRunOutput(
        y_mean=mem_mean,
        y_std=mem_std,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "swiglu",
        "x_name": "T",
        "x_label": "sequence length",
        "x_values": [2**i for i in range(10, 14)],
        "kernel_providers": ["liger", "huggingface"],
        "extra_benchmark_configs": [{"B": 4, "hidden_size": 4096, "dtype": torch.bfloat16, "intermediate_size": 11008, "hidden_act": "silu"}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_swiglu,
        kernel_operation_modes=["forward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs
    )
    run_benchmarks(
        bench_test_fn=bench_memory_swiglu,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs
    )