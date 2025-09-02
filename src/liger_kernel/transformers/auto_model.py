import inspect

from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
from liger_kernel.transformers.monkey_patch import _apply_liger_kernel


def _get_model_config(model_dir, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir, **model_init_kwargs)
    return config


class AutoLigerKernelForCausalLM(AutoModelForCausalLM):
    """Drop-in replacement for AutoModelForCausalLM with automatic Liger Kernel optimizations.
    
    This class automatically applies Liger Kernel optimizations to supported transformer models
    during model loading. It inherits all functionality from AutoModelForCausalLM while seamlessly
    integrating memory-efficient and performance-optimized kernels.
    
    The class automatically detects the model type and applies the appropriate Liger Kernel
    optimizations if the model architecture is supported. This includes optimized implementations
    of normalization layers, activation functions, loss functions, and other operations.
    
    Usage:
        ```python
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        
        # Load a model with automatic Liger Kernel optimizations
        model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        ```
    
    Note:
        All parameters and functionality remain the same as AutoModelForCausalLM. The only
        difference is that Liger Kernel optimizations are applied automatically when available.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_config = _get_model_config(pretrained_model_name_or_path, **kwargs)

        # Determine the model type and apply the Liger Kernel if applicable
        # Note: _apply_liger_kernel will only pass relevant kwargs to the apply_liger_kernel_to_* function
        model_type = model_config.model_type

        _apply_liger_kernel(model_type, **kwargs)

        # Filter out kwargs that were passed to the apply_liger_* function, which will cause
        # model initialization errors otherwise
        apply_fn = MODEL_TYPE_TO_APPLY_LIGER_FN[model_type]
        apply_fn_signature = inspect.signature(apply_fn)

        applicable_kwargs = {key: value for key, value in kwargs.items() if key not in apply_fn_signature.parameters}

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **applicable_kwargs)
