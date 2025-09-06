# Model Patching

This section describes the supported Hugging Face model architectures for which Liger provides patching functions. Each `apply_liger_kernel_to_<model>` function monkey patches the target model code to replace standard layers (like rotary embeddings, RMSNorm, activation functions, and loss layers) with more efficient implementations provided by Liger.

When using `AutoLigerKernelForCausalLM` or enabling the Liger Kernel flag in popular training frameworks (like the Hugging Face `Trainer` or `SFTTrainer`), the patching is applied automatically based on the model architecture, so you don't need to call the patching function manually.
## Supported Models

The table below lists all currently supported models, the corresponding patching API, and the key operations that are replaced or accelerated.

<!-- TODO: Auto-generate this table. We add this list manually for now because mkdocs cuts off the name of functions in the table of contents, so it's not clear which functions are available. -->

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| Gemma1      | [apply_liger_kernel_to_gemma](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_gemma)    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2      | [apply_liger_kernel_to_gemma2](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_gemma2)   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Text) | [apply_liger_kernel_to_gemma3_text](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_gemma3_text) | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Multimodal) | [apply_liger_kernel_to_gemma3](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_gemma3) | RoPE, RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4       | [apply_liger_kernel_to_glm4](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_glm4) | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4v      | [apply_liger_kernel_to_glm4v](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_glm4v) | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Granite 3.0 & 3.1 | [apply_liger_kernel_to_granite](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_granite) | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| LLaMA 2 & 3 | [apply_liger_kernel_to_llama](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_llama)   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 4 (Text & Multimodal) | [apply_liger_kernel_to_llama4](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_llama4) | RoPE, RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| LLaMA 3.2-Vision | [apply_liger_kernel_to_mllama](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_mllama)   | RoPE, RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| LLaVA       | [apply_liger_kernel_to_llava](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_llava) | CrossEntropyLoss, FusedLinearCrossEntropy |
| Mistral     | [apply_liger_kernel_to_mistral](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_mistral)  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral     | [apply_liger_kernel_to_mixtral](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_mixtral)  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| OLMO2       | [apply_liger_kernel_to_olmo2](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_olmo2) | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| PaliGemma   | [apply_liger_kernel_to_paligemma](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_paligemma) | RoPE, RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Phi3 & Phi3.5 | [apply_liger_kernel_to_phi3](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_phi3)     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ | [apply_liger_kernel_to_qwen2](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_qwen2)    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL    | [apply_liger_kernel_to_qwen2_vl](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_qwen2_vl) | RoPE, RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen2.5-VL  | [apply_liger_kernel_to_qwen2_5_vl](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_qwen2_5_vl) | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Qwen3       | [apply_liger_kernel_to_qwen3](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_qwen3)    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3 MoE   | [apply_liger_kernel_to_qwen3_moe](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_qwen3_moe) | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| SmolLM3     | [apply_liger_kernel_to_smollm3](#liger_kernel.transformers.monkey_patch.apply_liger_kernel_to_smollm3) | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |

## Example Usage

```python
import transformers
from liger_kernel.transformers import apply_liger_kernel_to_llama

# 1a. Adding this line automatically monkey-patches the model with the optimized Liger kernels
apply_liger_kernel_to_llama()

# 1b. You could alternatively specify exactly which kernels are applied
apply_liger_kernel_to_llama(
  rope=True,
  swiglu=True,
  cross_entropy=True,
  fused_linear_cross_entropy=False,
  rms_norm=False
)

# 2. Instantiate patched model
model = transformers.AutoModelForCausalLM("path/to/llama/model")
```


::: liger_kernel.transformers.monkey_patch