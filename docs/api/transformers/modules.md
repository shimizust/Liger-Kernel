<!-- # Liger Kernel Modules

## Hugging Face-Compatible Layers

These modules are drop-in replacements for Hugging Face transformer layers and are automatically patched by Liger Kernel.

::: liger_kernel.transformers.rms_norm
::: liger_kernel.transformers.layer_norm
::: liger_kernel.transformers.fused_linear_cross_entropy

## General/Advanced Kernels

These modules are available for advanced or custom use, but are **not** automatically patched into Hugging Face models.

- ::: liger_kernel.transformers.multi_token_attention
- ::: liger_kernel.transformers.sparsemax -->

::: liger_kernel.transformers.rms_norm
    options:
      members:
        - LigerRMSNorm
        - LigerRMSNormForGemma
        - LigerRMSNormForGemma2
        - LigerRMSNormForGemma3
        - LigerRMSNormForOlmo2
        - LigerRMSNormForGlm4