# Liger Kernel Modules

This section documents the individual Liger Kernel components, including both PyTorch modules and optimized functions that can be used as drop-in replacements for standard transformer operations.

## Quick Reference

| Category | Modules / Functions | Description |
|----------|---------|-------------|
| [Normalization Layers](#normalization-layers) | `LigerRMSNorm` (+ model-specific variants), `LigerLayerNorm`, `LigerFusedAddRMSNorm` | Optimized normalization implementations |
| [Activation Functions](#activation-functions) | `LigerSwiGLUMLP` (+ model-specific variants), `LigerGEGLUMLP`, `LigerBlockSparseTop2MLP` | Gated activation functions for transformers |
| [Loss Functions](#loss-functions) | `LigerCrossEntropyLoss`, `LigerJSD`, `LigerKLDIVLoss`, `LigerTVDLoss` | Memory-efficient loss implementations |
| [Position Encodings](#position-encodings) | `liger_rotary_pos_emb`, `liger_llama4_*_rotary_pos_emb`, `liger_multimodal_rotary_pos_emb` | Rotary position embedding variants |
| [Attention Mechanisms](#attention-mechanisms) | `LigerMultiTokenAttention`, `LigerFusedNeighborhoodAttention` | Optimized attention implementations |
| [Other Functions](#other-functions) | `LigerSoftmax`, `LigerSparsemax`, `LigerDyT` | Additional utility modules |

## Normalization Layers

::: liger_kernel.transformers.rms_norm
::: liger_kernel.transformers.layer_norm
::: liger_kernel.transformers.fused_add_rms_norm

## Activation Functions

::: liger_kernel.transformers.swiglu

::: liger_kernel.transformers.geglu

## Loss Functions

::: liger_kernel.transformers.cross_entropy

::: liger_kernel.transformers.fused_linear_cross_entropy

::: liger_kernel.transformers.jsd

::: liger_kernel.transformers.fused_linear_jsd

::: liger_kernel.transformers.kl_div

::: liger_kernel.transformers.tvd

## Position Encodings

::: liger_kernel.transformers.rope

::: liger_kernel.transformers.llama4_rope

::: liger_kernel.transformers.qwen2vl_mrope

## Attention Mechanisms

::: liger_kernel.transformers.multi_token_attention

::: liger_kernel.transformers.fused_neighborhood_attention

## Other Functions

::: liger_kernel.transformers.softmax

::: liger_kernel.transformers.sparsemax

::: liger_kernel.transformers.dyt