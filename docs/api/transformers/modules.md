# Liger Kernel Modules

This section documents the individual Liger Kernel components, including both PyTorch modules and optimized functions that can be used as drop-in replacements for standard transformer operations.

## Quick Reference

| Category | Module | Description | Key Classes / Functions |
|----------|--------|-------------|-------------|
| [Normalization Layers](#normalization-layers) | [`liger_kernel.transformers.rms_norm`](#liger_kernel.transformers.rms_norm) | Optimized RMS norm (+ model-specific variants) | `liger_kernel.transformers.LigerRMSNorm` |
| | [`liger_kernel.transformers.layer_norm`](#liger_kernel.transformers.layer_norm) | Optimized layer normalization | `liger_kernel.transformers.LigerLayerNorm` |
| | [`liger_kernel.transformers.fused_add_rms_norm`](#liger_kernel.transformers.fused_add_rms_norm) | Fused residual addition + RMS norm | `liger_kernel.transformers.LigerFusedAddRMSNorm` |
| | [`liger_kernel.transformers.group_norm`](#liger_kernel.transformers.group_norm) | Group normalization | `liger_kernel.transformers.LigerGroupNorm` |
| [Activation Functions](#activation-functions) | [`liger_kernel.transformers.swiglu`](#liger_kernel.transformers.swiglu) | SwiGLU MLP implementations for various models | `liger_kernel.transformers.LigerSwiGLUMLP` |
| | [`liger_kernel.transformers.geglu`](#activation-functions) | GEGLU MLP block implementation | `liger_kernel.transformers.LigerGEGLUMLP` |
| [Loss Functions](#loss-functions) | [`liger_kernel.transformers.cross_entropy`](#liger_kernel.transformers.cross_entropy) | Memory-efficient cross entropy loss | `liger_kernel.transformers.LigerCrossEntropyLoss` |
| | [`liger_kernel.transformers.fused_linear_cross_entropy`](#liger_kernel.transformers.fused_linear_cross_entropy) | Fused linear layer + cross entropy | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss` |
| | [`liger_kernel.transformers.jsd`](#liger_kernel.transformers.jsd) | Jensen-Shannon Divergence loss | `liger_kernel.transformers.LigerJSD` |
| | [`liger_kernel.transformers.kl_div`](#liger_kernel.transformers.kl_div) | KL Divergence loss implementation | `liger_kernel.transformers.LigerKLDIVLoss` |
| [Position Encodings](#position-encodings) | [`liger_kernel.transformers.rope`](#liger_kernel.transformers.rope) | Standard rotary position embedding | `liger_kernel.transformers.liger_rotary_pos_emb` |
| | [`liger_kernel.transformers.llama4_rope`](#liger_kernel.transformers.llama4_rope) | LLaMA4-specific RoPE implementations | `liger_kernel.transformers.liger_llama4_text_rotary_pos_emb` |
| | [`liger_kernel.transformers.qwen2vl_mrope`](#liger_kernel.transformers.qwen2vl_mrope) | Multimodal RoPE for vision-language models | `liger_kernel.transformers.liger_multimodal_rotary_pos_emb` |
| [Attention Mechanisms](#attention-mechanisms) | [`liger_kernel.transformers.multi_token_attention`](#liger_kernel.transformers.multi_token_attention) | Multi-token attention mechanism | `liger_kernel.transformers.LigerMultiTokenAttention` |
| | [`liger_kernel.transformers.fused_neighborhood_attention`](#liger_kernel.transformers.fused_neighborhood_attention) | Fused neighborhood attention | `liger_kernel.transformers.LigerFusedNeighborhoodAttention` |
| [Other Functions](#other-functions) | [`liger_kernel.transformers.softmax`](#liger_kernel.transformers.softmax) | Optimized softmax implementation | `liger_kernel.transformers.LigerSoftmax` |
| | [`liger_kernel.transformers.sparsemax`](#liger_kernel.transformers.sparsemax) | Sparsemax activation function | `liger_kernel.transformers.LigerSparsemax` |
| [Experimental Modules](#experimental-modules) | [`liger_kernel.transformers.experimental.embedding`](#liger_kernel.transformers.experimental.embedding) | Experimental embedding layer | `liger_kernel.transformers.experimental.LigerEmbedding` |

## Normalization Layers

::: liger_kernel.transformers.rms_norm
::: liger_kernel.transformers.layer_norm
::: liger_kernel.transformers.fused_add_rms_norm
::: liger_kernel.transformers.group_norm

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

## Experimental Modules

::: liger_kernel.transformers.experimental.embedding

## Other Functions

::: liger_kernel.transformers.softmax
::: liger_kernel.transformers.sparsemax
::: liger_kernel.transformers.dyt