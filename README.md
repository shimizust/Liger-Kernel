![Banner](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/logo-banner.png)

# Liger Kernel: Efficient Triton Kernels for LLM Training

[![PyPI Version](https://img.shields.io/pypi/v/liger-kernel?color=green)](https://pypi.org/project/liger-kernel) 
[![Downloads](https://static.pepy.tech/badge/liger-kernel)](https://pepy.tech/project/liger-kernel)
[![Build Status](https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml/badge.svg)](https://github.com/linkedin/Liger-Kernel/actions/workflows/nvi-ci.yml)

---

Liger Kernel provides high-performance Triton kernels to **train large language models faster and with lower memory**.

## 🚀 Key Features

- **Fast & Memory Efficient:** Kernel fusion and chunking reduce memory by up to 80% and boost multi-GPU throughput by 20%. The saved memory enables larger batch sizes, which means even more speedups in practice.
- **Exact:** Computation is exact--no approximations! Rigorous unit and convergence tests.
- **Simple:** Drop-in patching for Hugging Face models.
- **Compatible:** Minimal dependencies (only Torch and Triton)
- **Multi-GPU Support:** Compatible with PyTorch FSDP, DeepSpeed, DDP, etc.
- **Integrated:** Supported by [Hugging Face Trainer](https://github.com/huggingface/transformers/pull/32860), [TRL SFTTrainer](https://github.com/huggingface/trl/releases/tag/v0.10.1), [Axolotl](https://github.com/axolotl-ai-cloud/axolotl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [SWIFT](https://github.com/modelscope/ms-swift), [verl](https://github.com/volcengine/verl) and more.
- **Broad Coverage:** Optimized kernels for common LLM layers (e.g. RMSNorm, LayerNorm, SwiGLU, RoPE, Fused Linear CrossEntropy) plus specialized alignment & distillation losses (e.g. GRPO, DPO, ORPO, SimPO, KTO, JSD, TVD).


📚 [**Full Documentation and API Reference →**](https://linkedin.github.io/Liger-Kernel)  
💬 [**Join our Slack →**](https://join.slack.com/t/ligerkernel/shared_invite/zt-38amujbrb-MEeFtoAO0PP0fniWmTcm9Q)



## 📦 Installation

```bash
# Stable
pip install liger-kernel

# Nightly
pip install liger-kernel-nightly

# From source
git clone https://github.com/linkedin/Liger-Kernel.git
cd Liger-Kernel
pip install -e .
```

## Quickstart

There are a couple of ways to apply Liger kernels, depending on the level of customization required.

### 1. Use AutoLigerKernelForCausalLM

Using the `AutoLigerKernelForCausalLM` is the simplest approach, as you don't have to import a model-specific patching API. If the model type is supported, the modeling code will be automatically patched using the default settings.

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# This AutoModel wrapper class automatically monkey-patches the
# model with the optimized Liger kernels if the model is supported.
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/some/model")
```

### 2. Apply Model-Specific Patching APIs

Using the [patching APIs](#patching), you can swap Hugging Face models with optimized Liger Kernels.

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

> If you’re training with popular libraries like Transformers Trainer or TRL’s SFTTrainer, you may not need to patch manually — Liger Kernel already integrates with these frameworks.Check the [Integrations Guide]() for examples.

## Benchmarks

### Supervised Fine-Tuning (SFT)

| Speed Up                 | Memory Reduction        |
|--------------------------|-------------------------|
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

> **Note:**
> - Benchmark conditions: LLaMA 3-8B, Batch Size = 8, Data Type = `bf16`, Optimizer = AdamW, Gradient Checkpointing = True, Distributed Strategy = FSDP1 on 8 A100s.
> - Hugging Face models start to OOM at a 4K context length, whereas Hugging Face + Liger Kernel scales up to 16K.

### Alignment and Distillation

<p align="center">
    <img src="https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/post-training.png" width="50%" alt="Post Training">
</p>

We provide optimized post training kernels like DPO, ORPO, SimPO, and more which can reduce memory usage by up to 80%. You can easily use them as python modules.

```python
from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
orpo_loss = LigerFusedLinearORPOLoss()
y = orpo_loss(lm_head.weight, x, target)
```

## High-level APIs

### AutoModel

| **AutoModel Variant** | **API** |
|-----------|---------|
| AutoModelForCausalLM | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |


### Patching

| **Model**   | **API**                                                      | **Supported Operations**                                                |
|-------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| Gemma1      | `liger_kernel.transformers.apply_liger_kernel_to_gemma`    | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma2      | `liger_kernel.transformers.apply_liger_kernel_to_gemma2`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Text)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text`   | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Gemma3 (Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_gemma3`   | RoPE, RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| GLM-4   | `liger_kernel.transformers.apply_liger_kernel_to_glm4`     | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| GLM-4v   | `liger_kernel.transformers.apply_liger_kernel_to_glm4v`     | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| Granite 3.0 & 3.1   | `liger_kernel.transformers.apply_liger_kernel_to_granite`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| LLaMA 2 & 3 | `liger_kernel.transformers.apply_liger_kernel_to_llama`   | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaMA 4 (Text & Multimodal)      | `liger_kernel.transformers.apply_liger_kernel_to_llama4`   | RoPE, RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| LLaMA 3.2-Vision | `liger_kernel.transformers.apply_liger_kernel_to_mllama`   | RoPE, RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| LLaVA | `liger_kernel.transformers.apply_liger_kernel_to_llava`   | CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mistral     | `liger_kernel.transformers.apply_liger_kernel_to_mistral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Mixtral     | `liger_kernel.transformers.apply_liger_kernel_to_mixtral`  | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| OLMO2   | `liger_kernel.transformers.apply_liger_kernel_to_olmo2`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| PaliGemma      | `liger_kernel.transformers.apply_liger_kernel_to_paligemma`   | RoPE, RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Phi3 & Phi3.5       | `liger_kernel.transformers.apply_liger_kernel_to_phi3`     | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy         |
| Qwen2, Qwen2.5, & QwQ      | `liger_kernel.transformers.apply_liger_kernel_to_qwen2`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2-VL       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl`    | RoPE, RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen2.5-VL       | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl`    | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy        |
| Qwen3   | `liger_kernel.transformers.apply_liger_kernel_to_qwen3`    |  RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| Qwen3 MoE | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |
| SmolLM3 | `liger_kernel.transformers.apply_liger_kernel_to_smollm3` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy       |


## Low-level APIs

- `Fused Linear` kernels combine linear layers with losses, reducing memory usage by up to 80% - ideal for HBM-constrained workloads.
- Other kernels use fusion and in-place techniques for memory and performance optimization.

### Model Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| RMSNorm                         | `liger_kernel.transformers.LigerRMSNorm`                    |
| LayerNorm                       | `liger_kernel.transformers.LigerLayerNorm`                  |
| RoPE                            | `liger_kernel.transformers.liger_rotary_pos_emb`            |
| SwiGLU                          | `liger_kernel.transformers.LigerSwiGLUMLP`                  |
| GeGLU                           | `liger_kernel.transformers.LigerGEGLUMLP`                   |
| CrossEntropy                    | `liger_kernel.transformers.LigerCrossEntropyLoss`           |
| Fused Linear CrossEntropy       | `liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss`|
| Multi Token Attention           | `liger_kernel.transformers.LigerMultiTokenAttention`        |
| Softmax                         | `liger_kernel.transformers.LigerSoftmax`                    |
| Sparsemax                       | `liger_kernel.transformers.LigerSparsemax`                  |


### Alignment Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Fused Linear CPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearCPOLoss`       |
| Fused Linear DPO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearDPOLoss`       |
| Fused Linear ORPO Loss          | `liger_kernel.chunked_loss.LigerFusedLinearORPOLoss`      |
| Fused Linear SimPO Loss         | `liger_kernel.chunked_loss.LigerFusedLinearSimPOLoss`     |
| Fused Linear KTO Loss           | `liger_kernel.chunked_loss.LigerFusedLinearKTOLoss`     |

### Distillation Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| KLDivergence                    | `liger_kernel.transformers.LigerKLDIVLoss`                  |
| JSD                             | `liger_kernel.transformers.LigerJSD`                        |
| Fused Linear JSD                  | `liger_kernel.transformers.LigerFusedLinearJSD`             |
| TVD                             | `liger_kernel.transformers.LigerTVDLoss`                    |

### Experimental Kernels

| **Kernel**                      | **API**                                                     |
|---------------------------------|-------------------------------------------------------------|
| Embedding                       | `liger_kernel.transformers.experimental.LigerEmbedding`     |
| Matmul int2xint8                | `liger_kernel.transformers.experimental.matmul` |


## Contributing, Acknowledgements, and License

- [Contributing Guidelines](https://linkedin.github.io/Liger-Kernel/contributing/)
- [Acknowledgements](https://linkedin.github.io/Liger-Kernel/acknowledgement/)
- [License Information](https://linkedin.github.io/Liger-Kernel/license/)

## Sponsorship and Collaboration

- [Glows.ai](https://platform.glows.ai/): Sponsoring NVIDIA GPUs for our open source developers.
- [AMD](https://www.amd.com/en.html): Providing AMD GPUs for our AMD CI.
- [Intel](https://www.intel.com/): Providing Intel GPUs for our Intel CI.
- [Modal](https://modal.com/): Free 3000 credits from GPU MODE IRL for our NVIDIA CI.
- [EmbeddedLLM](https://embeddedllm.com/): Making Liger Kernel run fast and stable on AMD.
- [HuggingFace](https://huggingface.co/): Integrating Liger Kernel into Hugging Face Transformers and TRL.
- [Lightning AI](https://lightning.ai/): Integrating Liger Kernel into Lightning Thunder.
- [Axolotl](https://axolotl.ai/): Integrating Liger Kernel into Axolotl.
- [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory): Integrating Liger Kernel into Llama-Factory.

## Contact

- For issues, create a Github ticket in this repository
- For open discussion, join [our Slack channel](https://join.slack.com/t/ligerkernel/shared_invite/zt-38amujbrb-MEeFtoAO0PP0fniWmTcm9Q)
- For formal collaboration, send an email to Yanning Chen(yannchen@linkedin.com) and Zhipeng Wang(zhipwang@linkedin.com)

## Citation
If you use Liger Kernel in your research, please cite:

```bib
@inproceedings{
hsu2025ligerkernel,
title={Liger-Kernel: Efficient Triton Kernels for {LLM} Training},
author={Pin-Lun Hsu and Yun Dai and Vignesh Kothapalli and Qingquan Song and Shao Tang and Siyu Zhu and Steven Shimizu and Shivam Sahni and Haowen Ning and Yanning Chen and Zhipeng Wang},
booktitle={Championing Open-source DEvelopment in ML Workshop @ ICML25},
year={2025},
url={https://openreview.net/forum?id=36SjAIT42G}
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=linkedin/Liger-Kernel&type=Date)](https://www.star-history.com/#linkedin/Liger-Kernel&Date)

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
