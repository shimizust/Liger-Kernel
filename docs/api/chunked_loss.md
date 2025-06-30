# Chunked Loss Modules

This section documents the chunked loss modules in `liger_kernel.chunked_loss`. These modules provide memory-efficient implementations of various loss functions used in preference learning and alignment.


| Category | Base Class | Implementations |
|----------|------------|-----------------|
| **Paired Preference Learning** | [LigerFusedLinearPreferenceBase](#liger_kernel.chunked_loss.fused_linear_preference.LigerFusedLinearPreferenceBase) | [DPO](#liger_kernel.chunked_loss.dpo_loss) • [CPO](#liger_kernel.chunked_loss.cpo_loss) • [ORPO](#liger_kernel.chunked_loss.orpo_loss) • [SimPO](#liger_kernel.chunked_loss.simpo_loss) |
| **Unpaired Preference Learning** | [LigerFusedLinearUnpairedPreferenceBase](#liger_kernel.chunked_loss.fused_linear_unpaired_preference.LigerFusedLinearUnpairedPreferenceBase) | [KTO](#liger_kernel.chunked_loss.kto_loss) |
| **PPO Learning** | [LigerFusedLinearPPOBase](#liger_kernel.chunked_loss.fused_linear_ppo.LigerFusedLinearPPOBase) | [GRPO](#liger_kernel.chunked_loss.grpo_loss) |
| **Knowledge Distillation** | [LigerFusedLinearDistillationBase](#liger_kernel.chunked_loss.fused_linear_distillation.LigerFusedLinearDistillationBase) | [JSD](#liger_kernel.chunked_loss.jsd_loss) |


## Design Pattern

Each loss module in `liger_kernel.chunked_loss` follows the same general structure:

- **Base Class**: Every loss starts from a base class that is a custom `torch.autograd.Function`. This base class fuses the final linear projection layer (the LM head) with the chunked loss computation. The base class defines:
    - A `forward` method that runs the fused projection and loops over chunks
    - An abstract `*_loss_fn` method (e.g. `preference_loss_fn`, `ppo_loss_fn`, `distillation_loss_fn`) that computes the loss for a chunk. Child classes will override to implement a specific objective.
- **Concrete Loss Function**: Each specific loss (e.g. DPO, CPO, ORPO) extends the base class and implements the corresponding `*_loss_fn` method using normal PyTorch operations.
- **Convenience Module**: Each fused loss function has a corresponding `torch.nn.Module` wrapper class.

This design pattern makes it easy to implement new loss functions without having to write custom Triton kernels.

## Paired Preference Learning

These modules implement preference learning algorithms that work with paired data (chosen vs rejected examples). 

::: liger_kernel.chunked_loss.fused_linear_preference.LigerFusedLinearPreferenceBase
    options:
        heading_level: 3
        
::: liger_kernel.chunked_loss.dpo_loss
    options:
        heading_level: 3

::: liger_kernel.chunked_loss.cpo_loss
    options:
        heading_level: 3

::: liger_kernel.chunked_loss.orpo_loss
    options:
        heading_level: 3

::: liger_kernel.chunked_loss.simpo_loss
    options:
        heading_level: 3

## Unpaired Preference Learning

These modules implement preference learning algorithms that work with unpaired data (examples labeled as preferred vs not preferred).

::: liger_kernel.chunked_loss.fused_linear_unpaired_preference.LigerFusedLinearUnpairedPreferenceBase
    options:
        heading_level: 3

::: liger_kernel.chunked_loss.kto_loss
    options:
        heading_level: 3

## PPO Learning

These modules implement Proximal Policy Optimization variants for reinforcement learning from human feedback.

::: liger_kernel.chunked_loss.fused_linear_ppo.LigerFusedLinearPPOBase
    options:
        heading_level: 3

::: liger_kernel.chunked_loss.grpo_loss
    options:
        heading_level: 3

## Knowledge Distillation

These modules implement knowledge distillation techniques for transferring knowledge from a teacher model to a student model.

::: liger_kernel.chunked_loss.fused_linear_distillation.LigerFusedLinearDistillationBase
    options:
        heading_level: 3

::: liger_kernel.chunked_loss.jsd_loss
    options:
        heading_level: 3
