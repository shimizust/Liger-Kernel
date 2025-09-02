import torch
import torch.nn as nn

from liger_kernel.ops.dyt import LigerDyTFunction


class LigerDyT(nn.Module):
    """Liger implementation of Dynamic Tanh (DyT) normalization replacement.
    
    Dynamic Tanh (DyT) is an element-wise operation DyT(x) = tanh(αx) that serves as a drop-in 
    replacement for normalization layers (LayerNorm and RMSNorm) in Transformers. DyT is inspired by
    the observation that layer normalization in Transformers often produces tanh-like, S-shaped
    input-output mappings.
    
    This technique demonstrates that Transformers without traditional normalization can achieve 
    the same or better performance using this remarkably simple approach. By incorporating DyT, 
    Transformers without normalization can match or exceed the performance of their normalized 
    counterparts, mostly without hyperparameter tuning.
    
    Reference:
        Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu. 
        "Transformers without Normalization." CVPR 2025.
        https://arxiv.org/abs/2503.10622
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        beta (bool): Whether to include a learnable bias parameter. Default: True.
        init_alpha (float): Initial value for the alpha scaling parameter. Default: 0.5.
    """
    def __init__(self, hidden_size, beta=True, init_alpha=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = None
        if beta:
            self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LigerDyTFunction.apply(x, self.alpha, self.gamma, self.beta)

    def extra_repr(self):
        return f"{self.hidden_size}, init_alpha={self.init_alpha}, beta={self.beta}"
