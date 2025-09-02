import math

import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair

from liger_kernel.ops.multi_token_attention import LigerMultiTokenAttentionFunction


class LigerMultiTokenAttention(nn.Module):
    """Liger implementation of Multi Token Attention mechanism.
    
    Multi Token Attention is a new attention mechanism introduced by Meta Research that can operate 
    on multiple Q and K inputs. This kernel implementation provides an optimized fused implementation 
    of multi-token attention over the standard PyTorch model baseline.
    
    The operation can be mathematically described as:
        out = mask_{0}(conv2d(softmax(mask_{-∞}(scores))))
    
    This implementation achieves significant speedups compared to the PyTorch baseline through 
    fused operations that avoid intermediate materializations and provide better memory efficiency.
    
    Reference:
        Meta Research. "Multi Token Attention."
        https://arxiv.org/abs/2504.00927
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1.
        padding (int): Padding added to all four sides of the input. Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
        sparse (bool): If True, uses sparse operations for efficiency. Default: False.
        
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        sparse: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.sparse = sparse

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return LigerMultiTokenAttentionFunction.apply(
            scores,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.sparse,
        )
