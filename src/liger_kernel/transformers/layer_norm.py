import torch
import torch.nn as nn

from liger_kernel.ops.layer_norm import LigerLayerNormFunction


class LigerLayerNorm(nn.Module):
    """
    Liger Kernel implementation of Layer Normalization.
    
    This is an optimized drop-in replacement for `torch.nn.LayerNorm` that provides
    memory-efficient layer normalization using Triton kernels.
    
    Layer normalization normalizes the input across the features dimension, computing
    the mean and variance for each sample in the batch independently.
    
    Args:
        hidden_size (int): The number of features in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical 
            stability. Defaults to 1e-6.
        bias (bool, optional): If True, adds a learnable bias parameter 
            (zero-initialized). Defaults to False.
        init_fn (str, optional): Initialization function for the weight parameter.
            Must be either "ones" or "zeros". Defaults to "ones".
        
    Examples:
        >>> # Basic usage
        >>> layer_norm = LigerLayerNorm(512)
        >>> input_tensor = torch.randn(32, 128, 512)
        >>> output = layer_norm(input_tensor)
        
        >>> # Custom epsilon and zero initialization
        >>> layer_norm_custom = LigerLayerNorm(512, eps=1e-5, init_fn="zeros")
        >>> output = layer_norm_custom(input_tensor)
    """
    
    def __init__(self, hidden_size, eps=1e-6, bias=False, init_fn="ones"):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size) if bias else torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Apply layer normalization to the input tensor.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (N, *, hidden_size).
            
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
            
        Note:
            The normalization is computed as:
            output = (input - mean) / sqrt(variance + eps) * weight + bias
            where mean and variance are computed across the last dimension.
        """
        return LigerLayerNormFunction.apply(hidden_states, self.weight, self.bias, self.variance_epsilon)

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"
