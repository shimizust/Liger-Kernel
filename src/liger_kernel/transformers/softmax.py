import torch
import torch.nn as nn

from liger_kernel.ops.softmax import LigerSoftmaxFunction


class LigerSoftmax(nn.Module):
    """Liger implementation of Softmax activation function.
    
    The Softmax kernel implementation provides an optimized implementation of the softmax operation, 
    which is a fundamental component in neural networks for converting raw scores into probability 
    distributions.
    
    The implementation shows notable speedups compared to the Softmax PyTorch implementation and can 
    be used as a drop-in replacement for torch.nn.Softmax in attention mechanisms, classification 
    tasks, and other neural network components.
    
    Example:
        ```python
        import torch
        from liger_kernel.transformers.softmax import LigerSoftmax
        
        # Create softmax layer (applies softmax along last dimension)
        liger_softmax = LigerSoftmax()
        
        # Apply to input tensor
        x = torch.randn(2, 4, 10)
        output = liger_softmax(x)  # Probability distribution along last dim
        ```
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply softmax to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
            
        Returns:
            torch.Tensor: Output tensor with same shape as input, containing probability distributions.
        """
        return LigerSoftmaxFunction.apply(x)
