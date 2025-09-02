import torch
import torch.nn as nn

from liger_kernel.ops.sparsemax import LigerSparsemaxFunction


class LigerSparsemax(nn.Module):
    """Liger implementation of Sparsemax activation function.
    
    Sparsemax is a sparse alternative to softmax that produces sparse probability distributions. 
    This kernel implements an efficient version of the sparsemax operation that can be used as a 
    drop-in replacement for softmax in attention mechanisms or classification tasks.
    
    The implementation achieves significant speed improvements and memory savings compared to 
    standard PyTorch implementations, particularly for large input tensors.
    
    Args:
        dim (int): The dimension along which to apply sparsemax. Default: -1.
        
    Example:
        ```python
        import torch
        from liger_kernel.transformers.sparsemax import LigerSparsemax
        
        # Create sparsemax layer
        liger_sparsemax = LigerSparsemax(dim=-1)
        
        # Apply to input tensor
        x = torch.randn(2, 4, 10)
        output = liger_sparsemax(x)  # Sparse probability distribution
        ```
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LigerSparsemaxFunction.apply(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
