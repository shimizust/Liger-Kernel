import torch.nn as nn

from liger_kernel.ops.tvd import LigerTVDLossFunction


class LigerTVDLoss(nn.Module):
    """
    Liger Kernel implementation of Total Variation Distance (TVD) Loss.
    
    This is an optimized implementation of TVD loss using Triton kernels for
    memory-efficient computation. TVD measures the maximum difference between
    two probability distributions.
    
    The TVD between distributions P and Q is computed as:
    TVD(P || Q) = 0.5 * sum(|P - Q|)
    
    Args:
        reduction (str, optional): Reduction method for the loss.
            Options: "none", "sum", "mean", "batchmean". Defaults to "batchmean".
        ignore_index (int, optional): Index to ignore in loss computation.
            Defaults to -100.
            
    Examples:
        >>> # Basic usage
        >>> tvd_loss = LigerTVDLoss()
        >>> p = torch.softmax(torch.randn(32, 10000), dim=-1)
        >>> q = torch.softmax(torch.randn(32, 10000), dim=-1)
        >>> loss = tvd_loss(p, q)
        
        >>> # With labels and ignore_index
        >>> tvd_loss = LigerTVDLoss(reduction="mean", ignore_index=-100)
        >>> labels = torch.randint(0, 10000, (32,))
        >>> loss = tvd_loss(p, q, labels)
    """
    
    def __init__(self, reduction="batchmean", ignore_index: int = -100):
        super(LigerTVDLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, p, q, shift_labels=None):
        """
        Compute Total Variation Distance loss.
        
        Args:
            p (torch.Tensor): First distribution of shape (batch_size, vocab_size).
            q (torch.Tensor): Second distribution of shape (batch_size, vocab_size).
            shift_labels (torch.Tensor, optional): Labels of shape (batch_size,).
                Used for masking with ignore_index.
                
        Returns:
            torch.Tensor: Computed TVD loss.
        """
        return LigerTVDLossFunction.apply(
            p, q, shift_labels, self.reduction, self.ignore_index
        )
