import torch
import torch.nn as nn

from liger_kernel.ops.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    """Liger implementation of Root Mean Square Layer Normalization.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): Offset value applied during normalization. Default: 0.0.
        casting_mode (str): Precision casting mode ("llama", "gemma", etc.). Default: "llama".
        init_fn (str): Weight initialization function ("ones" or "zeros"). Default: "ones".
        in_place (bool): Whether to perform operations in-place for memory efficiency. Default: True.
        row_mode (str, optional): Row processing mode. Default: None.
    """
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=True,
        row_mode=None,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place, self.row_mode = (
            eps,
            offset,
            casting_mode,
            in_place,
            row_mode,
        )

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
            self.row_mode,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}, row_mode={self.row_mode}"


class LigerRMSNormForGemma(LigerRMSNorm):
    """RMSNorm implementation optimized for Gemma models.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): Offset value for Gemma models. Default: 1.0.
        casting_mode (str): Precision casting mode. Default: "gemma".
        init_fn (str): Weight initialization function. Default: "zeros".
        in_place (bool): Whether to perform operations in-place. Default: True.
        row_mode (str, optional): Row processing mode. Default: None.
    """
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=True, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGemma2(LigerRMSNorm):
    """RMSNorm implementation optimized for Gemma2 models.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): Offset value for Gemma models. Default: 1.0.
        casting_mode (str): Precision casting mode. Default: "gemma".
        init_fn (str): Weight initialization function. Default: "zeros".
        in_place (bool): Whether to perform operations in-place. Default: False.
        row_mode (str, optional): Row processing mode. Default: None.
    """
    def __init__(
        self, hidden_size, eps=1e-6, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGemma3(LigerRMSNorm):
    """RMSNorm implementation optimized for Gemma3 models.
    
    Args:
        dim (int): The dimension size (replaces hidden_size for Gemma3 compatibility).
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): Offset value for Gemma models. Default: 1.0.
        casting_mode (str): Precision casting mode. Default: "gemma".
        init_fn (str): Weight initialization function. Default: "zeros".
        in_place (bool): Whether to perform operations in-place. Default: False.
    """
    def __init__(self, dim, eps=0.000001, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)


class LigerRMSNormForOlmo2(LigerRMSNorm):
    """RMSNorm implementation optimized for OLMo2 models.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): Offset value for OLMo2 models. Default: 0.0.
        casting_mode (str): Precision casting mode. Default: "llama".
        init_fn (str): Weight initialization function. Default: "ones".
        in_place (bool): Whether to perform operations in-place. Default: False.
        row_mode (str, optional): Row processing mode. Default: None.
    """
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)


class LigerRMSNormForGlm4(LigerRMSNorm):
    """RMSNorm implementation optimized for GLM-4 models.
    
    Args:
        hidden_size (int): The size of the hidden dimension.
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-6.
        offset (float): Offset value for GLM-4 models. Default: 0.0.
        casting_mode (str): Precision casting mode. Default: "llama".
        init_fn (str): Weight initialization function. Default: "ones".
        in_place (bool): Whether to perform operations in-place. Default: False.
        row_mode (str, optional): Row processing mode. Default: None.
    """
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones", in_place=False, row_mode=None
    ):
        super().__init__(hidden_size, eps, offset, casting_mode, init_fn, in_place, row_mode)
