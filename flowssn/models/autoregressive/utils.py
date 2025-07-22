import torch
import numpy as np

from torch.distributions import constraints
from torch.distributions.transforms import Transform


def safe_log(x: torch.Tensor):
    return torch.log(x.clamp(min=1e-7))


def safe_exp(x: torch.Tensor, min_log: float = np.log(1e-5), max_log: float = 3.0):
    return torch.where(
        x >= max_log,
        x - max_log + np.exp(max_log),
        torch.exp(x.clamp(min=min_log)),
    )


class SoftmaxCentered(Transform):
    """
    Implements softmax as a bijection, the forward transformation appends a value to the
    input and the inverse removes it. The appended coordinate represents a pivot, e.g.,
    softmax(x) = exp(x-c) / sum(exp(x-c)) where c is the implicit last coordinate.

    Adapted from a Tensorflow implementation: https://tinyurl.com/48vuh7yw
    """

    bijective = True
    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def _call(self, x: torch.Tensor):
        zero_pad = torch.zeros(*x.shape[:-1], 1, device=x.device)
        x_padded = torch.cat([x, zero_pad], dim=-1)
        return (x_padded / self.temperature).softmax(dim=-1)

    def _inverse(self, y: torch.Tensor):
        log_y = safe_log(y)
        unorm_log_probs = log_y[..., :-1] - log_y[..., -1:]
        return unorm_log_probs * self.temperature

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor):
        """log|det(dy/dx)|"""
        Kplus1 = torch.tensor(y.size(-1), dtype=y.dtype, device=y.device)
        return 0.5 * Kplus1.log() + torch.sum(safe_log(y), dim=-1)

    def log_abs_det_jacobian_alternative(self, x: torch.Tensor, y: torch.Tensor):
        """-log|det(dx/dy)|. Rename to log_abs_det_jacobian for active use."""
        Kplus1 = torch.tensor(x.size(-1) + 1, dtype=x.dtype, device=x.device)
        return (
            0.5 * Kplus1.log()
            + torch.sum(x, dim=-1)
            - Kplus1 * torch.nn.functional.softplus(torch.logsumexp(x, dim=-1))
        )

    def forward_shape(self, shape: torch.Size):
        return shape[:-1] + (shape[-1] + 1,)  # forward appends one dim

    def inverse_shape(self, shape: torch.Size):
        if shape[-1] <= 1:
            raise ValueError
        return shape[:-1] + (shape[-1] - 1,)  # inverse removes one dim
