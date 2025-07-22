import numpy as np
import torch
import torch.nn as nn

from torch.distributions import constraints
from torch.distributions.transforms import Transform

from flowssn.utils import LambdaModule


class InverseAutoregressive(Transform):

    bijective = True
    sign = +1  # type: ignore

    def __init__(
        self,
        autoregressive_nn: nn.Module,
        flow_type: str = "default",
        event_dim: int = 0,
    ):
        super().__init__(cache_size=1)
        assert flow_type in ["default", "gated", "volume_preserving"]
        self.autoregressive_nn = autoregressive_nn
        self.flow_type = flow_type
        self._event_dim = event_dim
        self._cached_log_scale = None

    @property
    def event_dim(self):
        return self._event_dim

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, self.event_dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, self.event_dim)

    def _call(self, x: torch.Tensor):
        loc, lg_scale = self.autoregressive_nn(x)

        if self.flow_type == "default":
            log_scale = lg_scale.clamp(min=np.log(1e-5))
            scale = torch.exp(log_scale)
        elif self.flow_type == "gated":
            scale = torch.sigmoid(lg_scale + 1) + 1e-5
            log_scale = torch.log(scale)
            loc = (1 - scale) * loc
        elif self.flow_type == "volume_preserving":
            scale = 1.0
            log_scale = torch.zeros_like(loc)
        else:
            raise NotImplementedError

        self._cached_log_scale = log_scale
        y = loc + scale * x
        return y

    def _inverse(self, y: torch.Tensor):
        raise NotImplementedError("Too expensive for high-dimensional variables.")

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor):
        x_cached, y_cached = self._cached_x_y

        if x is not x_cached or y is not y_cached:
            self(x)  # update cache via self._call

        if self._cached_log_scale is not None:
            log_scale = self._cached_log_scale
        else:
            loc, lg_scale = self.autoregressive_nn(x)

            if self.flow_type == "default":
                log_scale = lg_scale.clamp(min=np.log(1e-5))
            elif self.flow_type == "gated":
                scale = torch.sigmoid(lg_scale + 1) + 1e-5
                log_scale = torch.log(scale)
            elif self.flow_type == "volume_preserving":
                log_scale = torch.zeros_like(loc)
            else:
                raise NotImplementedError

        shape = x.shape
        if self.event_dim:
            flat_size = log_scale.size()[: -self.event_dim] + (-1,)
            log_scale = log_scale.reshape(flat_size).sum(dim=-1)
            shape = x.shape[: -self.event_dim]
        return log_scale.expand(shape)


class ConditionalInverseAutoregressive(InverseAutoregressive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def condition(self, h: torch.Tensor):
        cond_nn = LambdaModule(lambda x: self.autoregressive_nn(x, context=h))
        return InverseAutoregressive(cond_nn, self.flow_type, self.event_dim)


class MaskedAutoregressive(InverseAutoregressive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._call, self._inverse = self._inverse, self._call  # type: ignore


class ConditionalMaskedAutoregressive(MaskedAutoregressive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def condition(self, h: torch.Tensor):
        cond_nn = LambdaModule(lambda x: self.autoregressive_nn(x, context=h))
        return MaskedAutoregressive(cond_nn, self.flow_type, self.event_dim)
