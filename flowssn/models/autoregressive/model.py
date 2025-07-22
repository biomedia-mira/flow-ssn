from typing import Dict, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

from .distributions import InverseAutoregressiveFlow
from .transforms import InverseAutoregressive, ConditionalInverseAutoregressive


class AutoregressiveFlowSSN(nn.Module):
    def __init__(
        self,
        flow_type: Literal["default", "gated", "volume_preserving"],
        flow_nets: Dict[str, nn.ModuleList],
        base_nets: Optional[nn.ModuleDict] = None,
        cond_flow: bool = False,
        cond_base: bool = False,
        base_std: float = 1.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.cond_flow = cond_flow
        self.cond_base = cond_base
        self.base_nets = base_nets
        self.base_std = base_std  # learns diag cov of base dist if 0, else fixed

        if cond_base and base_nets is not None:
            assert base_nets["iaf"].out_channels == num_classes * 2  # pred (loc, scale)
        else:
            assert flow_nets["iaf"][0].input_shape[0] == num_classes
            base_shape = flow_nets["iaf"][0].input_shape  # (num_classes, h, w)
            self.register_buffer("base_loc", torch.zeros(*base_shape))
            self.register_buffer("base_scale", torch.ones(*base_shape))

        assert flow_nets["iaf"][0].out_channels == num_classes * 2
        transform = {"iaf": InverseAutoregressive}
        if self.cond_flow:
            transform["iaf"] = ConditionalInverseAutoregressive

        self.flows = nn.ModuleDict()
        self.transf = {}
        for k, nets in flow_nets.items():
            self.flows[k] = nn.ModuleList([AutoregressiveModel(net) for net in nets])
            self.transf[k] = [
                transform[k](model, flow_type, event_dim=3)
                for model in self.flows[k]  # type: ignore
            ]

    def forward(self, batch: Dict[str, torch.Tensor], mc_samples: int = 1):
        batch_size, _, h, w = batch["x"].shape

        base = self.get_base_dist("iaf", batch["x"])
        iaf = InverseAutoregressiveFlow(base, self.transf["iaf"])
        context = None
        if self.cond_flow:
            context = self.maybe_expand(batch["x"], mc_samples)
            iaf = iaf.condition(context)

        sample_shape = torch.Size(
            [mc_samples] if self.cond_base else [mc_samples, batch_size]
        )
        # (m, b, k, h, w)
        logits = iaf.rsample(sample_shape)

        loss, std = None, 0
        if "y" in batch.keys():  # training
            # (m, b, h, w, k)
            logits = logits.permute(0, 1, 3, 4, 2)
            # (m, b, h, w), broadcasting y to m
            log_py_n = dist.OneHotCategorical(logits=logits).log_prob(batch["y"])
            # (m, b)
            log_prob = log_py_n.sum(dim=(-2, -1))

            if mc_samples > 1:
                std = log_prob.exp().std(dim=0).mean()
                log_prob = torch.logsumexp(log_prob, dim=0) - np.log(mc_samples)

            loss = -torch.mean(log_prob) / (h * w)
        else:
            # (m, b, h, w, k)
            logits = logits.permute(0, 1, 3, 4, 2)
        return dict(loss=loss, logits=logits, std=std)

    def get_base_dist(self, flow: Literal["iaf"], x: Optional[torch.Tensor] = None):
        if not self.cond_base:
            loc, scale = self.base_loc, self.base_scale
        else:
            assert self.base_nets is not None
            # (b, k, h, w), (b, k, h, w)
            loc, log_scale = self.base_nets[flow](x).chunk(2, dim=1)
            if self.base_std:
                scale = self.base_std
            else:
                scale = torch.exp(log_scale.clamp(min=np.log(1e-5)))
        return dist.Normal(loc, scale)

    def maybe_expand(self, x: torch.Tensor, mc_samples: int = 1):
        # (m*b, ...) <- (b, ...)
        return (
            x[None, ...].expand(mc_samples, *x.shape).reshape(-1, *x.shape[1:])
            if mc_samples > 1
            else x
        )


class AutoregressiveModel(nn.Module):
    def __init__(self, autoregressive_nn: nn.Module):
        super().__init__()
        self.autoregressive_nn = autoregressive_nn

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        mc_samples = 1
        if x.ndim > 4:
            mc_samples = x.shape[0]
            # (m*b, 2k, h, w) <- (m, b, 2k, h, w)
            x = x.reshape(-1, *x.shape[2:])
            if context is not None:
                assert x.shape[0] == context.shape[0]
        # (m*b, 2*k, h, w)
        out = self.autoregressive_nn(x, y=context)
        # (m, b, 2*k, h, w)
        out = out.reshape(mc_samples, -1, *out.shape[1:])
        loc, log_scale = out.chunk(2, dim=2)
        return loc, log_scale
