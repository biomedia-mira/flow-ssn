from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist

from .solvers import ode_solve


class ContinuousFlowSSN(nn.Module):
    def __init__(
        self,
        flow_net: nn.Module,
        base_net: Optional[nn.Module] = None,
        num_classes: int = 2,
        cond_base: bool = False,
        cond_flow: bool = False,
        base_std: float = 1.0,
    ):
        super().__init__()
        assert flow_net.out_channels == num_classes
        self.flow_net = flow_net
        self.base_net = base_net
        self.num_classes = num_classes
        self.cond_base = cond_base
        self.cond_flow = cond_flow
        self.base_std = base_std  # learns diag cov of base dist if 0, else fixed

        if self.cond_base:
            assert base_net is not None
            assert base_net.out_channels == num_classes * 2  # pred (loc, scale)
        else:
            assert flow_net.input_shape[0] == num_classes
            base_shape = flow_net.input_shape  # (num_classes, h, w)
            self.register_buffer("base_loc", torch.zeros(*base_shape))
            self.register_buffer("base_scale", torch.ones(*base_shape))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        mc_samples: int = 1,
        ode_kwargs: Optional[Dict[str, Any]] = None,
    ):
        batch_size, _, h, w = batch["x"].shape
        base_dist = self.get_base_dist(batch["x"])
        sample_shape = torch.Size(
            [mc_samples] if self.cond_base else [mc_samples, batch_size]
        )
        # (m*b, k, h, w)
        u = base_dist.rsample(sample_shape).reshape(mc_samples * batch_size, -1, h, w)
        # (m*b, c, h, w)
        context = self.maybe_expand(batch["x"], mc_samples) if self.cond_flow else None

        loss, logits, probs, std = *(None,) * 3, 0.0
        if "y" in batch.keys():  # training
            # (b, k, h, w)
            y = batch["y"].float().permute(0, 3, 1, 2)  # * 2 - 1  # [-1, 1]
            # (m*b, k, h, w)
            y = self.maybe_expand(y, mc_samples)
            # (m*b)
            t = self.maybe_expand(
                torch.zeros(batch_size, device=y.device).uniform_(0, 1), mc_samples
            )
            # (m*b, k, h, w)
            y_t = self.interpolant(u, y, t)
            loss, std = self.logit_pred_loss(batch["y"], y_t, t, context, mc_samples)
        else:  # test
            if ode_kwargs is None:
                ode_kwargs = {
                    "method": "euler",
                    "t": torch.tensor([0.0, 1.0]).to(u.device),
                    "options": dict(step_size=1.0 / self.eval_T),
                }
            # (m*b, k, h, w)
            y_hat = ode_solve(
                self.flow_net, u, context, field="categorical", **ode_kwargs
            )
            # shifts to min of 0
            probs = y_hat - y_hat.min(dim=1, keepdim=True).values  # type: ignore
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-7)  # renorm
            # (m, b, h, w, k)
            probs = probs.permute(0, 2, 3, 1).reshape(mc_samples, batch_size, h, w, -1)

        if logits is not None:
            # (m, b, h, w, k)
            logits = logits.permute(0, 2, 3, 1).reshape(
                mc_samples, batch_size, h, w, -1
            )
        return dict(loss=loss, logits=logits, probs=probs, std=std)

    def get_base_dist(self, x: Optional[torch.Tensor] = None):
        if not self.cond_base:
            loc, scale = self.base_loc, self.base_scale
        else:
            assert self.base_net is not None
            # (b, k, h, w), (b, k, h, w)
            loc, log_scale = self.base_net(x).chunk(2, dim=1)
            if self.base_std:
                scale = self.base_std  # will broadcast
            else:
                scale = torch.exp(log_scale.clamp(min=np.log(1e-5)))
        return dist.Normal(loc, scale)

    def interpolant(self, u: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
        # (m*b, k, h, w)
        t = t.reshape(-1, *([1] * (u.ndim - 1)))  # for spatial broadcasting
        y_t = (1 - t) * u + t * y
        return y_t

    def logit_pred_loss(
        self,
        y: torch.Tensor,
        y_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mc_samples: int = 1,
    ):
        std = 0
        # (m*b, k, h, w)
        logits = self.flow_net(y_t, t, context)
        # (m, b, h, w, k)
        logits = logits.reshape(mc_samples, -1, *logits.shape[1:]).permute(
            0, 1, 3, 4, 2
        )
        # (m, b, h, w), broadcasting y to m
        log_py_n = dist.OneHotCategorical(logits=logits).log_prob(y)
        # (m, b)
        log_prob = log_py_n.mean(dim=(-2, -1))

        if mc_samples > 1:
            std = log_prob.exp().std(dim=0).mean()
            log_prob = torch.logsumexp(log_prob, dim=0) - np.log(mc_samples)

        loss = -torch.mean(log_prob)  # / (h * w)
        return loss, std

    def maybe_expand(self, x: torch.Tensor, mc_samples: int = 1):
        # (m*b, ...) <- (b, ...)
        return (
            x[None, ...].expand(mc_samples, *x.shape).reshape(-1, *x.shape[1:])
            if mc_samples > 1
            else x
        )
