from typing import Optional, Any, Literal, Callable

import torch
import torch.nn as nn
from torchdiffeq import odeint


def euler_solver(func: Callable, y0: torch.Tensor, t: torch.Tensor, **kwargs):
    y = torch.zeros(len(t), *y0.shape, device=y0.device)
    y[0] = y0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        y[i] = y[i - 1] + h * func(t[i - 1], y[i - 1])
    return y


def ode_solve(
    model: nn.Module,
    u: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    field: Literal["unconstrained", "categorical"] = "categorical",
    **kwargs: Any,
):
    if field == "unconstrained":
        fn = lambda t, x: model(x, t.repeat(x.shape[0]), context)
    elif field == "categorical":
        fn = lambda t, x: model(x, t.repeat(x.shape[0]), context).softmax(dim=1) - u
    else:
        raise NotImplementedError
    return odeint(fn, u, **kwargs)[-1]
