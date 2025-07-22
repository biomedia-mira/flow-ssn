import math
import torch
import torch.nn as nn


class ChannelNorm(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        channelwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.channelwise_affine = channelwise_affine

        if self.channelwise_affine:
            self.weight = nn.Parameter(torch.empty(1, num_channels, 1, 1))
            if bias:
                self.bias = nn.Parameter(torch.empty(1, num_channels, 1, 1))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4  # (b, c, h, w)
        # (b, 1, h, w)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        # (b, c, h, w)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.channelwise_affine:
            x = self.weight * x + self.bias
        return x

    def extra_repr(self) -> str:
        return (
            "{num_channels}, eps={eps}, "
            "channelwise_affine={channelwise_affine}".format(**self.__dict__)
        )


def zero_module(module: nn.Module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    t = t * 1000  # [0, 1] -> [0, 1000]
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding  # (b, dim)
