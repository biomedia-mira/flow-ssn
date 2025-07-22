from typing import Tuple, Union, List, Optional

import torch
import torch.nn as nn

from .utils import ChannelNorm, zero_module
from flowssn.utils import LambdaModule

activation = nn.SiLU()
normalization = lambda x: ChannelNorm(x)
# normalization = lambda x: nn.Identity()


class MaskedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        mask_center: bool = False,
        dropout: float = 0.1,
        emb_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert out_channels % 2 == 0
        self.mask_center = mask_center

        if in_channels % 2 != 0:
            self.proj_x = nn.Conv2d(in_channels, in_channels + 1, 1)
            in_channels += 1
        else:
            self.proj_x = nn.Identity()

        if emb_channels:
            self.emb_proj = nn.Sequential(
                activation,
                nn.Conv2d(emb_channels, 2 * in_channels, 1),
            )
        in_ch = kwargs["in_channels"] = in_channels // 2
        out_ch = kwargs["out_channels"] = out_channels // 2
        kernel_size = nn.modules.utils._pair(kernel_size)  # type: ignore
        kwargs["dilation"] = nn.modules.utils._pair(dilation)  # type: ignore
        calc_padding = lambda dilation, kernel_size: tuple(
            d * (k - 1) // 2 for d, k in zip(dilation, kernel_size)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(
                kernel_size=kernel_size,
                padding=calc_padding(kwargs["dilation"], kernel_size),
                **kwargs,
            ),
            normalization(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, 1),
        )
        k0, k1 = kernel_size  # type: ignore
        self.conv_h = nn.Sequential(
            nn.Conv2d(
                kernel_size=(1, k1),
                padding=calc_padding(kwargs["dilation"], (1, k1)),
                **kwargs,
            ),
            normalization(out_ch),
            activation,
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.merge = nn.Sequential(
            normalization(out_ch),
            nn.Conv2d(out_ch, 4 * out_ch, 1),
            activation,
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(4 * out_ch, out_ch, 1)),
        )
        self.skip_connection = (
            nn.Identity() if out_ch == in_ch else nn.Conv2d(in_ch, out_ch, 1)
        )

        self.register_buffer("mask_v", torch.ones(1, 1, k0, k1))
        self.mask_v[..., k0 // 2 + 1 :, :] = 0
        self.register_buffer("mask_h", torch.ones(1, 1, 1, k1))
        self.mask_h[..., k1 // 2 + 1 :] = 0

        if self.mask_center:
            self.mask_v[..., k0 // 2, :] = 0
            self.mask_h[..., k1 // 2] = 0

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        assert x.ndim == 4
        self.conv_h[0].weight.data *= self.mask_h
        self.conv_v[0].weight.data *= self.mask_v

        x = self.proj_x(x)
        if y is not None:
            scale, shift = self.emb_proj(y).chunk(2, dim=1)
            x = x * (1 + scale) + shift
        h, v = x.chunk(2, dim=1)
        v = self.conv_v(v)
        h = self.skip_connection(h) + self.merge(self.conv_h(h) + v)
        x = torch.cat([h, v], dim=1)
        return x


class PixelCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 32,
        channel_mult: List[int] = [1, 2, 4],
        dilation_rates: List[int] = [2, 4, 2],
        dropout: float = 0.0,
    ):
        super().__init__()
        assert len(channel_mult) == len(dilation_rates)

        ch = int(channel_mult[0] * model_channels)
        self.stem = MaskedBlock(
            in_channels, ch, kernel_size=7, mask_center=True, dropout=dropout
        )
        emb_ch = ch
        self.y_embed = nn.Sequential(
            nn.Conv2d(1, emb_ch, 1),  # assumes y is 1xhxw
            normalization(emb_ch),
            activation,
            nn.Conv2d(emb_ch, emb_ch, 1),
        )
        self.blocks = nn.ModuleList()

        kwargs = {"dropout": dropout, "emb_channels": emb_ch}
        for i, mult in enumerate(channel_mult):
            out_ch, di = int(mult * model_channels), dilation_rates[i]
            self.blocks.append(MaskedBlock(ch, out_ch, 5, dilation=di, **kwargs))
            self.blocks.append(MaskedBlock(out_ch, out_ch, 3, dilation=1, **kwargs))
            ch = out_ch

        ch = ch // 2
        self.head = nn.Sequential(
            LambdaModule(lambda x: x.chunk(2, dim=1)[0]),  # get h features only
            normalization(ch),
            nn.Conv2d(ch, 4 * ch, 1),
            activation,
            zero_module(nn.Conv2d(4 * ch, out_channels, 1)),
        )

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        assert x.ndim == 4
        if y is not None:
            y = self.y_embed(y)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x, y)
        x = self.head(x)
        return x
