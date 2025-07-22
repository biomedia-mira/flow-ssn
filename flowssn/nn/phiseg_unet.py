from typing import List
import torch
import torch.nn as nn

activation = nn.ReLU()
normalization = lambda x: nn.BatchNorm2d(x)


class PHiSegUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        model_channels: int = 32,
        channel_mult: List[int] = [1, 2, 4, 6, 6, 6, 6],
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        widths = [int(c * model_channels) for c in channel_mult]
        in_ch = in_channels

        for out_ch in widths:
            self.encoder.append(self.block(in_ch, out_ch))
            in_ch = out_ch

        self.decoder = nn.ModuleList()
        rev_widths = widths[::-1]

        for i in range(len(rev_widths) - 1):
            concat_ch = rev_widths[i] + rev_widths[i + 1]
            k, p = (1, 0) if i == len(widths) - 2 else (3, 1)
            self.decoder.append(self.block(concat_ch, rev_widths[i + 1], k, p))

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        kwargs = dict(kernel_size=kernel_size, padding=padding, bias=False)
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),  # type: ignore
            normalization(out_channels),
            activation,
            *[
                nn.Conv2d(out_channels, out_channels, **kwargs),  # type: ignore
                normalization(out_channels),
                activation,
            ]
            * 2,
        )

    def forward(self, x: torch.Tensor):
        skips = []
        for i, layer in enumerate(self.encoder):
            x = self.downsample(x) if i > 0 else x
            x = layer(x)
            skips.append(x)

        for i, layer in enumerate(self.decoder):
            x = self.upsample(x)
            x = torch.cat([x, skips[-(i + 2)]], dim=1)
            x = layer(x)
        x = self.head(x)
        return x
