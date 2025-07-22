"""
Much simpler version of the UNet from: github.com/openai/guided-diffusion
"""

from typing import Optional, List, Tuple

import math
import torch
import torch.nn as nn

from .utils import zero_module, timestep_embedding


activation = nn.SiLU()
normalization = lambda x: nn.GroupNorm(max(1, x // 16), x)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        emb_channels: Optional[int] = None,
        dropout: float = 0.1,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            normalization(in_channels),
            activation,
            nn.Conv2d(in_channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        self.h_upd = self.x_upd = (
            Upsample() if up else Downsample() if down else nn.Identity()
        )

        if self.emb_channels:
            self.emb_layers = nn.Sequential(
                activation,
                nn.Linear(self.emb_channels, 2 * self.out_channels),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        self.skip_connection = (
            nn.Identity()
            if self.out_channels == in_channels
            else nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]):
        emb = None
        if isinstance(x, tuple):
            x, emb = x

        if self.updown:
            h = self.in_layers[:-1](x)
            h, x = self.h_upd(h), self.x_upd(x)  # up/down sampling
            h = self.in_layers[-1](h)
        else:
            h = self.in_layers(x)

        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype)
            if emb_out.ndim == 4:
                emb_out = emb_out.permute(0, 3, 1, 2)  # to channels first
            else:
                while len(emb_out.shape) < len(h.shape):
                    emb_out = emb_out[..., None]
            scale, shift = emb_out.chunk(2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(
        self, in_channels: int, num_heads: int = 1, num_head_channels: int = -1
    ):
        super().__init__()
        self.in_channels = in_channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                in_channels % num_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = in_channels // num_head_channels
        self.norm = normalization(in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(x, tuple):
            x, _ = x
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    def __init__(self, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor):
        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct, bcs -> bts",
            (q * scale).view(bs * self.num_heads, ch, length),
            (k * scale).view(bs * self.num_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts, bcs -> bct", weight, v.reshape(bs * self.num_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        model_channels: int,
        out_channels: int = 4,
        num_res_blocks: int = 1,
        attention_resolutions: List[int] = [],
        dropout: float = 0.0,
        channel_mult: List[int] = [1, 2, 4, 8],
        num_heads: int = 1,
        num_head_channels: int = 64,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.image_size = input_shape[-1]
        self.in_channels = input_shape[0]
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        emb_channels = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_channels),
            activation,
            nn.Linear(emb_channels, emb_channels),
        )
        # image-based conditioning
        ch = int(channel_mult[0] * model_channels)
        self.label_embed = nn.Sequential(
            nn.Conv2d(1, ch, 1),  # NOTE: change me, hard coded input channels
            normalization(ch),
            activation,
            nn.Conv2d(ch, ch, 1),
        )
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.stem = nn.Conv2d(self.in_channels, ch, 3, padding=1)
        self.encoder_blocks = nn.ModuleList([])
        encoder_ch = [ch]
        res = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):

                layers = []
                out_ch = int(mult * model_channels)
                layers.append(ResBlock(ch, out_ch, emb_channels, dropout))
                ch = out_ch

                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))

                self.encoder_blocks.append(nn.Sequential(*layers))
                encoder_ch.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.encoder_blocks.append(
                    ResBlock(ch, out_ch, emb_channels, dropout, down=True)
                )
                ch = out_ch
                encoder_ch.append(ch)
                res *= 2  # downsampling factor

        if attention_resolutions[0] == -1:
            self.middle_block = ResBlock(ch, emb_channels=emb_channels, dropout=dropout)
        else:
            self.middle_block = nn.Sequential(
                ResBlock(ch, emb_channels=emb_channels, dropout=dropout),
                AttentionBlock(ch, num_heads, num_head_channels),
                ResBlock(ch, emb_channels=emb_channels, dropout=dropout),
            )

        self.decoder_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):

                layers = []
                in_ch = encoder_ch.pop()
                out_ch = int(mult * model_channels)
                layers.append(ResBlock(ch + in_ch, out_ch, emb_channels, dropout))
                ch = out_ch

                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))

                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, out_ch, emb_channels, dropout, up=True))
                    res //= 2
                self.decoder_blocks.append(nn.Sequential(*layers))

        self.head = nn.Sequential(
            normalization(ch),
            activation,
            zero_module(nn.Conv2d(input_ch, self.out_channels, 3, padding=1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        emb = None
        if t is not None:
            # (b, emb_dim)
            emb = self.time_embed(timestep_embedding(t, self.model_channels))

        y_emb = None
        if y is not None:
            y = y.repeat(1, 3, 1, 1) if y.shape[1] == 1 else y
            drop = None
            if self.training:
                drop = torch.rand(y.shape[0], 1, 1, 1, device=y.device) > 0.1
            y_emb = self.label_embed(y if drop is None else y * drop)  # v1

        hs = []
        h = self.stem(x)
        if y_emb is not None:
            h = h + y_emb
        hs.append(h)

        res = h.shape[-1]
        for block in self.encoder_blocks:
            # checks if the next block downsamples the input
            if isinstance(block, ResBlock) and block.updown:
                res //= 2
            h = block((h, emb))
            hs.append(h)

        h = self.middle_block((h, emb))

        for block in self.decoder_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block((h, emb))
            res = h.shape[-1]
        return self.head(h)
