from typing import Optional, Tuple

import torch
import torch.nn as nn

from .utils import zero_module

from .utils import timestep_embedding
from flowssn.utils import LambdaModule


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        context_shape: Tuple[int, int, int],
        strip_size: Tuple[int, int],
        out_channels: int = 4,
        embed_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 6,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.embed_x = StripEmbedding(input_shape, strip_size, embed_dim)
        n = self.embed_x.num_strips

        self.embed_y = nn.Sequential(
            StripEmbedding(context_shape, strip_size, embed_dim),
            nn.LayerNorm(embed_dim),
            activation,
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_t = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            activation,
            nn.Linear(embed_dim, embed_dim),
        )

        self.transformer = TransformerDecoder(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=dropout,
            activation=activation,
            num_layers=num_blocks,
        )
        self.register_buffer("tgt_mask", torch.triu(torch.ones(n, n), 1).bool())
        self.bos_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.empty(1, n, embed_dim))
        self.pos_embed_y = nn.Parameter(torch.empty(1, n, embed_dim))
        nn.init.trunc_normal_(self.bos_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_y, mean=0.0, std=0.02)

        self.head = nn.Sequential(
            LambdaModule(
                lambda x: x.permute(0, 2, 1).view(-1, embed_dim, *self.embed_x.grid_hw)
            ),
            nn.ConvTranspose2d(
                embed_dim, out_channels, kernel_size=strip_size, stride=strip_size
            ),
            activation,
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ):
        # (b, seq_len, emb_dim)
        x = self.embed_x(x) + self.pos_embed
        memory = None
        if y is not None:
            # (b, seq_len, emb_dim)
            memory = self.embed_y(y) + self.pos_embed_y
        if t is not None:
            # (b, 1, emb_dim)
            t = self.embed_t(timestep_embedding(t, self.embed_dim)).unsqueeze(1)
            memory = memory + t if memory is not None else t
        # (b, 1, emb_dim)
        bos_token = self.bos_token.repeat(x.shape[0], 1, 1)
        # (b, seq_len, embd_dim), target is x shifted to the right by one
        tgt = torch.cat([bos_token, x[:, :-1, :].clone()], dim=1)
        if memory is not None:
            tgt = tgt + memory
        # (b, seq_len, embd_dim)
        out = self.transformer(tgt, memory, self.tgt_mask, tgt_is_causal=True)
        # (b, out_ch, h, w)
        return self.head(out)


class StripEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        strip_size: Tuple[int, int],
        embed_dim: int,
    ):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        if h % strip_size[0] != 0 or w % strip_size[1] != 0:
            raise ValueError("image height/width not divisible by strip size(s)")
        self.strip_size = strip_size
        self.embed_dim = embed_dim
        self.grid_hw = h // strip_size[0], w // strip_size[1]
        self.num_strips = self.grid_hw[0] * self.grid_hw[1]
        self.conv = nn.Conv2d(c, embed_dim, kernel_size=strip_size, stride=strip_size)

    def forward(self, x: torch.Tensor):
        # (batch_size, embed_dim, grid_h, grid_w)
        x = self.conv(x)
        # (batch_size, num_strips, embed_dim)
        x = x.view(x.shape[0], self.embed_dim, -1).permute(0, 2, 1)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        kwargs = {"dropout": dropout, "batch_first": True}
        self.self_attn = nn.MultiheadAttention(d_model, nhead, **kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, **kwargs)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
    ):
        x = self.norm1(tgt)
        out = self.self_attn(
            x, x, x, attn_mask=tgt_mask, is_causal=tgt_is_causal, need_weights=False
        )[0]
        x = tgt + self.dropout1(out)
        if memory is not None:
            out = self.multihead_attn(
                self.norm2(x), memory, memory, need_weights=False
            )[0]
            x = x + self.dropout2(out)
        out = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(x)))))
        x = x + self.dropout3(out)
        return x

    def _reset_parameters(self):
        zero_module(self.self_attn.out_proj)
        zero_module(self.multihead_attn.out_proj)
        zero_module(self.linear2)
        nn.init.xavier_uniform_(self.linear1.weight, nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.linear1.bias)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        num_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
    ):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_is_causal)
        return self.norm(x)
