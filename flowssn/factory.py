from typing import TypedDict, Literal, Optional, Tuple, Union, List

import argparse
import torch.nn as nn

from flowssn.nn.unet import UNetModel
from flowssn.nn.pixel_cnn import PixelCNN
from flowssn.nn.phiseg_unet import PHiSegUNet
from flowssn.nn.transformer import Transformer


class TransformerConfig(TypedDict):
    input_shape: Tuple[int, int, int]
    context_shape: Tuple[int, int, int]
    strip_size: Tuple[int, int]
    out_channels: int
    embed_dim: int
    num_blocks: int
    num_heads: int
    dropout: float


class UNetConfig(TypedDict):
    input_shape: Tuple[int, int, int]
    model_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: List[int]
    dropout: float
    channel_mult: List[int]
    num_heads: int
    num_head_channels: int


class PHiSegUNetConfig(TypedDict):
    in_channels: int
    out_channels: int
    model_channels: int
    channel_mult: List[int]


class PixelCNNConfig(TypedDict):
    in_channels: int
    out_channels: int
    model_channels: int
    channel_mult: List[int]
    dilation_rates: List[int]
    dropout: float


Config = Union[TransformerConfig, PixelCNNConfig, UNetConfig, PHiSegUNetConfig]


def continuous_flowssn_args(parser: argparse.ArgumentParser):
    parser.add_argument("--eval_T", type=int, default=10)
    parser.add_argument("--cond_base", action="store_true", default=False)
    parser.add_argument("--cond_flow", action="store_true", default=False)
    parser.add_argument("--base_std", type=float, default=0.0)


def autoregressive_flowssn_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--flow_type",
        type=str,
        choices=["default", "gated", "volume_preserving"],
        default="default",
    )
    parser.add_argument("--num_flows", type=int, default=1)
    parser.add_argument("--cond_base", action="store_true", default=False)
    parser.add_argument("--cond_flow", action="store_true", default=False)
    parser.add_argument("--base_std", type=float, default=0.0)


def transformer_args(parser: argparse.ArgumentParser, prefix: Optional[str] = ""):
    p = prefix
    parser.add_argument(f"--{p}input_shape", type=int, nargs=3, default=(2, 128, 128))
    parser.add_argument(f"--{p}context_shape", type=int, nargs=3, default=(1, 128, 128))
    parser.add_argument(f"--{p}strip_size", type=int, nargs=2, default=(1, 16))
    parser.add_argument(f"--{p}out_channels", type=int, default=4)
    parser.add_argument(f"--{p}embed_dim", type=int, default=128)
    parser.add_argument(f"--{p}num_blocks", type=int, default=6)
    parser.add_argument(f"--{p}num_heads", type=int, default=2)
    parser.add_argument(f"--{p}dropout", type=float, default=0.1)


def unet_args(parser: argparse.ArgumentParser, prefix: Optional[str] = ""):
    p = prefix
    parser.add_argument(f"--{p}input_shape", type=int, nargs=3, default=(2, 128, 128))
    parser.add_argument(f"--{p}model_channels", type=int, default=32)
    parser.add_argument(f"--{p}out_channels", type=int, default=192)
    parser.add_argument(f"--{p}num_res_blocks", type=int, default=1)
    parser.add_argument(f"--{p}attention_resolutions", nargs="+", type=int, default=[])
    parser.add_argument(f"--{p}dropout", type=float, default=0.1)
    parser.add_argument(f"--{p}channel_mult", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument(f"--{p}num_heads", type=int, default=1)
    parser.add_argument(f"--{p}num_head_channels", type=int, default=64)


def phiseg_unet_args(parser: argparse.ArgumentParser, prefix: Optional[str] = ""):
    p = prefix
    parser.add_argument(f"--{p}in_channels", type=int, default=1)
    parser.add_argument(f"--{p}out_channels", type=int, default=2)
    parser.add_argument(f"--{p}model_channels", type=int, default=32)
    parser.add_argument(f"--{p}channel_mult", nargs="+", type=int, default=[1, 2, 4, 8])


def pixel_cnn_args(parser: argparse.ArgumentParser, prefix: Optional[str] = ""):
    p = prefix
    parser.add_argument(f"--{p}in_channels", type=int, default=2)
    parser.add_argument(f"--{p}out_channels", type=int, default=2)
    parser.add_argument(f"--{p}model_channels", type=int, default=32)
    parser.add_argument(f"--{p}channel_mult", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(f"--{p}dilation_rates", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(f"--{p}dropout", type=float, default=0.1)


def parse_ssn_args(
    name: Literal["c-flowssn", "ar-flowssn"],
    parser: argparse.ArgumentParser,
) -> argparse.Namespace:
    if name == "c-flowssn":
        continuous_flowssn_args(parser)
    elif name == "ar-flowssn":
        autoregressive_flowssn_args(parser)
    else:
        raise NotImplementedError
    args = parser.parse_known_args()[0]
    return args


def parse_nn_args(
    name: Literal["transformer", "unet", "phiseg_unet", "pixel_cnn"],
    parser: argparse.ArgumentParser,
    prefix: Optional[str] = "",
) -> argparse.Namespace:
    if name == "transformer":
        transformer_args(parser, prefix)
    elif name == "unet":
        unet_args(parser, prefix)
    elif name == "phiseg_unet":
        phiseg_unet_args(parser, prefix)  # not used in paper
    elif name == "pixel_cnn":
        pixel_cnn_args(parser, prefix)  # not used in paper
    else:
        raise NotImplementedError
    args = parser.parse_known_args()[0]
    return args


def build_nn(
    name: Literal["transformer", "unet", "phiseg_unet", "pixel_cnn"],
    parser: Optional[argparse.ArgumentParser] = None,
    args: Optional[argparse.Namespace] = None,
    prefix: Optional[str] = "",
) -> Tuple[nn.Module, argparse.Namespace]:
    if args is None and parser is not None:
        args = parse_nn_args(name, parser, prefix)
    assert args is not None
    p = prefix

    if name == "transformer":
        config: Config = {
            "input_shape": tuple(getattr(args, f"{p}input_shape")),
            "context_shape": tuple(getattr(args, f"{p}context_shape")),
            "strip_size": tuple(getattr(args, f"{p}strip_size")),
            "out_channels": getattr(args, f"{p}out_channels"),
            "embed_dim": getattr(args, f"{p}embed_dim"),
            "num_blocks": getattr(args, f"{p}num_blocks"),
            "num_heads": getattr(args, f"{p}num_heads"),
            "dropout": getattr(args, f"{p}dropout"),
        }
        model = Transformer(**config)
    elif name == "unet":
        config: Config = {
            "input_shape": tuple(getattr(args, f"{p}input_shape")),
            "model_channels": getattr(args, f"{p}model_channels"),
            "out_channels": getattr(args, f"{p}out_channels"),
            "num_res_blocks": getattr(args, f"{p}num_res_blocks"),
            "attention_resolutions": getattr(args, f"{p}attention_resolutions"),
            "dropout": getattr(args, f"{p}dropout"),
            "channel_mult": getattr(args, f"{p}channel_mult"),
            "num_heads": getattr(args, f"{p}num_heads"),
            "num_head_channels": getattr(args, f"{p}num_head_channels"),
        }
        model = UNetModel(**config)
    elif name == "pixel_cnn":
        config: Config = {
            "in_channels": getattr(args, f"{p}in_channels"),
            "out_channels": getattr(args, f"{p}out_channels"),
            "model_channels": getattr(args, f"{p}model_channels"),
            "channel_mult": getattr(args, f"{p}channel_mult"),
            "dilation_rates": getattr(args, f"{p}dilation_rates"),
            "dropout": getattr(args, f"{p}dropout"),
        }
        model = PixelCNN(**config)  # not used in paper
    elif name == "phiseg_unet":
        config: Config = {
            "in_channels": getattr(args, f"{p}in_channels"),
            "out_channels": getattr(args, f"{p}out_channels"),
            "model_channels": getattr(args, f"{p}model_channels"),
            "channel_mult": getattr(args, f"{p}channel_mult"),
        }
        model = PHiSegUNet(**config)  # not used in paper
    else:
        raise NotImplementedError
    return model, args
