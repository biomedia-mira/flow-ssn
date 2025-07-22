from typing import Optional, Callable, Dict

import os
import torch
import argparse
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torch._prims_common import DeviceLikeType
from torchvision import tv_tensors
from torchvision.transforms import v2

from flowssn.data.utils import cache_data


class RefugeMultirater(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        cache_res: int = 128,
    ):
        self.split = split
        self.transform = transform
        self.cache_res = cache_res
        self.df = pd.read_csv(os.path.join(root, f"./refuge_{split}.csv"))
        self.images = [i for i in self.df["img_path"]]

        if split == "train":
            self.images = [i.split(".")[0] + ".png" for i in self.images]

        self.cached_images = cache_data(
            load_fn=lambda x: self._load_image(x, rgb=True),
            files=self.images,
            name=f"RefugeMultirater {self.split} images",
        )
        self.cached_cups = {}
        for i in range(1, 8):
            self.cached_cups[i] = cache_data(
                load_fn=lambda x: self._load_image(x),
                files=[j for j in self.df[f"seg_cup_{i}_path"]],
                name=f"RefugeMultirater {self.split} cups_{i}",
            )

    def __len__(self):
        return len(self.images)

    def _load_image(self, file: str, rgb: bool = False):
        with Image.open(file) as img:
            img = img.convert("RGB") if rgb else img.convert("L")
            img = img.resize((self.cache_res, self.cache_res))
        return v2.functional.pil_to_tensor(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # (c, h, w)
        x = tv_tensors.Image(self.cached_images[idx])
        # (n, 1, h, w)
        y = torch.stack([self.cached_cups[m][idx] for m in range(1, 8)], dim=0)
        # (n, h, w)
        y = tv_tensors.Mask(y.squeeze(1))

        if self.transform is not None:
            x, y = self.transform(x, y)
        return dict(x=x, y=y / 255)


def get_refuge(args: argparse.Namespace) -> Dict[str, RefugeMultirater]:
    transform = dict(
        train=v2.Compose(
            [
                v2.Resize(args.resolution),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=(-20, 20)),
                v2.RandomResizedCrop(args.resolution, scale=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                v2.ToDtype(torch.float32, scale=True),  # [0,1]
            ]
        ),
        eval=v2.Compose(
            [
                v2.Resize(args.resolution),
                v2.ToDtype(torch.float32, scale=True),  # [0,1]
            ]
        ),
    )
    datasets = {
        k: RefugeMultirater(
            root=args.data_dir,
            split=k,
            transform=transform[(k if k == "train" else "eval")],
            cache_res=args.resolution,
        )
        for k in ["train", "valid", "test"]
    }
    return datasets


def preprocess_refuge_fn(
    batch: Dict[str, torch.Tensor],
    device: Optional[DeviceLikeType] = None,
) -> Dict[str, torch.Tensor]:
    # b: batch size, c: #channels, h/w: height/width, r: #raters, k: #classes
    # (b, c, h, w)
    batch["x"] = batch["x"].to(device) * 2 - 1  # [-1, 1]
    y = batch["y"].to(device)
    # (b, h, w, r)
    y = y.permute(0, *range(2, y.ndim), 1)  # channels last
    b, _, _, r = y.shape
    batch["y_all"] = y.clone().float()
    # sample a random annotation for each batch element
    idx = torch.distributions.Categorical(probs=torch.ones(r) / r).sample([b])
    # (b, h, w, k)
    batch["y"] = torch.nn.functional.one_hot(
        y[torch.arange(b), ..., idx].long(), num_classes=2
    )
    return batch
