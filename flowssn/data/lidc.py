from typing import Optional, Callable, Dict

import h5py
import argparse
import torch
import numpy as np

from torch.utils.data import Dataset
from torch._prims_common import DeviceLikeType
from torchvision import tv_tensors
from torchvision.transforms import v2


class LIDC(Dataset):
    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        self.split = split[:3] if split == "valid" else split
        self.transform = transform
        print(f"Loading LIDC {self.split}:")

        with h5py.File(root, "r") as f:
            data = f[self.split]
            images = np.array(data["images"][:], dtype=np.float32)  # type: ignore
            labels = np.array(data["labels"][:], dtype=np.uint8)  # type: ignore
        self.images = (images + 0.5)[:, None, ...]  # to [0,1]
        self.labels = np.moveaxis(labels, 3, 1)
        print(f"images: {self.images.shape}, masks: {self.labels.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = tv_tensors.Image(self.images[idx])
        y = tv_tensors.Mask(self.labels[idx])

        if self.transform is not None:
            x, y = self.transform(x, y)
        return dict(x=x, y=y)


def get_lidc(args: argparse.Namespace) -> Dict[str, LIDC]:
    transform = dict(
        train=v2.Compose(
            [
                v2.Resize(args.resolution),
                v2.RandomChoice([v2.RandomRotation([d, d]) for d in [0, 90, 180, 270]]),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=False),  # [0,1] already
            ]
        ),
        eval=v2.Compose(
            [
                v2.Resize(args.resolution),
                v2.ToDtype(torch.float32, scale=False),  # [0,1] already
            ]
        ),
    )

    datasets = {
        k: LIDC(
            root=args.data_dir,
            split=k,
            transform=transform[(k if k == "train" else "eval")],
        )
        for k in ["train", "valid", "test"]
    }
    return datasets


def preprocess_lidc_fn(
    batch: Dict[str, torch.Tensor], device: Optional[DeviceLikeType] = None
) -> Dict[str, torch.Tensor]:
    # b: batch size, c: #channels, h/w: height/width, r: #raters, k: #classes
    # (b, c, h, w)
    batch["x"] = batch["x"].to(device) * 2 - 1  # [-1, 1]
    y = batch["y"].to(device)
    # (b, h, w, r)
    y = y.permute(0, *range(2, y.ndim), 1)  # to channels last
    b, _, _, r = y.shape
    batch["y_all"] = y.clone().float()
    # sample a random annotation for each batch element
    idx = torch.distributions.Categorical(probs=torch.ones(r) / r).sample([b])
    # (b, h, w, k)
    batch["y"] = torch.nn.functional.one_hot(
        y[torch.arange(b), ..., idx].long(), num_classes=2
    )
    return batch
