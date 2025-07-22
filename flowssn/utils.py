from typing import Iterable, Callable, Any

import copy
import random
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def seed_all(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def count_params(module: nn.Module | nn.ModuleList):
    if isinstance(module, nn.Module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    elif isinstance(module, nn.ModuleList):
        num_params = 0.0
        for m in module:
            num_params += sum(p.numel() for p in m.parameters() if p.requires_grad)
        return num_params
    else:
        raise NotImplementedError


class LambdaModule(nn.Module):
    def __init__(self, lambda_fn: Callable[..., Any]):
        super().__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x: torch.Tensor):
        return self.lambda_fn(x)


class EMA:
    def __init__(self, params: Iterable[nn.Parameter], rate: float = 0.999):
        self.rate = rate
        self.params = list(params)  # reference
        self.ema_params = [
            copy.deepcopy(p).detach().requires_grad_(False) for p in self.params
        ]
        self.ema_loss = None

    @torch.no_grad()
    def update(self):
        for ema_p, p in zip(self.ema_params, self.params):
            ema_p.mul_(self.rate).add_(p, alpha=1 - self.rate)

    @torch.no_grad()
    def apply(self):
        self.stored_params = [p.clone() for p in self.params]
        for p, ema_p in zip(self.params, self.ema_params):
            p.copy_(ema_p)

    @torch.no_grad()
    def restore(self):
        assert getattr(self, "stored_params") is not None
        for p, stored_p in zip(self.params, self.stored_params):
            p.copy_(stored_p)
        del self.stored_params

    @torch.no_grad()
    def update_loss(self, loss: float, rate: float = 0.99):
        assert self.ema_loss is not None
        self.ema_loss = rate * self.ema_loss + (1 - rate) * loss


@torch.no_grad()
def plots(model: nn.Module, dataloader: DataLoader, save_path: str):
    M = 6
    viz_batch = next(iter(dataloader))
    viz_batch["x"] = viz_batch["x"][:M].clone().cuda() * 2 - 1  # [-1,1]
    y_obs = viz_batch["y"][:M].clone().cuda()
    dataset = dataloader.dataset.__class__.__name__  # bit hacky

    y_obs = y_obs.permute(0, *range(2, y_obs.ndim), 1)  # to channels last
    b, *_, r = y_obs.shape
    idx = torch.distributions.Categorical(probs=torch.ones(r) / r).sample([b])
    y_rand_rater = y_obs[torch.arange(b), ..., idx]  # choose random rater
    y_for_plot = y_rand_rater.clone().unsqueeze(1).repeat(1, 3, 1, 1)
    y_obs = nn.functional.one_hot(y_rand_rater.long(), num_classes=2).float()
    # (b, 2, h, w)
    y_obs = y_obs.permute(0, -1, *range(1, y_obs.ndim - 1))  # to channels first

    viz_batch.pop("y")

    mc_samples = 32
    while True:
        try:
            out = model(viz_batch, mc_samples=mc_samples)
            break
        except:
            torch.cuda.empty_cache()
            mc_samples = max(mc_samples - 8, 8)

    # (m, b, h, w, k)
    if "probs" in out.keys():
        probs = out["probs"]
        log_probs = probs.clamp(min=1e-12).log()
    else:
        log_probs = out["logits"].log_softmax(dim=-1)
        probs = torch.exp(log_probs)
    # (b, h, w, k),  H[E_p(n|x;w)[p(y|n)]]
    log_mean = torch.logsumexp(log_probs, dim=0) - np.log(mc_samples)
    mean = torch.exp(log_mean)
    # (b, h, w)
    pred_unc = -torch.sum(mean * log_mean, dim=-1)
    # (b, h, w),  E_p(n|x;w)[H[p(y|n)]]
    alea_unc = torch.mean(-torch.sum(probs * log_probs, dim=-1), dim=0)
    # (b, h, w), I[y, n | x, w] = H[E_p(n|x;w)[p(y|n)]] - E_p(n|x;w)[H[p(y|n)]]
    dist_unc = torch.clamp(pred_unc - alea_unc, min=0)

    uncertainty = {}
    for k, v in zip(
        ["predictive", "aleatoric", "distributional"],
        [pred_unc, alea_unc, dist_unc],
    ):
        spatial_max = v.view(v.shape[0], -1).max(dim=1)[0]
        v = v / spatial_max[..., None, None]  # [0, 1]
        # (b, k, h, w)
        uncertainty[k] = v.unsqueeze(1).repeat(1, 3, 1, 1)
        uncertainty[k][:, 1, :, :] = 0.0
        uncertainty[k][:, 2, :, :] = 0.0
        if (uncertainty[k] < 0).sum() > 0:
            print(k)  # checks for neg uncertainties

    out = model(viz_batch, mc_samples=M)
    # (m, m, h, w, k)
    if "probs" in out.keys():
        samples = out["probs"]
    else:
        samples = out["logits"].softmax(dim=-1)
    # (m*m, h, w, k),
    samples = samples.reshape(-1, *samples.shape[2:])

    if dataset == "LIDC":
        samples = samples.argmax(-1).unsqueeze(1).repeat(1, 3, 1, 1)
        mean = mean.argmax(-1).unsqueeze(1).repeat(1, 3, 1, 1)
    elif dataset == "RefugeMultirater":
        samples = samples.argmax(-1).unsqueeze(1).repeat(1, 3, 1, 1)
        mean = mean.argmax(-1).unsqueeze(1).repeat(1, 3, 1, 1)
    else:
        raise NotImplementedError

    r = 3 if viz_batch["x"].shape[1] == 1 else 1
    imgs = torch.cat(
        [
            (viz_batch["x"].clone().cpu().repeat(1, r, 1, 1) + 1) * 0.5,
            y_for_plot.cpu(),
            mean.cpu(),
            uncertainty["predictive"].cpu(),
            uncertainty["aleatoric"].cpu(),
            uncertainty["distributional"].cpu(),
            samples.cpu(),
        ],
        dim=0,
    )

    fig = plt.figure(figsize=(imgs.shape[0], imgs.shape[0]))
    ax = fig.add_subplot()
    ax.axis("off")
    ax.imshow(make_grid(imgs, nrow=M, padding=1).permute(1, 2, 0))
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close(fig)
    torch.cuda.empty_cache()  # seems to help
