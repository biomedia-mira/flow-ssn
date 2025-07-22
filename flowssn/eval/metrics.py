from typing import Optional, List, Optional, Union

import torch
from scipy.optimize import linear_sum_assignment

"""m: #samples, n: #labels, b: batch size, h: height, w: width, k: #classes"""
BackgroundFilter = Optional[Union[bool, List[int]]]


def is_one_hot(x: torch.Tensor):
    return x.ndim >= 2 and ((x == 0) | (x == 1)).all() and (x.sum(dim=-1) == 1).all()


def check_inputs(x: torch.Tensor, filter_bg: BackgroundFilter = None):
    x = x.unsqueeze(-1) if not is_one_hot(x) else x
    if filter_bg is not None:
        x = (
            x[..., filter_bg]  # specifies list of class indices to keep
            if isinstance(filter_bg, List)
            else x[..., :-1]  # assumes background is the last class index
        )
    return x


def intersection_over_union(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: torch.types._size | int | None,
    eps: float = 1e-8,
):
    assert x.ndim == y.ndim
    # (..., b, h, w, k), (..., b, h, w, k)
    x, y = x.float(), y.float()
    # (..., b, k)
    intersection = torch.sum(x * y, dim=dim)
    total_area = torch.sum(x + y, dim=dim)
    union = total_area - intersection

    if y.shape[-1] > 1:  # if multiclass
        iou = (intersection + eps) / (union + eps)
        iou[total_area == 0] = torch.nan
        return torch.nanmean(iou, dim=-1)
    else:  # LIDC
        iou = intersection / union
        iou[total_area == 0] = 1.0
        return torch.mean(iou, dim=-1)


def jaccard_distance(x: torch.Tensor, y: torch.Tensor):
    assert x.ndim == y.ndim and x.ndim == 5
    # (m, 1, b, h, w, k), (1, n, b, h, w, k)
    x, y = x.unsqueeze(1), y.unsqueeze(0)
    try:
        jd = 1 - intersection_over_union(x.clone(), y.clone(), dim=(-3, -2))
    except RuntimeError:
        torch.cuda.empty_cache()
        # print("OOM: jaccard_distance")
        jd = _jaccard_distance_looped(x, y)
    # (m, n, b)
    return jd


def _jaccard_distance_looped(x: torch.Tensor, y: torch.Tensor):
    assert x.ndim == y.ndim and x.ndim == 6
    m, n, b = x.shape[0], y.shape[1], x.shape[2]
    iou = torch.zeros(m, n, b, device=x.device, dtype=torch.float32)
    for i in range(m):
        for j in range(n):
            # (b, h, w, k), (b, h, w, k)
            _x, _y = x[i, 0, ...], y[0, j, ...]
            # (b,)
            iou[i, j, :] = intersection_over_union(_x, _y, dim=(-3, -2))
    # (m, n, b)
    return 1 - iou


def energy_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    filter_bg: BackgroundFilter = None,
):
    assert x.ndim == y.ndim and x.ndim > 3
    # (m, b, h, w, k), (n, b, h, w, k)
    x, y = check_inputs(x, filter_bg), check_inputs(y, filter_bg)
    # (m, n, b)
    d_xy = jaccard_distance(x, y)
    # (m, m, b)
    d_xx = jaccard_distance(x, x)
    # (n, n, b)
    d_yy = jaccard_distance(y, y)

    d_xx = torch.mean(d_xx, dim=(0, 1))
    if weights is not None:
        # (b,), weights are of shape (n, b)
        d_xy = torch.sum(torch.mean(d_xy, dim=0) * weights, dim=0)
        d_yy = torch.sum(d_yy * weights.unsqueeze(0) * weights.unsqueeze(1), dim=(0, 1))
    else:
        # (b,)
        d_xy = torch.mean(d_xy, dim=(0, 1))
        d_yy = torch.mean(d_yy, dim=(0, 1))

    ged_sq = 2 * d_xy - d_xx - d_yy
    return ged_sq, d_xx


def hungarian_matched_iou(
    x: torch.Tensor, y: torch.Tensor, filter_bg: BackgroundFilter = None
):
    assert x.ndim == y.ndim and x.ndim > 3
    # (m, b, h, w, k), (n, b, h, w, k)
    x, y = check_inputs(x, filter_bg), check_inputs(y, filter_bg)
    # (m, n, b)
    jd = jaccard_distance(x, y)
    cost = jd.cpu().numpy()
    hm_ious = []
    for i in range(cost.shape[-1]):  # batch_size
        if torch.isnan(torch.from_numpy(cost[:, :, i])).sum() > 0:
            print("NaNs: hungarian_matched_iou")
            continue
        row_idx, col_idx = linear_sum_assignment(cost[:, :, i])
        hm_ious.append(torch.from_numpy(1 - cost[row_idx, col_idx, i]).mean())
    # (b,)
    hm_iou = torch.stack(hm_ious)
    return hm_iou


def mean_iou(x: torch.Tensor, y: torch.Tensor, filter_bg: BackgroundFilter = None):
    assert x.ndim == y.ndim and x.ndim > 2
    # (b, h, w, k), (b, h, w, k)
    x, y = check_inputs(x, filter_bg), check_inputs(y, filter_bg)
    try:
        # (b,)
        iou = intersection_over_union(x, y, dim=(-3, -2))
    except RuntimeError:
        torch.cuda.empty_cache()
        print("OOM: mean_iou")
        iou = []
        for i in range(y.shape[0]):  # batch_size
            # (h, w, k), (h, w, k)
            iou.append(intersection_over_union(x[i], y[i], dim=(-3, -2)))
        # (b,)
        iou = torch.stack(iou, dim=0)
    # (b,)
    return iou


def dice_score(
    x: torch.Tensor,
    y: torch.Tensor,
    filter_bg: BackgroundFilter = None,
    eps: float = 1e-8,
):
    assert x.ndim == y.ndim and x.ndim > 2
    # (b, h, w, k)
    x = check_inputs(x, filter_bg).float()
    y = check_inputs(y, filter_bg).float()
    # (b, k)
    intersection = torch.sum(x * y, dim=(-3, -2))
    total_area = torch.sum(x + y, dim=(-3, -2))
    dice = (2 * intersection + eps) / (total_area + eps)
    # (b,)
    return torch.mean(dice, dim=-1)


def expected_dice(
    x: torch.Tensor,
    y: torch.Tensor,
    filter_bg: BackgroundFilter = None,
):
    assert x.ndim == y.ndim and x.ndim > 2
    # E_y~data,y'~model[dice(y, y')]
    mc_samples, *_ = x.shape
    num_raters, *_ = y.shape
    dice = torch.zeros(x.shape[1], device=x.device)
    for m in range(mc_samples):
        for r in range(num_raters):
            dice += dice_score(x[m], y[r], filter_bg)
    # (b,)
    return dice / (mc_samples * num_raters)


def dice_expected(
    x: torch.Tensor,
    y: torch.Tensor,
    filter_bg: BackgroundFilter = None,
):
    assert x.ndim == (y.ndim - 1) and x.ndim > 2
    # E_y~data[dice(y, E_y'~model[Y'])]
    num_raters, *_ = y.shape
    dice = torch.zeros(x.shape[0], device=x.device)
    for r in range(num_raters):
        dice += dice_score(x, y[r], filter_bg)
    # (b,)
    return dice / num_raters
