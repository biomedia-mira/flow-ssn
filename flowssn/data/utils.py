from typing import Callable, List

import numpy as np
import torch

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def cache_data(
    load_fn: Callable[[str], torch.Tensor | np.ndarray], files: List[str], name: str
):
    with ThreadPoolExecutor() as executor:
        return list(
            tqdm(
                executor.map(load_fn, files),
                total=len(files),
                desc=f"Caching {name}",
                mininterval=15,
                disable=True,
            )
        )
