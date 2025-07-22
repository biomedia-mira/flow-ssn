import torch
import torch.distributions as dist

from .transforms import (
    MaskedAutoregressive,
    InverseAutoregressive,
    ConditionalMaskedAutoregressive,
    ConditionalInverseAutoregressive,
)


class InverseAutoregressiveFlow(dist.TransformedDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not any(
            type(t) in {MaskedAutoregressive, ConditionalMaskedAutoregressive}
            for t in self.transforms
        )

    def condition(self, h: torch.Tensor):
        transforms = []
        for t in self.transforms:
            transforms.append(t.condition(h) if hasattr(t, "condition") else t)  # type: ignore
        return dist.TransformedDistribution(self.base_dist, transforms)


class MaskedAutoregressiveFlow(dist.TransformedDistribution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not any(
            type(t) in {InverseAutoregressive, ConditionalInverseAutoregressive}
            for t in self.transforms
        )

    def condition(self, h: torch.Tensor):
        transforms = []
        for t in self.transforms:
            transforms.append(t.condition(h) if hasattr(t, "condition") else t)  # type: ignore
        return dist.TransformedDistribution(self.base_dist, transforms)
