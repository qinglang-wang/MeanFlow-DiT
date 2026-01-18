import torch
from utils.utils import sg

def adaptive_l2_loss(error: torch.Tensor, gamma: int =0.5, c: int=1e-3) -> torch.Tensor:
    """Adaptive L2 loss as defined in MeanFlow paper."""
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    return (sg(w) * delta_sq).mean()
