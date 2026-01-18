import torch

class Normalizer:
    def __init__(self, mode: str=None, **kwargs):
        assert mode in ['image', 'latent']

        if mode == 'latent':
            assert 'mean' in kwargs.keys() and 'std' in kwargs.keys()
            self.mean = torch.tensor(kwargs['mean']).view(-1, 1, 1).float()
            self.std = torch.tensor(kwargs['std']).view(-1, 1, 1).float()

        self.mode = mode

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'image':
            return x * 2 - 1
        elif self.mode == 'latent':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def reverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'image':
            return (x.clip(-1, 1) + 1) * 0.5
        elif self.mode == 'latent':
            return x * self.std.to(x.device) + self.mean.to(x.device)