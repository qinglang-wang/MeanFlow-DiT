import torch
import numpy as np
from einops import rearrange
from typing import Union
from core.scheduler import SamplerScheduler
from core.normalizer import Normalizer
from core.loss import adaptive_l2_loss
from utils.utils import sg


class MeanFlowEngine(torch.nn.Module):
    def __init__(self, 
                 scheduler: SamplerScheduler,
                 normalizer: Normalizer,
                 num_classes: int=10,
                 cfg_ratio: float=0.1,
                 cfg_scale: float=2.0,
                 jvp_api: str='autograd',
                 image_size: int=32,
                 channels: int=1):
        super().__init__()
        self.scheduler = scheduler
        self.normalizer = normalizer
        self.num_classes = num_classes
        self.cfg_ratio = cfg_ratio
        self.cfg_scale = cfg_scale
        self.jvp_api = jvp_api
        self.image_size = image_size
        self.channels = channels

        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        else:
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

    def forward_loss(self, model, x: torch.Tensor, c: torch.Tensor, iteration: int):
        batch_size = x.shape[0]
        device = x.device

        # Sample timestep
        r, t = self.scheduler.sample(batch_size, iteration, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach()
        r_ = rearrange(r, "b -> b 1 1 1").detach()
        
        # Construct flow
        x = self.normalizer.norm(x)  # x: clean data (t=0)
        e = torch.randn_like(x)  # e: noise (t=1)

        z = (1 - t_) * x + t_ * e
        v = e - x

        # CFG
        if c is not None and self.cfg_ratio > 0:
            # Drop condition logic
            c_null = torch.full_like(c, self.num_classes)
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_ratio
            c_dropped = torch.where(drop_mask, c_null, c)
            
            # Calculate distilled velocity target (v_hat)
            if self.cfg_scale != 1.0:
                # Unconditional forward
                with torch.no_grad():
                    u_uncond = model(z, t, t, c_null) # instantaneous velocity
                
                v_hat = self.cfg_scale * v + (1 - self.cfg_scale) * u_uncond
            else:
                v_hat = v
        else:
            c_dropped = c
            v_hat = v

        # JVP
        jvp_args = (
            lambda z, t, r: model(z, t, r, y=c_dropped),  # warp model into f(z, t, r) -> u
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt
        
        error = u - sg(u_tgt)

        loss = adaptive_l2_loss(error)
        mse = sg(error).pow(2).mean()

        return loss, mse

    @torch.no_grad()
    def sample(self, model, labels: Union[np.ndarray, torch.Tensor], steps: int=1, device: str='cuda') -> torch.Tensor:
        model.eval()

        c = torch.as_tensor(labels, device=device)
        z = torch.randn(c.shape[0], self.channels, self.image_size, self.image_size, device=device)

        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t = torch.full((z.size(0),), t_steps[i], device=device)
            r = torch.full((z.size(0),), t_steps[i+1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
            r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

            v = model(z, t, r, c)
            z = z - (t_ - r_) * v

        return self.normalizer.reverse(z)

    def sample_each_class(self, model, classes: np.ndarray=None, n_per_class: int=1, steps: int=1, device: str='cuda') -> torch.Tensor:
        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        return self.sample(model, c, steps, device)