import os
import torch
import torchvision
import torchvision.transforms as T
from torchdiffeq import odeint
from tqdm import tqdm
import math
import numpy as np
from typing import Optional, List

# Import your model
from models.dit import MFDiT
# Import utils to load config if available
from utils.utils import load_yaml

def div_fn(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence of u with respect to x using Hutchinson's Trace Estimator.
    """
    # epsilon ~ Rademacher distribution (random +1 or -1)
    epsilon = torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1
    
    # Compute vector-Jacobian product: epsilon^T * J
    vjp = torch.autograd.grad(u, x, grad_outputs=epsilon, create_graph=False)[0]
    
    # Compute trace approximation: epsilon^T * vjp
    div = (vjp * epsilon).sum(dim=[1, 2, 3])
    return div

class Evaluator:
    def __init__(self, exp_name: str, device: str = 'cuda'):
        self.exp_name = exp_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.exp_dir = os.path.join("results", exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "ckpt") # Or "checkpoints" based on your trainer
        if not os.path.exists(self.ckpt_dir):
             self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
             
        self.results_file = os.path.join(self.exp_dir, "evaluation_results.txt")
        
        # Try to load config from experiment directory to get model params
        # Assuming you saved config.yaml in result dir, otherwise use defaults
        self.model_config = {
            "input_size": 28, "patch_size": 2, "in_channels": 1,
            "dim": 384, "depth": 12, "num_heads": 6, "num_classes": 10
        }
        
        # Initialize Model
        self.model = MFDiT(**self.model_config).to(self.device)
        
        # Initialize Data
        self.transform = T.Compose([
            T.Resize((self.model_config['input_size'], self.model_config['input_size'])),
            T.ToTensor(),
        ])
        self.dataset = torchvision.datasets.MNIST(root="mnist", train=False, download=True, transform=self.transform)
        
    def get_available_checkpoints(self) -> List[int]:
        if not os.path.exists(self.ckpt_dir):
            return []
        files = [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pt") and "step_" in f]
        steps = sorted([int(f.split("_")[1].split(".")[0]) for f in files])
        return steps

    def run_evaluation(self, 
                       ckpt_step: Optional[int] = None, 
                       evaluate_all: bool = False, 
                       batch_size: int = 100, 
                       limit_batches: Optional[int] = None):
        """
        Main method to run evaluation logic.
        """
        all_steps = self.get_available_checkpoints()
        if not all_steps:
            print(f"No checkpoints found in {self.ckpt_dir}")
            return

        if evaluate_all:
            steps_to_eval = all_steps
        elif ckpt_step is not None:
            if ckpt_step not in all_steps:
                print(f"Checkpoint step {ckpt_step} not found.")
                return
            steps_to_eval = [ckpt_step]
        else:
            # Default to latest
            steps_to_eval = [all_steps[-1]]

        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        D = None # Dimension

        for step in steps_to_eval:
            # Check if already evaluated (simple check)
            if self._is_step_evaluated(step):
                print(f"Step {step} already evaluated. Skipping (Delete line in txt to re-run).")
                continue

            ckpt_path = os.path.join(self.ckpt_dir, f"step_{step}.pt")
            print(f"\nEvaluating Checkpoint: Iteration {step}.")
            
            try:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                print(f"Failed to load checkpoint {ckpt_path}: {e}")
                continue

            avg_nll, avg_bpd, total_samples = self._evaluate_single_epoch(dataloader, limit_batches)
            
            if total_samples > 0:
                self._log_results(step, total_samples, avg_nll, avg_bpd)
                print(f"Step {step}: NLL={avg_nll:.4f}, BPD={avg_bpd:.4f}")
            
            torch.cuda.empty_cache()

    def _evaluate_single_epoch(self, dataloader, limit_batches):
        total_nll = 0.0
        total_bpd = 0.0
        total_samples = 0
        D = None
        
        def log_normal_standard(z):
            return -0.5 * (z ** 2).sum(dim=[1,2,3]) - 0.5 * D * math.log(2 * math.pi)

        # ODE function closure
        def ode_func(t, state):
            x_t = state[0]
            with torch.set_grad_enabled(True):
                x_t.requires_grad_(True)
                t_tensor = t.repeat(x_t.shape[0])
                r_tensor = t_tensor.clone()
                v = self.model(x_t, t_tensor, r_tensor, y=c_batch)
                div = div_fn(v, x_t)
                return v, div

        with torch.no_grad():
            total_batches = limit_batches if limit_batches is not None else len(dataloader)
            for i, (x, y) in enumerate(tqdm(dataloader, total=total_batches, desc=" ODE Integration")):
                if limit_batches is not None and i >= limit_batches:
                    break
                
                x = x.to(self.device)
                if D is None: D = x[0].numel()
                c_batch = y.to(self.device)
                batch_size_curr = x.shape[0]

                # Dequantization & Normalization
                x_disc = (x * 255.0).floor()
                u = torch.rand_like(x_disc)
                x_deq = (x_disc + u) / 256.0 
                x_norm = x_deq * 2 - 1
                
                # Integration
                log_det_0 = torch.zeros(batch_size_curr, device=self.device)
                try:
                    out = odeint(ode_func, (x_norm, log_det_0), torch.tensor([0.0, 1.0], device=self.device), method='rk4', options={'step_size': 0.05})
                except Exception as e:
                    print(f"Integration error: {e}")
                    continue

                z_1 = out[0][-1]
                delta_log_det = out[1][-1]
                
                # Metrics Calculation
                log_pz = log_normal_standard(z_1)
                log_px_norm = log_pz + delta_log_det
                log_px_deq = log_px_norm + D * math.log(2.0)
                deq_const = D * math.log(256.0)
                log_px_disc = log_px_deq - deq_const
                
                nll = -log_px_disc.mean().item()
                bpd = nll / (D * math.log(2.0))
                
                total_nll += nll * batch_size_curr
                total_bpd += bpd * batch_size_curr
                total_samples += batch_size_curr

        avg_nll = total_nll / total_samples if total_samples > 0 else 0.0
        avg_bpd = total_bpd / total_samples if total_samples > 0 else 0.0
        return avg_nll, avg_bpd, total_samples

    def _log_results(self, step, samples, nll, bpd):
        with open(self.results_file, "a") as f:
            f.write(f"Step {step}:\n")
            f.write(f"  Num Samples: {samples}\n")
            f.write(f"  NLL: {nll:.4f}\n")
            f.write(f"  BPD: {bpd:.4f}\n\n")

    def _is_step_evaluated(self, step):
        if not os.path.exists(self.results_file):
            return False
        with open(self.results_file, 'r') as f:
            content = f.read()
        return f"Step {step}:" in content