import os
import torch
import torchvision
import torchvision.transforms as T
from torchdiffeq import odeint
from tqdm import tqdm
import math
import re
from typing import Optional, List, Dict

from models.dit import MFDiT

def div_fn(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute divergence of u with respect to x using Hutchinson's Trace Estimator."""
    epsilon = torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1
    vjp = torch.autograd.grad(u, x, grad_outputs=epsilon, create_graph=False)[0]
    div = (vjp * epsilon).sum(dim=[1, 2, 3])
    return div


class Evaluator:
    def __init__(self, exp_name: str, device: str = 'cuda'):
        self.exp_name = exp_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.exp_dir = os.path.join("results", exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, "ckpt")
        if not os.path.exists(self.ckpt_dir):
             self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
             
        self.results_file = os.path.join(self.exp_dir, "evaluation_results.txt")
        
        # Model Configuration (Hardcoded for now based on your project)
        self.model_config = {
            "input_size": 28, "patch_size": 2, "in_channels": 1,
            "dim": 384, "depth": 12, "num_heads": 6, "num_classes": 10
        }
        
        # Initialize Components
        self.model = MFDiT(**self.model_config).to(self.device)
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

    def get_existing_results(self) -> Dict[int, Dict]:
        """Parse the results file to find which steps are already fully evaluated."""
        results = {}
        if not os.path.exists(self.results_file):
            return results
            
        with open(self.results_file, 'r') as f:
            content = f.read()
            
        # Regex to find blocks like "Step 1000:... BPD: 1.234"
        # We assume if BPD is present, the step is done.
        # Pattern: Step (\d+):.*?BPD:\s+([0-9.]+)
        pattern = r"Step (\d+):[\s\S]*?BPD:\s+([0-9.]+)"
        matches = re.findall(pattern, content)
        
        for step_str, bpd_str in matches:
            results[int(step_str)] = float(bpd_str)
            
        return results

    def run_evaluation(self, 
                       ckpt_step: Optional[int] = None, 
                       evaluate_all: bool = False, 
                       batch_size: int = 100, 
                       limit_batches: Optional[int] = None,
                       force_rerun: bool = False):
        """
        Main evaluation loop with Resume capability.
        """
        all_steps = self.get_available_checkpoints()
        if not all_steps:
            print(f"No checkpoints found in {self.ckpt_dir}")
            return

        # 1. Determine steps to evaluate
        if evaluate_all:
            steps_to_eval = all_steps
        elif ckpt_step is not None:
            if ckpt_step not in all_steps:
                print(f"Checkpoint step {ckpt_step} not found.")
                return
            steps_to_eval = [ckpt_step]
        else:
            steps_to_eval = [all_steps[-1]] # Latest only

        # 2. Check what's already done
        existing_results = self.get_existing_results()
        
        # 3. Prepare Data
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
        # 4. Loop
        for step in steps_to_eval:
            # --- RESUME LOGIC ---
            if not force_rerun and step in existing_results:
                print(f"Step {step} already evaluated (BPD: {existing_results[step]:.4f}). Skipping.")
                continue
            # --------------------

            ckpt_path = os.path.join(self.ckpt_dir, f"step_{step}.pt")
            print(f"\nEvaluating Checkpoint: Step {step}...")
            
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
                print(f"Step {step} Finished: NLL={avg_nll:.4f}, BPD={avg_bpd:.4f}")
            
            torch.cuda.empty_cache()

    def _evaluate_single_epoch(self, dataloader, limit_batches):
        total_nll = 0.0
        total_bpd = 0.0
        total_samples = 0
        D = None
        
        def log_normal_standard(z):
            return -0.5 * (z ** 2).sum(dim=[1,2,3]) - 0.5 * D * math.log(2 * math.pi)

        def ode_func(t, state):
            x_t = state[0]
            with torch.set_grad_enabled(True):
                x_t.requires_grad_(True)
                t_tensor = t.repeat(x_t.shape[0])
                r_tensor = t_tensor.clone()
                # Assuming class conditional, using dummy labels if needed or real labels from batch
                # Here we use the global c_batch captured from loop
                v = self.model(x_t, t_tensor, r_tensor, y=c_batch)
                div = div_fn(v, x_t)
                return v, div

        with torch.no_grad():
            total_batches = limit_batches if limit_batches is not None else len(dataloader)
            # Use tqdm for progress bar
            pbar = tqdm(dataloader, total=total_batches, desc="  Integration")
            
            for i, (x, y) in enumerate(pbar):
                if limit_batches is not None and i >= limit_batches:
                    break
                
                x = x.to(self.device)
                if D is None: D = x[0].numel()
                c_batch = y.to(self.device) # Capture for ode_func
                batch_size_curr = x.shape[0]

                # Dequantization & Normalization
                x_disc = (x * 255.0).floor()
                u = torch.rand_like(x_disc)
                x_deq = (x_disc + u) / 256.0 
                x_norm = x_deq * 2 - 1
                
                # Integration (Using Euler for speed if needed, or RK4 for precision)
                # Recommended: 'rk4' with step_size 0.05 is standard, 'euler' is faster but less accurate
                log_det_0 = torch.zeros(batch_size_curr, device=self.device)
                try:
                    # Switch to 'euler' for speed during debugging? Or keep 'rk4'.
                    # Let's keep 'rk4' as it's the standard metric.
                    out = odeint(ode_func, (x_norm, log_det_0), torch.tensor([0.0, 1.0], device=self.device), 
                                 method='rk4', options={'step_size': 0.05})
                except Exception as e:
                    print(f"Integration error: {e}")
                    continue

                z_1 = out[0][-1]
                delta_log_det = out[1][-1]
                
                # Metrics
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
                
                # Update progress bar
                pbar.set_postfix({"BPD": f"{bpd:.3f}"})

        avg_nll = total_nll / total_samples if total_samples > 0 else 0.0
        avg_bpd = total_bpd / total_samples if total_samples > 0 else 0.0
        return avg_nll, avg_bpd, total_samples

    def _log_results(self, step, samples, nll, bpd):
        # Read all content first to avoid duplicates if partial write happened
        content = ""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                content = f.read()
        
        # If this step exists in file, we might want to replace it or just append?
        # Simple approach: Append. The parser reads the *last* occurrence usually or all.
        # But to be clean, let's just append.
        
        with open(self.results_file, "a") as f:
            f.write(f"Step {step}:\n")
            f.write(f"  Num Samples: {samples}\n")
            f.write(f"  NLL: {nll:.4f}\n")
            f.write(f"  BPD: {bpd:.4f}\n")
            f.write("-" * 20 + "\n")