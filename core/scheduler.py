import torch
from typing import Dict, Tuple


class SamplerScheduler:
    def __init__(self, total_iterations: int, phase_config: Dict):
        self.total_iterations = max(int(total_iterations), 1)
        self.device = None

        # Register initial phases
        self.phases = []
        for name, cfg in phase_config.items():
            self._register_phase(name, cfg)

    def _register_phase(self, name: str, phase_config: Dict):
        """
        Register a phase.

        phase_config:
          - "interval": [start, end]
          - "t": {"method": str, "config": dict}
          - "r": {"method": str, "config": dict}
          - "instant_prob": float (optional): Probability to force r = t
          - "resample": bool (optional): If instant_prob specified, resample will determine wherther use a standard lognorm to resample the instant t and r
        """ 
        start, end = phase_config["interval"]

        assert 0.0 <= start < end <= 1.0, "Invalid interval"
        assert all(end <= phase['start'] or phase['end'] <= start for phase in self.phases), "Interval overlapped"

        self.phases.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "t_sampler": getattr(self, f"_construct_{phase_config['t']['method']}")(**phase_config['t']['config']),
                "r_sampler": getattr(self, f"_construct_{phase_config['r']['method']}")(**phase_config['r']['config']),
                "instant_prob": phase_config.get("instant_prob", 0.0),
                "resample": phase_config.get("resample", False),
            }
        )

        self.phases.sort(key=lambda s: s["start"])

    def _current_stage(self, iteration: int) -> Dict:
        """
        Get current stage.
        """
        p = max(min(iteration, self.total_iterations - 1), 0) / (self.total_iterations - 1)
        for s in self.phases:
            if s["start"] <= p < s["end"]:
                return s
        return self.phases[-1]

    def sample(self, batch_size: int, iteration: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (r, t) from the corresponding stage according to iteration.
        """
        self.device = device
        stage = self._current_stage(iteration)

        t = stage["t_sampler"](batch_size)
        r = stage["r_sampler"](batch_size)

        return self._postprocess(r, t, stage)

    def _postprocess(self, r: torch.Tensor, t: torch.Tensor, stage: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Guarantee r <= t; Optional: Guarantee |t - r| >= min_delta or partially set r == t.
        """
        # Apply strict equality for a percentage of the batch
        if instant_prob := stage["instant_prob"]:
            n = int(t.shape[0] * instant_prob)
            instant_mask = torch.randperm(t.shape[0], device=self.device)[:n]
            if stage["resample"]:
                r[instant_mask] = t[instant_mask] = self._construct_lognorm(mu=-0.4, sigma=1)(n)
            else:
                r[instant_mask] = t[instant_mask]

        # Guarantee r <= t
        if (swap_mask := r > t).any():
            r[swap_mask], t[swap_mask] = t[swap_mask], r[swap_mask]

        return r, t

    def _construct_uniform(self, **kwargs) -> callable:
        """
        Uniform distribution.
        """
        return lambda x: torch.rand(x, device=self.device)
    
    def _construct_lognorm(self, **kwargs) -> callable:
        """
        Lognorm distribution.
        """
        return lambda x: torch.sigmoid(torch.randn(x, device=self.device) * kwargs['sigma'] + kwargs['mu'])
