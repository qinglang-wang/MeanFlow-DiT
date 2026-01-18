import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from typing import Any


class InfiniteLoader:
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            return next(self._iterator)

    def __len__(self):
        return len(self._dataloader)


def load_yaml(path: str) -> Any:
    try:
        with open(path, 'r', encoding="utf-8") as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.YAMLError as e: 
        raise ValueError(rf"Invalid YAML format of file: '{path}'. Error: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")

def dump_yaml(data: Any, path: str) -> None:
    with open(path, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)

def seed_everything(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def sg(x: torch.Tensor) -> torch.Tensor:
    return x.detach()