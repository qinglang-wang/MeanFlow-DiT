import argparse
import os
import torch
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from models.dit import MFDiT
from core.scheduler import SamplerScheduler
from core.engine import MeanFlowEngine
from core.normalizer import Normalizer
from trainer import Trainer
from utils.utils import load_yaml, seed_everything

def main():
    parser = argparse.ArgumentParser(description="Train MeanFlow on MNIST")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    args = parser.parse_args()
    config = load_yaml(args.config)

    seed_everything(42)

    if config['exp_name'] is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config['exp_name'] = f"mnist_{current_time}"

    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'], mixed_precision='fp16')
    if accelerator.is_main_process:
        print(f"Starting Experiment: {config['exp_name']}")
        print(f"Loading sampler config from: {config['phase_config']}")

    # Prepare data
    transform = transforms.Compose(
        [
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(), 
        ]
    )
    
    dataset = datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True
    )

    # Prepare model
    model = MFDiT(
        input_size=config['image_size'],
        in_channels=config['channels'],
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10
    )

    # Init SamplerScheduler
    phase_config = load_yaml(config['phase_config'])
    scheduler = SamplerScheduler(total_iterations=config['n_steps'], phase_config=phase_config)

    # Init Normalizer
    normalizer = Normalizer(mode='image')

    # Init Engine
    engine = MeanFlowEngine(
        scheduler=scheduler,
        normalizer=normalizer,
        num_classes=10,
        cfg_ratio=config['cfg_ratio'],
        cfg_scale=config['cfg_scale'],
        image_size=config['image_size'],
        channels=config['channels'],
        jvp_api=config['jvp_api']
    )

    # Init Trainer
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        engine=engine,
        config=config,
        accelerator=accelerator
    )
    
    trainer.train()

if __name__ == "__main__":
    main()