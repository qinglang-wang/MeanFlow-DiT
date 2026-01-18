import os
import time
import torch
import torch.optim as optim
from accelerate import Accelerator
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from core.engine import MeanFlowEngine
from models.dit import MFDiT
from utils.utils import InfiniteLoader
import logging


class Trainer:
    def __init__(self, model: MFDiT, dataloader, engine: MeanFlowEngine, config, accelerator: Accelerator):
        self.model = model
        self.dataloader = dataloader
        self.engine = engine
        self.config = config
        self.accelerator = accelerator
        
        self.optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.0)
        
        # Prepare via Accelerator
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        self.infinite_loader = InfiniteLoader(self.dataloader)
        
        self.global_step = 0
        self.result_dir = os.path.join("results", config['exp_name'])
        os.makedirs(os.path.join(self.result_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "ckpt"), exist_ok=True)
        self.log_file = os.path.join(self.result_dir, "log.txt")

    def train(self):
        total_steps = self.config['n_steps']
        progress_bar = tqdm(range(total_steps), disable=not self.accelerator.is_local_main_process)
        running_loss, running_mse = 0.0, 0.0
        
        self.model.train()
        while self.global_step < total_steps:
            data = next(self.infinite_loader)
            with self.accelerator.accumulate(self.model):
                x = data[0].to(self.accelerator.device)
                c = data[1].to(self.accelerator.device)
                  
                loss, mse = self.engine.forward_loss(self.model, x, c, self.global_step)
                
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                running_mse += mse.item()

            if self.accelerator.sync_gradients:
                self.global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), mse=mse.item())
                
                if self.global_step % self.config['log_interval'] == 0:
                    self.logging(running_loss / self.config['log_interval'], running_mse / self.config['log_interval'])
                
                if self.global_step % self.config['save_interval'] == 0:
                    self.save_checkpoint()
                    self.evaluate()

    def logging(self, loss, mse):
        if self.accelerator.is_main_process:
            current_time = time.asctime(time.localtime(time.time()))

            batch_info = f'Global Step: {self.global_step}'
            loss_info = f'Loss: {loss:.6f}    MSE_Loss: {mse:.6f}'
            lr_info = f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"

            log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}\n'
            
            with open(self.log_file, mode='a') as f:
                f.write(log_message)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            path = os.path.join(self.result_dir, "ckpt", f"step_{self.global_step}.pt")
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), path)

    def evaluate(self):
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            for steps in self.config['sample_steps']:
                samples = self.engine.sample_each_class(
                    unwrapped_model, 
                    n_per_class=1, 
                    steps=steps, 
                    device=self.accelerator.device
                )

                save_path = os.path.join(self.result_dir, "images", f"{steps}-step_{self.global_step}-iter.png")
                save_image(make_grid(samples, nrow=10), save_path)