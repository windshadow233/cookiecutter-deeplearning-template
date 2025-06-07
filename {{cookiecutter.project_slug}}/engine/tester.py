import torch
import logging
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from model.models import Model
from utils.config import load_cfg, get_config_value


class Tester:
    def __init__(self,
                 model: Model,
                 ckpt_name,
                 dataset,
                 exp_dir,
                 data_collate_fn=None):
        self.model = model.eval()
        self.ckpt_name = ckpt_name
        self.exp_dir = exp_dir

        self.cfg = self._load_config()

        batch_size = get_config_value(self.cfg, 'train.batch_size', 32)
        num_workers = get_config_value(self.cfg, 'train.num_workers', 0)

        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=data_collate_fn)

        accelerator_cfg = get_config_value(self.cfg, 'train.accelerator', {})

        self._load_checkpoint(ckpt_name)
        self.accelerator = Accelerator(
            device_placement=True,
            **OmegaConf.to_container(accelerator_cfg)
        )

        self.dataloader, self.model = self.accelerator.prepare(self.dataloader, self.model)

    def _load_config(self):
        if os.path.exists(cfg_path := os.path.join(self.exp_dir, 'config.yaml')):
            cfg = load_cfg(cfg_path)
            logging.info(f"Configuration loaded from {os.path.join(self.exp_dir, 'config.yaml')}")
            return cfg
        else:
            logging.warning(f"No configuration file found at {cfg_path}, using empty configuration.")
            return OmegaConf.create({})

    def _load_checkpoint(self, ckpt_name: str):
        if not os.path.exists(ckpt_path := os.path.join(self.exp_dir, 'checkpoints', 'model', ckpt_name)):
            logging.warning("No checkpoint found, skipping loading.")
            return
        self.model.load(ckpt_path)
        logging.info(f"Checkpoint loaded from {ckpt_path}")

    @torch.no_grad()
    def run(self):
        metrics = self.evaluate(self.model, self.dataloader)
        logging.info(f"Test metrics: {metrics}")

    def evaluate(self, model: Model, dataloader: DataLoader):
        raise NotImplementedError(
            "evaluate method should be implemented in the subclass of Tester"
        )
