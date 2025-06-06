import torch
import logging
import os
from accelerate import Accelerator
from omegaconf import OmegaConf
from models.model import Model
from utils.config import load_cfg, get_value_from_cfg


class Tester:
    def __init__(self, model: Model, ckpt_name, dataloader, exp_dir):
        self.model = model.eval()
        self.ckpt_name = ckpt_name
        self.dataloader = dataloader
        self.exp_dir = exp_dir

        self.cfg = self._load_config()

        accelerator_cfg = get_value_from_cfg(self.cfg, 'train.accelerator', {})

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
        if not os.path.isfile(ckpt_path := os.path.join(self.exp_dir, 'checkpoints', 'models', ckpt_name)):
            logging.warning("No checkpoint found, skipping loading.")
            return
        self.model.load(ckpt_path)

    @torch.no_grad()
    def run(self):
        metrics = self.test_fn(self.model, self.dataloader)
        logging.info(f"Test metrics: {metrics}")

    @torch.no_grad()
    def test_fn(self, model, dataloader):
        raise NotImplementedError(
            "test_fn method should be implemented in the subclass of Tester"
        )
