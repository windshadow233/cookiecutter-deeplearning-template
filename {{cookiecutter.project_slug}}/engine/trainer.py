import copy
import logging

import torch
import os

import tqdm
import yaml
from accelerate import Accelerator
from torch.utils.data import random_split, DataLoader
from torch.optim import lr_scheduler
from utils.logger import get_logger
from utils.config import load_cfg
from utils.other_utils import save_checkpoint, load_checkpoint
from models.model import Model


class Trainer:
    __SCHEDULERS__ = {
        "step": lr_scheduler.StepLR,
        "multistep": lr_scheduler.MultiStepLR,
        "constant": lr_scheduler.ConstantLR,
        "linear": lr_scheduler.LinearLR,
        "exp": lr_scheduler.ExponentialLR,
        "poly": lr_scheduler.PolynomialLR,
        "cosine": lr_scheduler.CosineAnnealingLR,
        "cosine_restart": lr_scheduler.CosineAnnealingWarmRestarts,
        "cyclic": lr_scheduler.CyclicLR,
        "onecycle": lr_scheduler.OneCycleLR,
        "plateau": lr_scheduler.ReduceLROnPlateau,
        "none": None
    }
    __EPOCH_BASED_SCHEDULERS__ = {
        "step",
        "multistep",
        "exp",
        "cosine",
        "cosine_restart",
        "poly",
    }
    __STEP_BASED_SCHEDULERS__ = {
        "constant",
        "linear",
        "cyclic",
        "onecycle",
    }
    __METRIC_BASED_SCHEDULERS__ = {
        "plateau",
    }

    def __init__(self,
                 model: Model,
                 dataset,
                 cfg,
                 exp_dir):
        self.total_cfg = cfg
        self.exp_dir = exp_dir
        self._load_config()
        self._save_config()
        self.logger = get_logger(self.total_cfg, exp_dir)
        self.train_cfg = self.total_cfg.get('train', {})
        self.saving_cfg = self.train_cfg.get('save', {})

        self._setup_device()

        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')

        self.metrics = {}
        self.epochs = self.train_cfg['epochs']
        self.current_epoch = 0
        self.global_steps = 0
        self.save_last = self.saving_cfg.get('save_last', True)
        self.save_last_freq = self.saving_cfg.get('save_last_freq', 1)
        self.save_best = self.saving_cfg.get('save_best', True)
        self.save_freq = self.saving_cfg.get('save_freq', 1)
        self.save_best_metric = self.saving_cfg.get('save_best_metric', 'val_loss')
        self.best_mode = self.saving_cfg.get('best_mode', 'min')
        self.best_metric = None
        if self.save_best:
            if self.best_mode == 'min':
                self.best_metric = float('inf')
            elif self.best_mode == 'max':
                self.best_metric = 0.0
            else:
                raise ValueError(f"Unsupported best_mode: {self.best_mode}. Use 'min' or 'max'.")

        self.model = model
        self.optimizer = self.build_optimizer(model.get_all_params())

        self.clip_grad = self.train_cfg.get('clip_grad', None)
        data_split = self.train_cfg.get('data_split', [0.8, 0.1, 0.1])
        batch_size = self.train_cfg.get('batch_size', 32)
        num_workers = self.train_cfg.get('num_workers', 0)
        train_len = int(len(dataset) * data_split[0])
        valid_len = int(len(dataset) * data_split[1])
        test_len = len(dataset) - train_len - valid_len

        train, valid, test = random_split(dataset, [train_len, valid_len, test_len])

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=self.data_collate_fn)
        valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=self.data_collate_fn)
        test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                     collate_fn=self.data_collate_fn)

        self.accelerator_cfg = self.train_cfg.get('accelerator', {})
        self.accelerator = Accelerator(
            device_placement=True,
            **self.accelerator_cfg
        )

        self.train_dataloader, self.valid_dataloader, self.test_dataloader, self.model, self.optimizer = \
            self.accelerator.prepare(
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                self.model,
                self.optimizer
            )

        self.scheduler_name, self.scheduler_update, self.scheduler = self.build_scheduler(self.optimizer)

        self._load_checkpoint()

    def _setup_device(self):
        device_cfg = self.train_cfg.get('device', {})
        device_type = device_cfg.get('type', 'auto')
        device_ids = device_cfg.get('ids', [])
        if device_type == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif device_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
        elif device_type == "auto":
            pass

    def _save_config(self):
        import yaml
        cfg = copy.deepcopy(self.total_cfg)
        if not os.path.exists(cfg_path := os.path.join(self.exp_dir, 'config.yaml')):
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)
            logging.info(f"Configuration saved to {cfg_path}")

    def _load_config(self):
        if os.path.exists(cfg_path := os.path.join(self.exp_dir, 'config.yaml')):
            self.total_cfg.update(load_cfg(cfg_path))
            logging.info(f"Configuration loaded from {os.path.join(self.exp_dir, 'config.yaml')}")
        self.total_cfg['base_dir'] = str(self.total_cfg.get('base_dir', ''))
        cfg = yaml.dump(self.total_cfg, sort_keys=False, default_flow_style=False)
        logging.info(f"\nConfiguration:\n{cfg}")

    def _save_checkpoint(self, ckpt_name='last.pt'):
        ckpt = os.path.join(self.checkpoint_dir, ckpt_name)
        save_checkpoint({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'steps': self.global_steps,
            'metrics': self.metrics,
            'best_metric': self.best_metric,
            'logger': self.logger.state_dict()
        }, ckpt)

    def _load_checkpoint(self, ckpt_name='last.pt'):
        if os.path.exists(last := os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.info(f"Resuming from last checkpoint: {last}")
            ckpt = load_checkpoint(last)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.scheduler is not None:
                if ckpt['scheduler'] is not None:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
            self.current_epoch = ckpt['current_epoch']
            self.global_steps = ckpt['steps']
            self.metrics = ckpt['metrics']
            self.best_metric = ckpt['best_metric']
            self.logger.load_state_dict(ckpt['logger'])
        else:
            if os.path.isdir(self.checkpoint_dir):
                self.logger.warning(f"Checkpoint '{ckpt_name}' not found.")
            else:
                os.makedirs(self.checkpoint_dir)

    def build_optimizer(self, params):
        optimizer_cfg = self.train_cfg.get('optimizer', {})
        optimizer_type = optimizer_cfg.get('type', 'adam')
        learning_rate = optimizer_cfg.get('lr', 1e-5)
        weight_decay = optimizer_cfg.get('weight_decay', 0.0)

        if optimizer_type == 'adam':
            beta1 = optimizer_cfg.get('beta1', 0.9)
            beta2 = optimizer_cfg.get('beta2', 0.999)
            return torch.optim.Adam(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(beta1, beta2)
            )
        elif optimizer_type == 'sgd':
            momentum = optimizer_cfg.get('momentum', 0.9)
            return torch.optim.SGD(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def build_scheduler(self, optimizer):
        scheduler_cfg = self.train_cfg.get('scheduler', {})
        params = scheduler_cfg.get("params", {})
        name = scheduler_cfg.get("name", "none").lower()
        if name in self.__EPOCH_BASED_SCHEDULERS__:
            scheduler_update = 'epoch'
        elif name in self.__STEP_BASED_SCHEDULERS__:
            scheduler_update = 'step'
        elif name in self.__METRIC_BASED_SCHEDULERS__:
            scheduler_update = 'metric'
            self.__setattr__('plateau_metric', params.pop('metric', 'val_loss'))
        else:
            scheduler_update = ''
        try:
            if name in self.__SCHEDULERS__:
                scheduler_fcn = self.__SCHEDULERS__[name]
                if scheduler_fcn is None:
                    return '', scheduler_update, None
                return name, scheduler_update, scheduler_fcn(optimizer, **params)
            else:
                self.logger.warning(f"Unknown scheduler: {name}, will not use a scheduler.")
                return '', scheduler_update, None
        except Exception as e:
            self.logger.exception(f"Error creating scheduler: {e}, will not use a scheduler.")
            return '', scheduler_update, None

    def data_collate_fn(self, batch):
        """
        Override this method in subclasses to implement custom collate function.

        Must return a dict.
        """
        data, label = zip(*batch)
        data = torch.stack(data, dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return {'x': data, 'y': label}

    def run(self):
        self.logger.info("Training started.")
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch + 1
            self.model.train()
            for data in tqdm.tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}/{self.epochs}"):
                self.global_steps += 1
                assert isinstance(data, dict), "Data must be a dict"
                out = self.model(**data)
                assert isinstance(out, dict) and 'loss' in out, "Model output must be a dict with a 'loss' key"
                loss = out['loss']
                log_items = {
                    "loss": loss.item()
                }
                if self.scheduler is not None:
                    log_items['lr'] = self.optimizer.param_groups[0]['lr']
                self.logger.log(log_items, step=self.global_steps)
                self.accelerator.backward(loss)
                if self.clip_grad is not None:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler and self.scheduler_update == "step":
                    self.scheduler.step()

            if self.scheduler and self.scheduler_update == "epoch":
                self.scheduler.step()
            save_last = self.save_last and (epoch + 1) % self.save_last_freq == 0
            save_best = self.save_best
            save_current = (epoch + 1) % self.save_freq == 0
            if save_last or save_current or self.scheduler_update == "metric":
                self.model.eval()
                self.metrics = self.validate_fn(self.valid_dataloader)
                for key, value in self.metrics.items():
                    self.logger.log({key: value}, step=self.global_steps)
                if self.scheduler and self.scheduler_update == "metric":
                    key = self.__getattribute__('plateau_metric')
                    assert key in self.metrics, f"Metric '{key}' not found in validation metrics"
                    self.scheduler.step(self.metrics[key])
                assert isinstance(self.metrics, dict), "Validation metrics must be a dict"
                if save_best:
                    if self.best_mode == 'min':
                        if (new_best := self.metrics[self.save_best_metric]) < self.best_metric:
                            self.best_metric = new_best
                            self._save_checkpoint('best.pt')
                            self.logger.info(f"New best metric ({self.save_best_metric}): {new_best:.4f}")
                    else:
                        if (new_best := self.metrics[self.save_best_metric]) > self.best_metric:
                            self.best_metric = new_best
                            self._save_checkpoint('best.pt')
                            self.logger.info(f"New best metric ({self.save_best_metric}): {new_best:.4f}")
                if save_last:
                    self._save_checkpoint('last.pt')
                if save_current:
                    self._save_checkpoint(f'ckpt_epoch_{epoch + 1}.pt')

        self.logger.info("Training completed.")

    @torch.no_grad()
    def validate_fn(self, dataloader):
        raise NotImplementedError("validate_fcn must be implemented in subclasses")

