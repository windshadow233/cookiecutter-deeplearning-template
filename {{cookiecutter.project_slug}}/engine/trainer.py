import logging
import os
import tqdm
import yaml
from omegaconf import OmegaConf
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from utils.logger import get_logger
from utils.config import load_cfg, get_config_value
from utils.misc import save_checkpoint, load_checkpoint
from models.model import Model


class Trainer:
    __OPTIMIZERS__ = {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adamax": optim.Adamax,
        "asgd": optim.ASGD,
        "adadelta": optim.Adadelta,
    }
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
        "plateau": lr_scheduler.ReduceLROnPlateau
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
                 train_dataset,
                 valid_dataset,
                 cfg,
                 exp_dir,
                 data_collate_fn=None):
        self.total_cfg = cfg
        self.exp_dir = exp_dir
        self._load_config()
        self._save_config()
        self.logger = get_logger(self.total_cfg, exp_dir)
        self.train_cfg = get_config_value(self.total_cfg, 'train', {})
        self.saving_cfg = get_config_value(self.train_cfg, 'save', {})

        self._setup_device()

        self.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        self.model_dir = os.path.join(self.checkpoint_dir, 'models')
        self.train_state_dir = os.path.join(self.checkpoint_dir, 'train_states')

        self.metrics = {}
        self.epochs = self.train_cfg['epochs']
        self.current_epoch = 0
        self.global_steps = 0

        self.save_last = get_config_value(self.saving_cfg, 'last.enable', True)
        self.save_last_freq = get_config_value(self.saving_cfg, 'last.freq', 1)
        self.save_best = get_config_value(self.saving_cfg, 'best.enable', True)
        self.save_freq = get_config_value(self.saving_cfg, 'best.freq', 1)
        self.save_best_metric = get_config_value(self.saving_cfg, 'best.metric', 'val_loss')
        self.best_mode = get_config_value(self.saving_cfg, 'best.mode', 'min')
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

        self.clip_grad = get_config_value(self.train_cfg, 'clip_grad', None)
        batch_size = get_config_value(self.train_cfg, 'batch_size', 32)
        num_workers = get_config_value(self.train_cfg, 'num_workers', 0)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=data_collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                      collate_fn=data_collate_fn)

        self.scheduler_name, self.scheduler_update, self.scheduler = self.build_scheduler(self.optimizer)

        resume_ckpt = get_config_value(self.train_cfg, 'resume_ckpt', 'last.pt')
        self._load_all(resume_ckpt)

        self.accelerator_cfg = get_config_value(self.train_cfg, 'accelerator', {})
        self.accelerator = Accelerator(
            device_placement=True,
            **self.accelerator_cfg
        )

        self.train_dataloader, self.valid_dataloader, self.model, self.optimizer = \
            self.accelerator.prepare(
                train_dataloader,
                valid_dataloader,
                self.model,
                self.optimizer
            )

    def _setup_device(self):
        device_cfg = get_config_value(self.train_cfg, 'device', {})
        device_type = get_config_value(device_cfg, 'type', 'auto')
        device_ids = get_config_value(device_cfg, 'ids', [])
        if device_type == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        elif device_type == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in device_ids)
        elif device_type == "auto":
            pass

    def _save_config(self):
        import yaml
        if not os.path.exists(cfg_path := os.path.join(self.exp_dir, 'config.yaml')):
            with open(cfg_path, 'w') as f:
                yaml.dump(OmegaConf.to_container(self.total_cfg), f, sort_keys=False)
            logging.info(f"Configuration saved to {cfg_path}")

    def _load_config(self):
        if os.path.exists(cfg_path := os.path.join(self.exp_dir, 'config.yaml')):
            self.total_cfg.update(load_cfg(cfg_path))
            logging.info(f"Configuration loaded from {os.path.join(self.exp_dir, 'config.yaml')}")
        cfg = yaml.dump(OmegaConf.to_container(self.total_cfg), sort_keys=False, default_flow_style=False)
        logging.debug(f"\nConfiguration:\n{cfg}")

    def _save_model(self, ckpt_name='last.pt'):
        ckpt = os.path.join(self.model_dir, ckpt_name)
        model = self.accelerator.unwrap_model(self.model)
        model.save(ckpt)

    def _save_train_state(self, ckpt_name='last.pt'):
        ckpt = os.path.join(self.train_state_dir, ckpt_name)
        state_dicts = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'current_epoch': self.current_epoch,
            'steps': self.global_steps,
            'metrics': self.metrics,
            'best_metric': self.best_metric,
            'logger': self.logger.state_dict()
        }
        save_checkpoint(state_dicts, ckpt)

    def _save_all(self, name):
        self._save_model(name)
        self._save_train_state(name)

    def _load_model(self, ckpt_name='last.pt'):
        if os.path.exists(ckpt := os.path.join(self.model_dir, ckpt_name)):
            self.logger.info(f"Resuming from last checkpoint: {ckpt}")
            self.model.load(ckpt)
        else:
            if os.path.isdir(self.model_dir):
                self.logger.warning(f"Checkpoint '{ckpt_name}' not found.")
            else:
                os.makedirs(self.model_dir)

    def _load_train_state(self, ckpt_name='last.pt'):
        if os.path.exists(ckpt := os.path.join(self.train_state_dir, ckpt_name)):
            self.logger.info(f"Resuming training state from: {ckpt}")
            state_dicts = load_checkpoint(ckpt)
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            if self.scheduler is not None and state_dicts['scheduler'] is not None:
                self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.current_epoch = state_dicts.get('current_epoch', 0)
            self.global_steps = state_dicts.get('steps', 0)
            self.metrics = state_dicts.get('metrics', {})
            self.best_metric = state_dicts.get('best_metric', None)
            self.logger.load_state_dict(state_dicts['logger'])
        else:
            if os.path.isdir(self.train_state_dir):
                self.logger.warning(f"Training state '{ckpt_name}' not found.")
            else:
                os.makedirs(self.train_state_dir)

    def _load_all(self, name):
        self._load_model(name)
        self._load_train_state(name)

    def build_optimizer(self, params):
        optimizer_cfg = get_config_value(self.train_cfg, 'optimizer', {})
        optimizer_type = get_config_value(optimizer_cfg, 'type', 'adam')
        learning_rate = get_config_value(optimizer_cfg, 'lr', 1e-5)
        weight_decay = get_config_value(optimizer_cfg, 'weight_decay', 0.0)
        optim_params = get_config_value(optimizer_cfg, 'params', {})
        optim_params = OmegaConf.to_container(optim_params)

        optimizer = self.__OPTIMIZERS__.get(optimizer_type.lower())
        if optimizer is None:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return optimizer(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **optim_params
        )

    def build_scheduler(self, optimizer):
        scheduler_cfg = get_config_value(self.train_cfg, 'scheduler', {})
        name = get_config_value(scheduler_cfg, 'name', 'none').lower()
        if name == 'none':
            return '', '', None
        params = get_config_value(scheduler_cfg, 'params', {})
        params = OmegaConf.to_container(params)
        if name in self.__EPOCH_BASED_SCHEDULERS__:
            scheduler_update = 'epoch'
        elif name in self.__STEP_BASED_SCHEDULERS__:
            scheduler_update = 'step'
        elif name in self.__METRIC_BASED_SCHEDULERS__:
            scheduler_update = 'metric'
            self.__setattr__('scheduler_metric', params.pop('metric', 'val_loss'))
        else:
            scheduler_update = ''
        try:
            if name in self.__SCHEDULERS__:
                scheduler_fcn = self.__SCHEDULERS__[name]
                if scheduler_fcn is None:
                    return '', '', None
                return name, scheduler_update, scheduler_fcn(optimizer, **params)
            else:
                self.logger.warning(f"Unknown scheduler: {name}, will not use a scheduler.")
                return '', '', None
        except Exception as e:
            self.logger.exception(f"Error creating scheduler: {e}, will not use a scheduler.")
            return '', '', None

    def scheduler_step(self, mode="step"):
        if self.scheduler_update != mode or not self.scheduler:
            return
        if mode != "metric":
            self.scheduler.step()
            return
        key = self.__getattribute__('scheduler_metric')
        assert key in self.metrics, f"Scheduler metric '{key}' not found in validation metrics"
        self.scheduler.step(self.metrics[key])

    def _train_step(self, data):
        self.global_steps += 1
        log_items = self.train_step(self.model, data, self.optimizer, self.accelerator)
        self.scheduler_step('step')
        if log_items:
            self.logger.log(log_items, step=self.global_steps)

    def train_step(self, model, data, optimizer, accelerator):
        """
        单次训练步骤，支持重写
        :param model: 模型实例
        :param data: 输入数据
        :param optimizer: 优化器
        :param accelerator: 加速器实例
        :return: 返回一个包含各种自定义日志记录指标的字典（也可什么都不返回）
        """
        assert isinstance(data, dict), "Data must be a dict"
        out = model(**data)
        assert isinstance(out, dict) and 'loss' in out, "Model output must be a dict with a 'loss' key"
        loss = out['loss']
        accelerator.backward(loss)
        if self.clip_grad is not None:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=self.clip_grad)
        optimizer.step()
        optimizer.zero_grad()

        log_items = {
            "loss": loss.item()
        }
        if self.scheduler is not None:
            log_items['lr'] = self.optimizer.param_groups[0]['lr']
        return log_items

    def _save(self, epoch):
        save_last = self.save_last and (epoch + 1) % self.save_last_freq == 0
        save_best = self.save_best
        save_current = (epoch + 1) % self.save_freq == 0
        if save_last or save_current or self.scheduler_update == "metric":
            self.metrics = self._evaluate()
            for key, value in self.metrics.items():
                self.logger.log({key: value}, step=self.global_steps)
            self.scheduler_step("metric")
            assert isinstance(self.metrics, dict), "Validation metrics must be a dict"
            if save_best:
                if self.best_mode == 'min':
                    if (new_best := self.metrics[self.save_best_metric]) < self.best_metric:
                        self.best_metric = new_best
                        self._save_all('best.pt')
                        self.logger.info(f"New best metric ({self.save_best_metric}): {new_best:.4f}")
                else:
                    if (new_best := self.metrics[self.save_best_metric]) > self.best_metric:
                        self.best_metric = new_best
                        self._save_all('best.pt')
                        self.logger.info(f"New best metric ({self.save_best_metric}): {new_best:.4f}")
            if save_last:
                self._save_all('last.pt')
            if save_current:
                self._save_all(f'ckpt_epoch_{epoch + 1}.pt')

    def run(self):
        self.logger.info("Training started.")
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch + 1
            self.model.train()
            for data in tqdm.tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}/{self.epochs}"):
                self._train_step(data)
            self._save(epoch)

            self.scheduler_step('epoch')

        self.logger.info("Training completed.")

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        return self.evaluate(self.model, self.valid_dataloader)

    def evaluate(self, model: Model, dataloader: DataLoader):
        raise NotImplementedError("evaluate must be implemented in subclasses")
