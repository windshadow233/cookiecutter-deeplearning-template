import os.path

from torch.utils.tensorboard import SummaryWriter
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from utils.config import get_config_value


class ColorFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.DEBUG: "\033[94m",    # 蓝
        logging.INFO: "\033[92m",     # 绿
        logging.WARNING: "\033[93m",  # 黄
        logging.ERROR: "\033[91m",    # 红
        logging.CRITICAL: "\033[95m", # 紫
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def init_logger(cfg):
    level = get_config_value(cfg, "logger.level", "INFO").upper()
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = ColorFormatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Logger:
    def __init__(self, cfg):
        level_str = get_config_value(cfg, "logger.level", "INFO").upper()
        self.logger = logging.getLogger()
        self.logger.setLevel(getattr(logging, level_str, logging.INFO))

    def log(self, metrics: dict, step: int):
        ...

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def exception(self, message):
        self.logger.exception(message)

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        ...


class TensorboardLogger(Logger):
    def __init__(self, cfg, tb_logger):
        super().__init__(cfg)
        self.tb = tb_logger

    def log(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, global_step=step)


class SimpleLogger(Logger):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logs = defaultdict(list)

    def log(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.logs[k].append((step, v))

    def summary(self):
        summary = {}
        for k, v in self.logs.items():
            if v:
                summary[k] = sum(val for _, val in v) / len(v)
        return summary

    def show(self):
        for k, v in self.logs.items():
            self.info(f"{k}: {sum(val for _, val in v) / len(v) if v else 0:.4f} (count: {len(v)})")

    def plot(self, save_path=None):
        if not self.logs:
            self.info("No data to plot.")
            return
        num_metrics = len(self.logs)
        ncols = math.ceil(math.sqrt(num_metrics))
        nrows = math.ceil(num_metrics / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for idx, (k, v) in enumerate(self.logs.items()):
            steps, values = zip(*v) if v else ([], [])
            axes[idx].plot(steps, values, label=k)
            axes[idx].set_title(k)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("Value")
            axes[idx].xaxis.set_major_locator(MaxNLocator(integer=True))
            axes[idx].xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True, prune='both'))
        for j in range(len(self.logs), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            self.info(f"Plot saved to {save_path}")
        else:
            plt.show()

    def state_dict(self):
        state = {"logs": dict(self.logs)}
        return state

    def load_state_dict(self, state_dict):
        if "logs" in state_dict:
            self.logs = defaultdict(list, state_dict["logs"])
            self.info("Logger state dict loaded successfully.")
        else:
            self.logs = defaultdict(list)


def get_logger(cfg, exp_dir):
    logger_type = get_config_value(cfg, "logger.type", "simple").lower()
    if logger_type == "tensorboard":
        log_dir = get_config_value(cfg, "logger.path", "logs")
        log_dir: str = os.path.join(exp_dir, log_dir)
        tb_logger = SummaryWriter(log_dir=log_dir)
        return TensorboardLogger(cfg, tb_logger)
    elif logger_type == "simple":
        return SimpleLogger(cfg)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
