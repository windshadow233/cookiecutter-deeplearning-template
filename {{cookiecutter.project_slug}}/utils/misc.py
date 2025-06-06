import torch
import logging
import os
from datetime import datetime
from pickle import UnpicklingError
from utils.config import get_value_from_cfg


def save_checkpoint(obj, file_path):
    """
    Save the model checkpoint.

    Args:
        obj: The obj to save.
        file_path: The file path to save the checkpoint.
    """
    torch.save(obj, file_path)
    logging.info(f"Checkpoint saved at {file_path}")


def load_checkpoint(file_path):
    """
    Load the model checkpoint.

    Args:
        file_path: The file path to load the checkpoint from.
    Returns:
        The loaded model.
    """
    obj = torch.load(file_path, map_location="cpu")
    logging.info(f"Checkpoint loaded from {file_path}")
    return obj


def create_exp_dir(cfg, exp_dir_base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = get_value_from_cfg(cfg, "exp.path", "runs")
    exp_folder = os.path.join(exp_folder, f"{exp_dir_base_name}_{timestamp}")
    os.makedirs(exp_folder, exist_ok=True)
    logging.info(f"Experiment folder created at {exp_folder}")
    return exp_folder
