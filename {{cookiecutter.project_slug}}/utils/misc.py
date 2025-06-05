import torch
import logging
import os
from datetime import datetime


def save_checkpoint(static_dicts, file_path):
    """
    Save the model checkpoint.

    Args:
        static_dicts: The model state dictionary to save.
        file_path: The file path to save the checkpoint.
    """
    torch.save(static_dicts, file_path)
    logging.info(f"Checkpoint saved at {file_path}")


def load_checkpoint(file_path):
    """
    Load the model checkpoint.

    Args:
        file_path: The file path to load the checkpoint from.
    Returns:
        The loaded model.
    """
    static_dicts = torch.load(file_path, map_location="cpu")
    logging.info(f"Checkpoint loaded from {file_path}")
    return static_dicts


def create_exp_dir(cfg, exp_dir_base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(cfg.get("exp", {}).get("path", "runs"), f"{exp_dir_base_name}_{timestamp}")
    os.makedirs(exp_folder, exist_ok=True)
    logging.info(f"Experiment folder created at {exp_folder}")
    return exp_folder
