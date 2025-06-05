import os
import yaml
import argparse
from pathlib import Path


def load_cfg(config_path: str = "config/end2end_config.yaml") -> dict:
    """
    加载配置文件，可通过命令行参数 --config 覆盖默认路径。

    Returns:
        dict: 配置字典
    """
    parser = argparse.ArgumentParser(description="加载配置文件")
    parser.add_argument("--config", "-cfg", type=str, default=config_path, help="Path to the configuration file.")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if 'base_dir' not in cfg:
        cfg['base_dir'] = str(Path(__file__).parent.parent)

    return cfg


def load_end2end_cfg():
    return load_cfg(str(Path(__file__).parent.parent / "config" / "end2end_config.yaml"))
