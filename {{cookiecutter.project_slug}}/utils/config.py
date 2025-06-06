import os
import yaml
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_cfg(config_path: str = "config/end2end_config.yaml") -> DictConfig:
    """
    加载配置文件，可通过命令行参数 --config 覆盖默认路径。

    Returns:
        DictConfig: 配置字典
    """
    parser = argparse.ArgumentParser(description="加载配置文件")
    parser.add_argument("--config", "-cfg", type=str, default=config_path, help="Path to the configuration file.")
    args, _ = parser.parse_known_args()

    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if 'base_dir' not in cfg:
        cfg['base_dir'] = str(Path(__file__).parent.parent)

    return OmegaConf.create(cfg)


def get_value_from_cfg(cfg: DictConfig, keys: str, default=None):
    """
    从配置字典中获取嵌套的值。
    Args:
        cfg (dict): 配置字典
        keys (str): 键嵌套路径，以点分隔的字符串形式，例如 "key1.key2.key3"
        default: 如果键序列不存在，则返回默认值（并自动设置）
    Returns:
        配置字典中指定键的值，如果不存在则返回默认值
    """
    value = OmegaConf.select(cfg, keys)
    if value is None:
        value = default
        OmegaConf.update(cfg, keys, default, merge=True)
    return value


def load_end2end_cfg():
    return load_cfg(str(Path(__file__).parent.parent / "config" / "end2end_config.yaml"))
