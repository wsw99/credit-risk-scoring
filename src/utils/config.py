"""YAML configuration loader with dot-notation access."""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """A dict-like config object with attribute-style (dot) access.

    Usage:
        cfg = Config.from_yaml("configs/xgboost.yaml")
        lr = cfg.training.learning_rate
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data or {})

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                value = value.to_dict()
            result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    def __getitem__(self, key: str) -> Any:
        return self.__dict__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__


def load_config(path: str) -> Config:
    """Load a YAML config file and return a Config object."""
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path("configs") / path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return Config.from_yaml(config_path)
