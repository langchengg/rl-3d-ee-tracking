from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return _load_config_file(config_path)


def _load_config_file(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    base = config.pop("base", None)
    if base is None:
        return config

    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    return _deep_merge(_load_config_file(base_path), config)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
