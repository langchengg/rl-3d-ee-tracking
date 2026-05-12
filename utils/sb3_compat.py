from __future__ import annotations

import builtins
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator


def _prepare_matplotlib_cache() -> None:
    cache_root = Path("results/logs/cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    cache_dir = cache_root / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


@contextmanager
def _without_torch_tensorboard() -> Iterator[None]:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch.utils.tensorboard" or name.startswith("torch.utils.tensorboard."):
            raise ImportError("TensorBoard import disabled for stable-baselines3 compatibility")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import
    try:
        yield
    finally:
        builtins.__import__ = real_import


def import_sac():
    _prepare_matplotlib_cache()
    with _without_torch_tensorboard():
        from stable_baselines3 import SAC

    return SAC


def import_sb3_check_env():
    _prepare_matplotlib_cache()
    with _without_torch_tensorboard():
        from stable_baselines3.common.env_checker import check_env

    return check_env


def import_monitor():
    _prepare_matplotlib_cache()
    with _without_torch_tensorboard():
        from stable_baselines3.common.monitor import Monitor

    return Monitor
