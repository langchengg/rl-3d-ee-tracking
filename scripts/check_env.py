from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import load_config
from envs.franka_tracking_env import FrankaTrackingEnv
from utils.sb3_compat import import_sb3_check_env


def main() -> None:
    cfg = load_config("configs/default.yaml")
    cfg["uncertainty"]["observation_noise_std"] = 0.0
    env = FrankaTrackingEnv(cfg)
    check_env = import_sb3_check_env()
    check_env(env, warn=True)
    env.close()
    print("Environment check passed")


if __name__ == "__main__":
    main()
