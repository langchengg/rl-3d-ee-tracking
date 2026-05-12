from __future__ import annotations

import argparse
from pathlib import Path

from configs import load_config
from envs.franka_tracking_env import FrankaTrackingEnv
from utils.logger import ensure_dir
from utils.sb3_compat import import_monitor, import_sac


def train(
    config_path: str,
    total_timesteps: int | None = None,
    checkpoint_path: str = "checkpoints/sac_residual_franka.zip",
    seed: int | None = None,
    verbose: int = 0,
) -> str:
    cfg = load_config(config_path)
    if total_timesteps is not None:
        cfg["rl"]["total_timesteps"] = int(total_timesteps)
    if seed is not None:
        cfg["env"]["seed"] = int(seed)

    Monitor = import_monitor()
    env = Monitor(FrankaTrackingEnv(cfg), filename=str(ensure_dir("results/logs") / "monitor.csv"))
    rl_cfg = cfg["rl"]
    learning_starts = min(int(rl_cfg.get("learning_starts", 500)), max(int(rl_cfg["total_timesteps"]) // 4, 1))
    SAC = import_sac()
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=float(rl_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(rl_cfg.get("buffer_size", 100000)),
        batch_size=int(rl_cfg.get("batch_size", 128)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        tau=float(rl_cfg.get("tau", 0.005)),
        learning_starts=learning_starts,
        train_freq=1,
        gradient_steps=1,
        verbose=verbose,
        seed=cfg["env"].get("seed", 0),
        tensorboard_log=None,
    )
    model.learn(total_timesteps=int(rl_cfg["total_timesteps"]), progress_bar=False)
    output = Path(checkpoint_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    env.close()
    print(output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--checkpoint", default="checkpoints/sac_residual_franka.zip")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    train(args.config, args.timesteps, args.checkpoint, args.seed, args.verbose)


if __name__ == "__main__":
    main()
