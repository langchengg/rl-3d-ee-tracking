from __future__ import annotations

import argparse

import numpy as np

from configs import load_config
from envs.franka_tracking_env import FrankaTrackingEnv
from plot_results import plot_evaluation
from utils.logger import ensure_dir, save_json
from utils.metrics import compute_tracking_metrics
from utils.sb3_compat import import_sac


def apply_evaluation_overrides(
    cfg: dict,
    *,
    clean: bool = False,
    obs_noise: float | None = None,
    action_noise: float | None = None,
    delay_steps: int | None = None,
    unreachable_prob: float | None = None,
) -> None:
    uncertainty = cfg.setdefault("uncertainty", {})
    if clean:
        uncertainty["observation_noise_std"] = 0.0
        uncertainty["action_noise_std"] = 0.0
        uncertainty["action_delay_steps"] = 0
        uncertainty["unreachable_target_prob"] = 0.0
    if obs_noise is not None:
        uncertainty["observation_noise_std"] = float(obs_noise)
    if action_noise is not None:
        uncertainty["action_noise_std"] = float(action_noise)
    if delay_steps is not None:
        uncertainty["action_delay_steps"] = int(delay_steps)
    if unreachable_prob is not None:
        uncertainty["unreachable_target_prob"] = float(unreachable_prob)


def evaluate(
    config_path: str,
    checkpoint: str | None = None,
    trajectory: str | None = None,
    controller: str = "policy",
    output_prefix: str | None = None,
    episodes: int = 1,
    seed: int = 0,
    make_plots: bool = True,
    clean: bool = False,
    obs_noise: float | None = None,
    action_noise: float | None = None,
    delay_steps: int | None = None,
    unreachable_prob: float | None = None,
) -> dict[str, float]:
    cfg = load_config(config_path)
    if trajectory:
        cfg["trajectory"]["type"] = trajectory
    apply_evaluation_overrides(
        cfg,
        clean=clean,
        obs_noise=obs_noise,
        action_noise=action_noise,
        delay_steps=delay_steps,
        unreachable_prob=unreachable_prob,
    )
    env = FrankaTrackingEnv(cfg)

    model = None
    if controller == "policy":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --controller policy")
        SAC = import_sac()
        model = SAC.load(checkpoint, env=env)
    elif controller != "ik":
        raise ValueError("controller must be 'policy' or 'ik'")

    all_times = []
    all_actual = []
    all_target = []
    all_actions = []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            if controller == "policy":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            all_times.append(info["time"] + ep * env.episode_length * env.control_dt)
            all_actual.append(info["ee_pos"])
            all_target.append(info["target_pos"])
            all_actions.append(info["dq_cmd"])
    env.close()

    actual = np.asarray(all_actual, dtype=np.float64)
    target_pos = np.asarray(all_target, dtype=np.float64)
    actions = np.asarray(all_actions, dtype=np.float64)
    times = np.asarray(all_times, dtype=np.float64)
    metrics = compute_tracking_metrics(actual, target_pos, actions)

    prefix = output_prefix or f"evaluation_{controller}_{cfg['trajectory']['type']}"
    log_dir = ensure_dir("results/logs")
    npz_path = log_dir / f"{prefix}.npz"
    np.savez(
        npz_path,
        times=times,
        actual_positions=actual,
        target_positions=target_pos,
        actions=actions,
    )
    save_json(log_dir / f"{prefix}_metrics.json", metrics)
    if make_plots:
        plot_evaluation(npz_path, "results/plots")
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--controller", choices=["policy", "ik"], default="policy")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--obs-noise", type=float, default=None)
    parser.add_argument("--action-noise", type=float, default=None)
    parser.add_argument("--delay-steps", type=int, default=None)
    parser.add_argument("--unreachable-prob", type=float, default=None)
    args = parser.parse_args()
    evaluate(
        config_path=args.config,
        checkpoint=args.checkpoint,
        trajectory=args.trajectory,
        controller=args.controller,
        output_prefix=args.output_prefix,
        episodes=args.episodes,
        seed=args.seed,
        make_plots=not args.no_plots,
        clean=args.clean,
        obs_noise=args.obs_noise,
        action_noise=args.action_noise,
        delay_steps=args.delay_steps,
        unreachable_prob=args.unreachable_prob,
    )


if __name__ == "__main__":
    main()
