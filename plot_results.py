from __future__ import annotations

import argparse
import os
from pathlib import Path

Path("results/logs/cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/logs/cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/logs/cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.logger import ensure_dir


def plot_evaluation(log_path: str | Path, output_dir: str | Path = "results/plots") -> list[Path]:
    data = np.load(log_path)
    times = data["times"]
    actual = data["actual_positions"]
    target = data["target_positions"]
    actions = data["actions"]
    errors = np.linalg.norm(actual - target, axis=1)
    smoothness = np.zeros(len(actions), dtype=np.float64)
    if len(actions) > 1:
        smoothness[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)

    output = ensure_dir(output_dir)
    written = []

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(target[:, 0], target[:, 1], target[:, 2], label="target", linewidth=2)
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], label="end effector", linewidth=2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    path = output / "3d_trajectory.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    labels = ["x", "y", "z"]
    for idx, axis in enumerate(axes):
        axis.plot(times, target[:, idx], label=f"target {labels[idx]}")
        axis.plot(times, actual[:, idx], label=f"actual {labels[idx]}")
        axis.set_ylabel(f"{labels[idx]} [m]")
        axis.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    path = output / "xyz_tracking.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, errors)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tracking error [m]")
    ax.set_title("Cartesian Tracking Error")
    path = output / "tracking_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, smoothness)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||a_t - a_{t-1}||")
    ax.set_title("Action Smoothness")
    path = output / "action_smoothness.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="results/logs/evaluation_policy_figure_eight.npz")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()
    written = plot_evaluation(args.log, args.output_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
