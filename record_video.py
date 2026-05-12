from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from configs import load_config
from envs.franka_tracking_env import FrankaTrackingEnv
from utils.sb3_compat import import_sac
from utils.video import save_video

Path("results/logs/cache/matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(Path("results/logs/cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path("results/logs/cache/matplotlib").resolve()))


def record_video(
    config_path: str,
    checkpoint: str | None = None,
    controller: str = "policy",
    trajectory: str | None = None,
    output: str = "results/videos/franka_tracking.mp4",
    seed: int = 0,
    width: int = 960,
    height: int = 720,
    render_mode: str = "auto",
) -> str:
    cfg = load_config(config_path)
    if trajectory:
        cfg["trajectory"]["type"] = trajectory
    env = FrankaTrackingEnv(cfg)
    model = None
    if controller == "policy":
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --controller policy")
        SAC = import_sac()
        model = SAC.load(checkpoint, env=env)
    elif controller != "ik":
        raise ValueError("controller must be 'policy' or 'ik'")

    obs, info = env.reset(seed=seed)
    renderer = None
    if render_mode in {"auto", "mujoco"}:
        try:
            renderer = mujoco.Renderer(env.model, height=height, width=width)
        except Exception as exc:
            if render_mode == "mujoco":
                raise
            print(f"MuJoCo renderer unavailable ({exc}); using matplotlib fallback.")
    frames = []
    actual_history = []
    target_history = []
    done = False
    while not done:
        if controller == "policy":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        actual_history.append(info["ee_pos"].copy())
        target_history.append(info["target_pos"].copy())
        if renderer is not None:
            renderer.update_scene(env.data, camera="track")
            frames.append(renderer.render())
        else:
            frames.append(_render_plot_frame(env, actual_history, target_history, width, height))
        done = terminated or truncated
    if renderer is not None:
        renderer.close()
    env.close()
    save_video(frames, output, fps=int(round(1.0 / cfg["env"]["control_dt"])))
    print(output)
    return str(Path(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--controller", choices=["policy", "ik"], default="policy")
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output", default="results/videos/franka_tracking.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--render-mode", choices=["auto", "mujoco", "plot"], default="auto")
    args = parser.parse_args()
    record_video(
        config_path=args.config,
        checkpoint=args.checkpoint,
        controller=args.controller,
        trajectory=args.trajectory,
        output=args.output,
        seed=args.seed,
        width=args.width,
        height=args.height,
        render_mode=args.render_mode,
    )


def _render_plot_frame(env: FrankaTrackingEnv, actual_history: list[np.ndarray], target_history: list[np.ndarray], width: int, height: int):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    body_names = [
        "panda_base",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]
    points = []
    for name in body_names:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            points.append(env.data.xpos[bid].copy())
    points.append(env.data.site_xpos[env.ee_site_id].copy())
    arm = np.asarray(points)
    actual = np.asarray(actual_history)
    target = np.asarray(target_history)

    ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], color="#1f2937", linewidth=4, marker="o", markersize=3)
    ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#2563eb", linewidth=2, label="target")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], color="#dc2626", linewidth=2, label="end effector")
    ax.scatter(target[-1, 0], target[-1, 1], target[-1, 2], color="#2563eb", s=35)
    ax.scatter(actual[-1, 0], actual[-1, 1], actual[-1, 2], color="#dc2626", s=35)
    ax.set_xlim(0.20, 0.70)
    ax.set_ylim(-0.32, 0.26)
    ax.set_zlim(0.15, 0.58)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-58)
    ax.legend(loc="upper left")
    fig.tight_layout()
    canvas.draw()
    frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    return frame


if __name__ == "__main__":
    main()
