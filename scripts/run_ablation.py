from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import evaluate
from utils.logger import ensure_dir


ABLATION_CASES = [
    ("Clean", "IK", "configs/clean.yaml", "ik"),
    ("Clean", "Residual SAC", "configs/clean.yaml", "policy"),
    ("Action noise", "IK", "configs/noise.yaml", "ik"),
    ("Action noise", "Residual SAC", "configs/noise.yaml", "policy"),
    ("Command delay", "IK", "configs/delay.yaml", "ik"),
    ("Command delay", "Residual SAC", "configs/delay.yaml", "policy"),
    ("Trajectory mismatch", "IK", "configs/mismatch.yaml", "ik"),
    ("Trajectory mismatch", "Residual SAC", "configs/mismatch.yaml", "policy"),
    ("Mild combined", "IK", "configs/robust.yaml", "ik"),
    ("Mild combined", "Residual SAC", "configs/robust.yaml", "policy"),
]


def format_markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Setting | Controller | RMSE ↓ | Mean Error ↓ | Max Error ↓ | Smoothness ↓ | Jerk ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {controller} | {rmse:.4f} | {mean_error:.4f} | {max_error:.4f} | {smoothness:.4f} | {jerk:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines)


def run_ablation(
    checkpoint: str,
    trajectory: str = "figure_eight",
    episodes: int = 1,
    seed: int = 0,
    output: str = "results/logs/ablation_table.md",
) -> list[dict]:
    rows = []
    for setting, controller_name, config_path, controller in ABLATION_CASES:
        metrics = evaluate(
            config_path=config_path,
            checkpoint=checkpoint if controller == "policy" else None,
            trajectory=trajectory,
            controller=controller,
            output_prefix=f"ablation_{controller}_{setting.lower().replace(' ', '_').replace('+', 'and')}",
            episodes=episodes,
            seed=seed,
            make_plots=False,
        )
        rows.append({"setting": setting, "controller": controller_name, **metrics})

    output_path = Path(output)
    ensure_dir(output_path.parent)
    output_path.write_text(format_markdown_table(rows) + "\n", encoding="utf-8")
    print(output_path)
    print(format_markdown_table(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sac_residual_franka.zip")
    parser.add_argument("--trajectory", default="figure_eight")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/logs/ablation_table.md")
    args = parser.parse_args()
    run_ablation(args.checkpoint, args.trajectory, args.episodes, args.seed, args.output)


if __name__ == "__main__":
    main()
