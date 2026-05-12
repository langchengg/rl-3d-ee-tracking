from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio


def save_video(frames: list, path: str | Path, fps: int = 50) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
