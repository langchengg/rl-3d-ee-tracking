from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    phase: float


class TrajectoryGenerator:
    def __init__(self, config: dict, rng: np.random.Generator | None = None):
        self.config = dict(config)
        self.rng = rng or np.random.default_rng()
        self.kind = self.config.get("type", "figure_eight")
        self.center = np.asarray(self.config.get("center", [0.48, 0.0, 0.42]), dtype=np.float64)
        self.frequency = float(self.config.get("frequency", 0.25))
        self.omega = 2.0 * np.pi * self.frequency

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        phase = self.omega * float(t)
        kind = self.kind.lower()
        if kind == "circle":
            pos, vel = self._circle(phase)
        elif kind in {"figure_eight", "figure-eight", "eight"}:
            pos, vel = self._figure_eight(phase)
        elif kind in {"noisy", "moving_target", "moving-target"}:
            pos, vel = self._figure_eight(phase)
            noise_std = float(self.config.get("target_noise_std", 0.01))
            pos = pos + self.rng.normal(0.0, noise_std, size=3)
        else:
            raise ValueError(f"Unsupported trajectory type: {self.kind}")
        return pos.astype(np.float64), vel.astype(np.float64), phase

    def _circle(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        radius = float(self.config.get("radius", 0.12))
        pos = self.center + np.array(
            [radius * np.cos(phase), radius * np.sin(phase), 0.0],
            dtype=np.float64,
        )
        vel = np.array(
            [-radius * self.omega * np.sin(phase), radius * self.omega * np.cos(phase), 0.0],
            dtype=np.float64,
        )
        return pos, vel

    def _figure_eight(self, phase: float) -> tuple[np.ndarray, np.ndarray]:
        amp_x = float(self.config.get("amplitude_x", self.config.get("radius", 0.12)))
        amp_y = float(self.config.get("amplitude_y", self.config.get("radius", 0.08)))
        pos = self.center + np.array(
            [amp_x * np.sin(phase), amp_y * np.sin(2.0 * phase), 0.0],
            dtype=np.float64,
        )
        vel = np.array(
            [
                amp_x * self.omega * np.cos(phase),
                2.0 * amp_y * self.omega * np.cos(2.0 * phase),
                0.0,
            ],
            dtype=np.float64,
        )
        return pos, vel
