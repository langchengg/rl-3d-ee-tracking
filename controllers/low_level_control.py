from __future__ import annotations

import numpy as np


def velocity_to_position_target(
    qpos: np.ndarray,
    dq_cmd: np.ndarray,
    dt: float,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    q_target = np.asarray(qpos, dtype=np.float64) + np.asarray(dq_cmd, dtype=np.float64) * float(dt)
    return np.clip(q_target, joint_ranges[:, 0], joint_ranges[:, 1])
