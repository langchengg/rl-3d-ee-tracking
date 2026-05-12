from __future__ import annotations

import numpy as np
import mujoco


class DampedLeastSquaresIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str,
        dof_indices: np.ndarray,
        damping: float = 0.05,
        gain: float = 6.0,
        max_velocity: float = 1.0,
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site not found in MuJoCo model: {site_name}")
        self.dof_indices = np.asarray(dof_indices, dtype=np.int32)
        self.damping = float(damping)
        self.gain = float(gain)
        self.max_velocity = float(max_velocity)

    def compute(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        ee_pos = np.asarray(data.site_xpos[self.site_id], dtype=np.float64)
        desired_vel = self.gain * (np.asarray(target_pos, dtype=np.float64) - ee_pos)
        if target_vel is not None:
            desired_vel = desired_vel + np.asarray(target_vel, dtype=np.float64)

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        j = jacp[:, self.dof_indices]
        lhs = j @ j.T + (self.damping**2) * np.eye(3)
        dq = j.T @ np.linalg.solve(lhs, desired_vel)
        return np.clip(dq, -self.max_velocity, self.max_velocity)
