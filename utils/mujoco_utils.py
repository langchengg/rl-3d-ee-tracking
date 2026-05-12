from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


PANDA_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
EE_SITE_NAME = "ee_site"


def resolve_model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    return model_path


def joint_addresses(model: mujoco.MjModel, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_addrs = []
    dof_addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
    return np.asarray(qpos_addrs, dtype=np.int32), np.asarray(dof_addrs, dtype=np.int32)


def joint_ranges(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    ranges = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ranges.append(model.jnt_range[jid])
    return np.asarray(ranges, dtype=np.float64)
