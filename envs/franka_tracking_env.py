from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from controllers.dls_ik import DampedLeastSquaresIK
from controllers.low_level_control import velocity_to_position_target
from envs.reward import compute_reward
from envs.trajectories import TrajectoryGenerator
from envs.wrappers import ActionDelayBuffer
from utils.mujoco_utils import EE_SITE_NAME, PANDA_JOINT_NAMES, joint_addresses, joint_ranges, resolve_model_path


class FrankaTrackingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        env_cfg = config["env"]
        model_path = resolve_model_path(env_cfg.get("model_path", "assets/franka_panda/panda.xml"))
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.episode_length = int(env_cfg.get("episode_length", 300))
        self.control_dt = float(env_cfg.get("control_dt", 0.02))
        self.frame_skip = int(env_cfg.get("frame_skip", 5))
        self.model.opt.timestep = self.control_dt / max(self.frame_skip, 1)
        self.model.opt.gravity[:] = np.asarray(env_cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)

        self.qpos_addrs, self.dof_addrs = joint_addresses(self.model, PANDA_JOINT_NAMES)
        self.joint_ranges = joint_ranges(self.model, PANDA_JOINT_NAMES)
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)
        if self.ee_site_id < 0:
            raise ValueError(f"Site not found in MuJoCo model: {EE_SITE_NAME}")

        control_cfg = config["control"]
        self.residual_scale = float(control_cfg.get("residual_scale", 0.15))
        self.max_joint_velocity = float(control_cfg.get("max_joint_velocity", 1.0))
        self.home_qpos = np.asarray(control_cfg.get("home_qpos", np.zeros(7)), dtype=np.float64)
        if self.home_qpos.shape != (7,):
            raise ValueError("control.home_qpos must contain 7 joint positions")
        self.ik = DampedLeastSquaresIK(
            self.model,
            EE_SITE_NAME,
            self.dof_addrs,
            damping=float(control_cfg.get("ik_damping", 0.05)),
            gain=float(control_cfg.get("ik_gain", 6.0)),
            max_velocity=self.max_joint_velocity,
        )

        uncertainty_cfg = config.get("uncertainty", {})
        self.observation_noise_std = float(uncertainty_cfg.get("observation_noise_std", 0.0))
        self.action_noise_std = float(uncertainty_cfg.get("action_noise_std", 0.0))
        self.unreachable_target_prob = float(uncertainty_cfg.get("unreachable_target_prob", 0.0))
        self.unreachable_offset = np.asarray(
            uncertainty_cfg.get("unreachable_offset", [0.25, 0.0, 0.12]),
            dtype=np.float64,
        )
        self.delay_buffer = ActionDelayBuffer(int(uncertainty_cfg.get("action_delay_steps", 0)), 7)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)

        self.rng = np.random.default_rng(int(env_cfg.get("seed", 0)))
        self.trajectory = TrajectoryGenerator(config["trajectory"], rng=self.rng)
        self.step_count = 0
        self.sim_time = 0.0
        self.current_target_pos = np.zeros(3, dtype=np.float64)
        self.current_target_vel = np.zeros(3, dtype=np.float64)
        self.current_phase = 0.0
        self.prev_action = np.zeros(7, dtype=np.float64)
        self.prev_prev_action = np.zeros(7, dtype=np.float64)
        self.last_dq_cmd = np.zeros(7, dtype=np.float64)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.trajectory = TrajectoryGenerator(self.config["trajectory"], rng=self.rng)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addrs] = np.clip(self.home_qpos, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        self.data.ctrl[:7] = self.data.qpos[self.qpos_addrs]
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.sim_time = 0.0
        self.delay_buffer.reset()
        self.prev_action.fill(0.0)
        self.prev_prev_action.fill(0.0)
        self.last_dq_cmd.fill(0.0)
        self._update_target(self.sim_time)
        obs = self._get_obs()
        return obs, self._info()

    def step(self, action: np.ndarray):
        raw_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delayed_residual = self.delay_buffer.push(raw_action)
        self._update_target(self.sim_time + self.control_dt)

        dq_ik = self.ik.compute(self.data, self.current_target_pos, self.current_target_vel)
        residual_dq = self.residual_scale * self.max_joint_velocity * delayed_residual
        dq_cmd = dq_ik + residual_dq
        if self.action_noise_std > 0.0:
            dq_cmd = dq_cmd + self.rng.normal(0.0, self.action_noise_std, size=7)
        dq_cmd = np.clip(dq_cmd, -self.max_joint_velocity, self.max_joint_velocity)

        q = self._joint_pos()
        q_target = velocity_to_position_target(q, dq_cmd, self.control_dt, self.joint_ranges)
        self.data.ctrl[:7] = q_target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.sim_time += self.control_dt

        ee_pos, ee_vel = self._end_effector_state()
        reward = compute_reward(
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            target_pos=self.current_target_pos,
            target_vel=self.current_target_vel,
            action=dq_cmd,
            prev_action=self.prev_action,
            prev_prev_action=self.prev_prev_action,
            joint_pos=self._joint_pos(),
            joint_ranges=self.joint_ranges,
            config=self.config["reward"],
        )

        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = dq_cmd.copy()
        self.last_dq_cmd = dq_cmd.copy()
        self.step_count += 1
        obs = self._get_obs()
        terminated = False
        truncated = self.step_count >= self.episode_length
        info = self._info()
        info.update(
            {
                "reward": reward,
                "dq_ik": dq_ik.copy(),
                "applied_residual_action": delayed_residual.copy(),
                "dq_cmd": dq_cmd.copy(),
            }
        )
        return obs, float(reward), terminated, truncated, info

    def render(self):
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data, camera="track")
        frame = renderer.render()
        renderer.close()
        return frame

    def close(self):
        return None

    def _update_target(self, t: float) -> None:
        pos, vel, phase = self.trajectory.sample(t)
        if self.unreachable_target_prob > 0.0 and self.rng.random() < self.unreachable_target_prob:
            pos = pos + self.unreachable_offset
        self.current_target_pos = pos
        self.current_target_vel = vel
        self.current_phase = phase

    def _get_obs(self) -> np.ndarray:
        q = self._joint_pos()
        dq = self._joint_vel()
        ee_pos, ee_vel = self._end_effector_state()
        obs = np.concatenate(
            [
                q,
                dq,
                ee_pos,
                ee_vel,
                self.current_target_pos,
                self.current_target_vel,
                self.current_target_pos - ee_pos,
                self.prev_action,
                [np.sin(self.current_phase), np.cos(self.current_phase)],
            ]
        ).astype(np.float32)
        if self.observation_noise_std > 0.0:
            obs = obs + self.rng.normal(0.0, self.observation_noise_std, size=obs.shape).astype(np.float32)
        return obs.astype(np.float32)

    def _joint_pos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.qpos_addrs], dtype=np.float64).copy()

    def _joint_vel(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self.dof_addrs], dtype=np.float64).copy()

    def _end_effector_state(self) -> tuple[np.ndarray, np.ndarray]:
        ee_pos = np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float64).copy()
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        ee_vel = jacp[:, self.dof_addrs] @ self._joint_vel()
        return ee_pos, ee_vel

    def _info(self) -> dict[str, Any]:
        ee_pos, ee_vel = self._end_effector_state()
        return {
            "time": self.sim_time,
            "ee_pos": ee_pos,
            "ee_vel": ee_vel,
            "target_pos": self.current_target_pos.copy(),
            "target_vel": self.current_target_vel.copy(),
            "tracking_error": float(np.linalg.norm(ee_pos - self.current_target_pos)),
            "last_dq_cmd": self.last_dq_cmd.copy(),
        }
