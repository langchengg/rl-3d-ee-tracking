from __future__ import annotations

import numpy as np


def compute_reward(
    *,
    ee_pos: np.ndarray,
    ee_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    prev_prev_action: np.ndarray,
    joint_pos: np.ndarray,
    joint_ranges: np.ndarray,
    config: dict,
) -> float:
    pos_error = np.asarray(ee_pos) - np.asarray(target_pos)
    vel_error = np.asarray(ee_vel) - np.asarray(target_vel)
    action = np.asarray(action)
    prev_action = np.asarray(prev_action)
    prev_prev_action = np.asarray(prev_prev_action)

    smooth = action - prev_action
    jerk = action - 2.0 * prev_action + prev_prev_action
    joint_penalty = _joint_limit_penalty(np.asarray(joint_pos), np.asarray(joint_ranges))

    reward = (
        -float(config.get("w_pos", 10.0)) * float(np.dot(pos_error, pos_error))
        - float(config.get("w_vel", 1.0)) * float(np.dot(vel_error, vel_error))
        - float(config.get("w_action", 0.01)) * float(np.dot(action, action))
        - float(config.get("w_smooth", 0.1)) * float(np.dot(smooth, smooth))
        - float(config.get("w_jerk", 0.05)) * float(np.dot(jerk, jerk))
        - float(config.get("w_joint_limit", 1.0)) * joint_penalty
    )
    return float(reward)


def _joint_limit_penalty(q: np.ndarray, ranges: np.ndarray) -> float:
    lower = ranges[:, 0]
    upper = ranges[:, 1]
    margin = 0.08 * (upper - lower)
    low_violation = np.maximum(0.0, lower + margin - q)
    high_violation = np.maximum(0.0, q - (upper - margin))
    normalized = (low_violation + high_violation) / np.maximum(margin, 1e-6)
    return float(np.dot(normalized, normalized))
