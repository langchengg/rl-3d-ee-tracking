from __future__ import annotations

import numpy as np


def compute_tracking_metrics(
    actual_positions: np.ndarray,
    target_positions: np.ndarray,
    actions: np.ndarray | None = None,
) -> dict[str, float]:
    actual = np.asarray(actual_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    errors = np.linalg.norm(actual - target, axis=1)
    metrics = {
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
        "mean_error": float(np.mean(errors)) if len(errors) else 0.0,
        "max_error": float(np.max(errors)) if len(errors) else 0.0,
        "smoothness": 0.0,
        "jerk": 0.0,
    }
    if actions is not None:
        action_arr = np.asarray(actions, dtype=np.float64)
        if len(action_arr) >= 2:
            da = np.diff(action_arr, axis=0)
            metrics["smoothness"] = float(np.mean(np.sum(da**2, axis=1)))
        if len(action_arr) >= 3:
            dda = np.diff(action_arr, n=2, axis=0)
            metrics["jerk"] = float(np.mean(np.sum(dda**2, axis=1)))
    return metrics
