import numpy as np

from configs import load_config
from envs.franka_tracking_env import FrankaTrackingEnv
from envs.reward import compute_reward
from envs.trajectories import TrajectoryGenerator
from utils.metrics import compute_tracking_metrics


def test_circle_trajectory_matches_position_and_velocity_equations():
    cfg = {
        "type": "circle",
        "radius": 0.2,
        "center": [0.5, -0.1, 0.4],
        "frequency": 0.5,
    }
    traj = TrajectoryGenerator(cfg, rng=np.random.default_rng(0))

    pos, vel, phase = traj.sample(0.0)

    assert np.allclose(pos, [0.7, -0.1, 0.4])
    assert np.allclose(vel, [0.0, 0.2 * 2.0 * np.pi * 0.5, 0.0])
    assert phase == 0.0


def test_figure_eight_velocity_matches_finite_difference():
    cfg = {
        "type": "figure_eight",
        "amplitude_x": 0.12,
        "amplitude_y": 0.08,
        "center": [0.5, 0.0, 0.45],
        "frequency": 0.25,
    }
    traj = TrajectoryGenerator(cfg, rng=np.random.default_rng(0))

    t = 0.7
    dt = 1e-5
    pos_prev, _, _ = traj.sample(t - dt)
    pos_next, _, _ = traj.sample(t + dt)
    _, vel, _ = traj.sample(t)

    assert np.allclose(vel, (pos_next - pos_prev) / (2 * dt), atol=1e-5)


def test_reward_penalizes_tracking_error_and_action_jitter():
    cfg = {
        "w_pos": 10.0,
        "w_vel": 1.0,
        "w_action": 0.01,
        "w_smooth": 0.1,
        "w_jerk": 0.05,
        "w_joint_limit": 1.0,
    }
    base = dict(
        ee_vel=np.zeros(3),
        target_vel=np.zeros(3),
        action=np.zeros(7),
        prev_action=np.zeros(7),
        prev_prev_action=np.zeros(7),
        joint_pos=np.zeros(7),
        joint_ranges=np.tile(np.array([-2.0, 2.0]), (7, 1)),
        config=cfg,
    )

    near = compute_reward(
        ee_pos=np.array([0.5, 0.0, 0.4]),
        target_pos=np.array([0.51, 0.0, 0.4]),
        **base,
    )
    far = compute_reward(
        ee_pos=np.array([0.5, 0.0, 0.4]),
        target_pos=np.array([0.7, 0.0, 0.4]),
        **base,
    )
    jitter = compute_reward(
        ee_pos=np.array([0.5, 0.0, 0.4]),
        target_pos=np.array([0.51, 0.0, 0.4]),
        **{**base, "action": np.ones(7), "prev_action": -np.ones(7)},
    )

    assert near > far
    assert near > jitter


def test_tracking_metrics_report_rmse_max_error_and_smoothness():
    target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    actual = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    metrics = compute_tracking_metrics(actual, target, actions)

    assert np.isclose(metrics["rmse"], np.sqrt(0.5))
    assert np.isclose(metrics["max_error"], 1.0)
    assert np.isclose(metrics["mean_error"], 0.5)
    assert metrics["smoothness"] > 0.0
    assert metrics["jerk"] > 0.0


def test_env_reset_step_contract_and_uncertainty_delay():
    cfg = load_config("configs/default.yaml")
    cfg["env"]["episode_length"] = 5
    cfg["env"]["frame_skip"] = 1
    cfg["uncertainty"]["observation_noise_std"] = 0.0
    cfg["uncertainty"]["action_noise_std"] = 0.0
    cfg["uncertainty"]["action_delay_steps"] = 2
    cfg["uncertainty"]["unreachable_target_prob"] = 0.0

    env = FrankaTrackingEnv(cfg)
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (7,)
    assert "ee_pos" in info

    action = np.linspace(-0.5, 0.5, 7, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert terminated is False
    assert truncated is False
    assert info["applied_residual_action"].shape == (7,)
    assert np.allclose(info["applied_residual_action"], np.zeros(7))

    env.close()


def test_env_filters_delayed_residual_action_and_reports_lookahead_target():
    cfg = load_config("configs/default.yaml")
    cfg["env"]["episode_length"] = 5
    cfg["env"]["frame_skip"] = 1
    cfg["control"]["residual_filter_beta"] = 0.5
    cfg["control"]["target_lookahead"] = 0.04
    cfg["uncertainty"]["observation_noise_std"] = 0.0
    cfg["uncertainty"]["action_noise_std"] = 0.0
    cfg["uncertainty"]["action_delay_steps"] = 0
    cfg["uncertainty"]["unreachable_target_prob"] = 0.0

    env = FrankaTrackingEnv(cfg)
    obs, info = env.reset(seed=123)
    action = np.ones(7, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert np.allclose(info["applied_residual_action"], 0.5 * np.ones(7))
    assert np.allclose(
        info["ik_target_pos"],
        info["target_pos"] + cfg["control"]["target_lookahead"] * info["target_vel"],
    )

    env.close()
