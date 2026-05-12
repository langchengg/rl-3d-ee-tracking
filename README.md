# 3D End-Effector Tracking with Residual Reinforcement Learning

This project trains and evaluates a Franka Panda-style 7-DoF arm in MuJoCo for continuous 3D end-effector trajectory tracking.

The control stack is residual RL:

```text
dq_final = dq_DLS_IK + alpha * dq_RL
```

A damped least-squares IK controller provides a stable baseline joint velocity command. SAC from Stable-Baselines3 learns a bounded residual correction on top of that command.

## Features

- MuJoCo Franka Panda-style 7-DoF arm model
- Gymnasium-compatible `FrankaTrackingEnv`
- Circle, figure-eight, and noisy moving-target trajectories
- SAC residual policy with Stable-Baselines3
- Observation noise, action noise, action delay, and occasional unreachable targets
- RMSE, max error, smoothness, and jerk metrics
- Plot generation and mp4 video recording

## Installation

```bash
pip install -r requirements.txt
```

The current implementation was tested with Python 3.12. In the provided environment, Stable-Baselines3 imported a broken TensorBoard/Keras stack, so the project uses a narrow compatibility wrapper that disables TensorBoard imports while keeping SAC intact.

## Quick Start

Check the Gymnasium/SB3 environment contract:

```bash
python scripts/check_env.py
```

Run the IK baseline:

```bash
python evaluate.py --controller ik --trajectory figure_eight --clean --output-prefix evaluation_ik_clean_figure_eight
```

Train the SAC residual policy for the default 1,000,000 timesteps:

```bash
python train.py --checkpoint checkpoints/sac_residual_franka.zip --seed 0
```

Evaluate the trained residual policy:

```bash
python evaluate.py --controller policy --checkpoint checkpoints/sac_residual_franka.zip --trajectory figure_eight --clean --output-prefix evaluation_policy_1m_clean_figure_eight
```

Generate a video:

```bash
python record_video.py --controller policy --checkpoint checkpoints/sac_residual_franka.zip --trajectory figure_eight --output results/videos/franka_tracking.mp4 --render-mode plot --width 480 --height 368
```

Use `--render-mode auto` on a machine with a working MuJoCo OpenGL context. The `plot` mode is a headless fallback for CI/sandboxed machines.

## State, Action, Reward

Observation, 38D:

```text
q, dq, ee_pos, ee_vel, target_pos, target_vel, target_pos - ee_pos, previous_action, sin(phase), cos(phase)
```

Action, 7D:

```text
bounded residual joint velocity in [-1, 1]
```

Reward:

```text
r = - w_pos ||x_ee - x_target||^2
    - w_vel ||v_ee - v_target||^2
    - w_action ||a_t||^2
    - w_smooth ||a_t - a_{t-1}||^2
    - w_jerk ||a_t - 2a_{t-1} + a_{t-2}||^2
    - w_joint joint_limit_penalty
```

The reward prioritizes Cartesian tracking accuracy while penalizing high-frequency action changes to encourage smooth, hardware-friendly motion.

## Trajectories

Circle:

```text
x = x0 + r cos(wt)
y = y0 + r sin(wt)
z = z0
```

Figure-eight:

```text
x = x0 + A sin(wt)
y = y0 + B sin(2wt)
z = z0
```

The noisy target mode adds Gaussian perturbations to the clean figure-eight target.

## Results From 1,000,000-Step Run

These are the commands I ran locally in this repository:

| Setting | Controller | RMSE ↓ | Mean Error ↓ | Max Error ↓ | Smoothness ↓ |
|---|---|---:|---:|---:|---:|
| Clean figure-eight | IK | 0.0549 m | 0.0526 m | 0.0827 m | 0.0080 |
| Clean circle | IK | 0.0630 m | 0.0622 m | 0.1014 m | 0.0071 |
| Clean figure-eight | SAC residual, 1,000,000 steps | 0.0547 m | 0.0528 m | 0.0775 m | 0.0117 |
| Noisy/delay/unreachable | SAC residual, 1,000,000 steps | 0.0753 m | 0.0623 m | 0.3333 m | 1.1742 |

The trained residual policy slightly reduces the clean figure-eight max tracking error relative to the IK baseline, while keeping RMSE roughly comparable. The robustness setting is harder because the default config includes observation noise, action noise, two-step action delay, and occasional unreachable target offsets.

Generated outputs:

- `results/plots/3d_trajectory.png`
- `results/plots/xyz_tracking.png`
- `results/plots/tracking_error.png`
- `results/plots/action_smoothness.png`
- `results/videos/franka_tracking.mp4`

## Design Notes

Residual RL is used because pure joint-space RL is slow and can produce unsafe exploratory motion. The IK controller handles the obvious geometry, while SAC learns corrections for tracking dynamics, delay, noise, and model mismatch.

The MuJoCo model uses a simplified Panda-style MJCF with seven revolute joints and Panda-like joint limits. The default simulation uses zero gravity to approximate a low-level gravity-compensated joint position controller, which is a common assumption when training higher-level joint velocity policies.
