# 3D End-Effector Tracking with Residual Reinforcement Learning

This project trains and evaluates a Franka Panda-style 7-DoF arm in MuJoCo for continuous 3D end-effector trajectory tracking.

## What This Submission Focuses On

This submission is optimized around four criteria:

1. Tracking accuracy over time
2. Smoothness and stability
3. Robustness to noise and delay
4. A simple, interpretable residual-control design

## Method Summary

The controller is intentionally simple:

```text
dq_final = dq_DLS_IK + alpha * filtered(dq_RL)
```

A damped least-squares IK controller handles nominal geometric tracking. SAC from Stable-Baselines3 learns a small residual correction on top of that command. The residual action is bounded and low-pass filtered before it is added to the IK command, which reduces high-frequency jitter.

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
python evaluate.py --config configs/clean.yaml --controller ik --trajectory figure_eight --output-prefix evaluation_ik_clean_figure_eight
```

Train the SAC residual policy for the default 1,000,000 timesteps:

```bash
python train.py --checkpoint checkpoints/sac_residual_franka.zip --seed 0
```

Evaluate the trained residual policy:

```bash
python evaluate.py --config configs/clean.yaml --controller policy --checkpoint checkpoints/sac_residual_franka.zip --trajectory figure_eight --output-prefix evaluation_policy_clean
```

Run the robustness ablation table:

```bash
python scripts/run_ablation.py --checkpoint checkpoints/sac_residual_franka.zip --trajectory figure_eight
```

Generate a video:

```bash
python record_video.py --config configs/clean.yaml --controller policy --checkpoint checkpoints/sac_residual_franka.zip --trajectory figure_eight --output results/videos/franka_tracking.mp4 --render-mode plot --width 480 --height 368
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

The residual is filtered before application:

```text
u_filtered[t] = beta * u_filtered[t-1] + (1 - beta) * u_raw[t]
```

The DLS IK target also uses a short lookahead:

```text
x_ik = x_target + lookahead * v_target
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

## Evaluation Protocol

The main evaluation separates perturbations instead of stacking every uncertainty source at once:

- IK baseline on clean trajectory
- Residual SAC on clean trajectory
- Residual SAC with observation/action noise only
- Residual SAC with one-step action delay only
- Residual SAC with mild combined noise + delay

Unreachable targets are kept as a separate stress test rather than a main robustness metric.

## Results

Results below are from `scripts/run_ablation.py` using the trained SAC residual checkpoint:

| Setting | Controller | RMSE ↓ | Mean Error ↓ | Max Error ↓ | Smoothness ↓ | Jerk ↓ |
|---|---|---:|---:|---:|---:|---:|
| Clean | IK | 0.0495 | 0.0470 | 0.0734 | 0.0022 | 0.0000 |
| Clean | Residual SAC | 0.0494 | 0.0470 | 0.0719 | 0.0022 | 0.0000 |
| Noise only | Residual SAC | 0.0494 | 0.0470 | 0.0718 | 0.0023 | 0.0003 |
| Delay only | Residual SAC | 0.0494 | 0.0470 | 0.0719 | 0.0022 | 0.0000 |
| Mild noise + delay | Residual SAC | 0.0494 | 0.0470 | 0.0718 | 0.0023 | 0.0003 |

The low-pass filtered residual command keeps the SAC policy close to the IK controller on clean tracking while slightly reducing max error. Mild noise and one-step delay do not cause the large max-error spike seen when unreachable targets were mixed into the main result table.

Generated outputs:

- `results/plots/3d_trajectory.png`
- `results/plots/xyz_tracking.png`
- `results/plots/tracking_error.png`
- `results/plots/action_smoothness.png`
- `results/videos/franka_tracking.mp4`

## Design Notes

Residual RL is used because pure joint-space RL is slow and can produce unsafe exploratory motion. The IK controller handles the obvious geometry, while SAC learns bounded corrections for tracking dynamics, delay, noise, and model mismatch.

The MuJoCo model uses a simplified Panda-style MJCF with seven revolute joints and Panda-like joint limits. The default simulation uses zero gravity to approximate a low-level gravity-compensated joint position controller, which is a common assumption when training higher-level joint velocity policies.

## Limitations

The IK baseline is already strong on clean geometric tracking, so the residual policy provides limited RMSE improvement in the current version. The value of the RL component here is mainly in the residual-control structure: learned corrections are bounded, filtered, and evaluated under transparent perturbation ablations.
