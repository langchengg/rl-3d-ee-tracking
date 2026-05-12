# Design Note

## Control Architecture

The policy is not asked to solve the full robot control problem from scratch. A damped least-squares IK controller computes a baseline joint velocity from Cartesian position error and target velocity:

```text
dq_ik = J^T (J J^T + lambda^2 I)^-1 (Kp (x_target - x_ee) + v_target)
```

The SAC policy outputs a normalized 7D residual action. The residual is delayed if configured, low-pass filtered, scaled, and then added to the IK velocity:

```text
u_filtered[t] = beta * u_filtered[t-1] + (1 - beta) * u_rl[t]
dq_final = dq_ik + residual_scale * max_joint_velocity * u_filtered[t]
```

This keeps exploration bounded and makes the learned component responsible for correction rather than basic reaching.

The IK target uses a short lookahead:

```text
x_ik = x_target + lookahead * v_target
```

This reduces phase lag on periodic trajectories.

## Trajectory Representation

The target is represented analytically as position, velocity, and phase. The phase is exposed as `sin(phase), cos(phase)` so the policy can infer where it is in the periodic trajectory without a discontinuous time variable.

## Uncertainty

The environment supports four uncertainty sources:

- observation noise
- action noise
- action delay
- occasional unreachable target offsets

The main results separate uncertainty into clean, noise-only, delay-only, and mild combined settings. Unreachable targets are kept as a separate stress test because mixing them into the primary robustness table can obscure ordinary noise/delay behavior.

## Evaluation

Tracking is evaluated over complete episodes using:

- RMSE of Cartesian position error
- mean and max Cartesian error
- action smoothness, `mean(||a_t - a_{t-1}||^2)`
- jerk, `mean(||a_t - 2a_{t-1} + a_{t-2}||^2)`

The README reports clean and robustness ablations using the trained SAC residual checkpoint, plus IK baselines for comparison.
