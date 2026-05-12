from pathlib import Path

from configs import load_config


def test_load_config_merges_base_file_with_nested_overrides(tmp_path: Path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        """
env:
  episode_length: 300
control:
  residual_scale: 0.15
uncertainty:
  observation_noise_std: 0.0
  action_delay_steps: 0
""",
        encoding="utf-8",
    )
    child.write_text(
        """
base: base.yaml
control:
  residual_scale: 0.06
uncertainty:
  action_delay_steps: 1
""",
        encoding="utf-8",
    )

    cfg = load_config(child)

    assert cfg["env"]["episode_length"] == 300
    assert cfg["control"]["residual_scale"] == 0.06
    assert cfg["uncertainty"]["observation_noise_std"] == 0.0
    assert cfg["uncertainty"]["action_delay_steps"] == 1
