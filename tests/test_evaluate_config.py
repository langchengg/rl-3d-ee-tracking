from evaluate import apply_evaluation_overrides


def test_clean_evaluation_override_disables_uncertainty():
    cfg = {
        "uncertainty": {
            "observation_noise_std": 0.1,
            "action_noise_std": 0.2,
            "action_delay_steps": 3,
            "unreachable_target_prob": 0.4,
        }
    }

    apply_evaluation_overrides(cfg, clean=True)

    assert cfg["uncertainty"]["observation_noise_std"] == 0.0
    assert cfg["uncertainty"]["action_noise_std"] == 0.0
    assert cfg["uncertainty"]["action_delay_steps"] == 0
    assert cfg["uncertainty"]["unreachable_target_prob"] == 0.0
