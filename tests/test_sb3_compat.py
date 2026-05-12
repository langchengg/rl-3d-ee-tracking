from utils.sb3_compat import import_monitor, import_sac, import_sb3_check_env


def test_stable_baselines3_imports_without_tensorboard_stack():
    sac = import_sac()
    monitor = import_monitor()
    check_env = import_sb3_check_env()

    assert sac.__name__ == "SAC"
    assert monitor.__name__ == "Monitor"
    assert callable(check_env)
