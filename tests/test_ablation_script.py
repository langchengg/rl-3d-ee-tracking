from scripts.run_ablation import format_markdown_table


def test_format_markdown_table_includes_metrics_and_controller_names():
    rows = [
        {
            "setting": "Clean",
            "controller": "IK",
            "rmse": 0.01234,
            "mean_error": 0.01111,
            "max_error": 0.02222,
            "smoothness": 0.03333,
            "jerk": 0.04444,
        }
    ]

    table = format_markdown_table(rows)

    assert "| Clean | IK | 0.0123 | 0.0111 | 0.0222 | 0.0333 | 0.0444 |" in table
    assert "Jerk" in table
