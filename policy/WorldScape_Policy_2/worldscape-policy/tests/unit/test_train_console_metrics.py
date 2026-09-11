from worldscape_policy.cli.train import _console_metrics, _format_console_metrics


def test_console_metrics_omit_routine_diagnostics_and_inactive_losses() -> None:
    metrics = {
        "step": 10.0,
        "lr": 1e-4,
        "loss": 0.3,
        "grad_norm": 2.0,
        "action_flow/loss": 0.1,
        "action_flow/weighted_loss": 0.1,
        "action_flow/guard_hit": 0.0,
        "action_flow/nonfinite_fraction": 0.0,
        "action_flow/unguarded_loss": 0.1,
        "action_flow/skipped": 0.0,
        "video_flow/loss": 0.2,
        "video_flow/weighted_loss": 0.2,
        "semantic_forcing/skipped": 1.0,
        "semantic_forcing/loss": 0.0,
        "planning_ce/skipped": 1.0,
        "planning_ce/loss": 0.0,
        "prompt_schedule/auto": 0.5,
        "prompt_schedule/progress": 0.1,
        "prompt_schedule/stage": 1.0,
    }

    logged = _console_metrics(metrics)

    assert logged == {
        "action_flow/weighted_loss": 0.1,
        "grad_norm": 2.0,
        "loss": 0.3,
        "lr": 1e-4,
        "prompt_schedule/auto": 0.5,
        "prompt_schedule/stage": 1.0,
        "step": 10.0,
        "video_flow/weighted_loss": 0.2,
    }


def test_console_metrics_include_active_alignment_details() -> None:
    metrics = {
        "semantic_forcing/skipped": 0.0,
        "semantic_forcing/loss": 0.4,
        "semantic_forcing/weighted_loss": 0.0004,
        "semantic_forcing/cosine": 0.4,
        "semantic_forcing/mse": 0.7,
    }

    assert _console_metrics(metrics) == {
        "semantic_forcing/weighted_loss": 0.0004,
    }


def test_console_metrics_include_step_timing() -> None:
    metrics = {
        "step": 50.0,
        "lr": 1e-4,
        "loss": 0.3,
        "grad_norm": 2.0,
        "action_flow/weighted_loss": 0.1,
        "video_flow/weighted_loss": 0.2,
        "prompt_schedule/auto": 0.5,
        "prompt_schedule/stage": 1.0,
    }
    timing = {
        "step_time/prepare": 0.5123,
        "step_time/forward": 6.2345,
        "step_time/backward": 1.1111,
        "step_time/optimizer": 0.0500,
        "step_time/postprocess": 0.0200,
        "step_time/total": 7.9279,
    }

    logged = _console_metrics(metrics, timing)

    assert logged["step_time/forward"] == 6.2345
    assert logged["step_time/total"] == 7.9279


def test_console_metrics_format_losses_to_four_decimal_places() -> None:
    rendered = _format_console_metrics(
        {
            "step": 50.0,
            "lr": 0.00006,
            "loss": 0.123456,
            "action_flow/weighted_loss": 0.1,
            "prompt_schedule/stage": 1.0,
        }
    )

    assert rendered == (
        '{"action_flow/weighted_loss": 0.1000, "loss": 0.1235, '
        '"lr": 6e-05, "prompt_schedule/stage": 1, "step": 50}'
    )
