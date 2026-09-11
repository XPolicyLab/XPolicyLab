import json
import sys
from types import SimpleNamespace

from worldscape_policy.training.wandb_logger import WandbRunLogger


class _Run:
    id = "generated-run-id"

    def __init__(self) -> None:
        self.logs: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self.logs.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_persists_and_resumes_run_id(tmp_path, monkeypatch) -> None:
    run = _Run()
    init_calls = []

    def init(**kwargs):
        init_calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    logger = WandbRunLogger(
        enabled=True,
        output_dir=tmp_path,
        project="worldscape_policy",
        name="fold-shirt",
        mode="offline",
        config={"training": {"max_steps": 10}},
    )

    logger.start()
    logger.log({"loss": 0.5}, step=3)
    logger.finish()

    assert init_calls[0]["project"] == "worldscape_policy"
    assert init_calls[0]["name"] == "fold-shirt"
    assert init_calls[0]["mode"] == "offline"
    assert run.logs == [({"loss": 0.5}, 3)]
    assert run.finished
    metadata = json.loads((tmp_path / "wandb_config.json").read_text())
    assert metadata["run_id"] == "generated-run-id"

    second_run = _Run()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: init_calls.append(kwargs) or second_run),
    )
    resumed = WandbRunLogger(
        enabled=True,
        output_dir=tmp_path,
        project="worldscape_policy",
        name="fold-shirt",
        mode="offline",
    )
    resumed.start()

    assert init_calls[-1]["id"] == "generated-run-id"
    assert init_calls[-1]["resume"] == "allow"
