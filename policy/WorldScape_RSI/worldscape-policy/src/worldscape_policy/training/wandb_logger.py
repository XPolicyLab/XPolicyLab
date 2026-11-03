"""Optional rank-zero Weights & Biases logging."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class WandbRunLogger:
    """Own one resumable W&B run without affecting trainer checkpoint state."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | Path,
        project: str,
        name: str,
        mode: str = "offline",
        entity: str | None = None,
        run_id: str | None = None,
        group: str | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.project = str(project)
        self.name = str(name)
        self.mode = str(mode)
        self.entity = entity
        self.run_id = run_id
        self.group = group
        self.tags = tuple(tags or ())
        self.config = dict(config or {})
        self._run: Any | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("WANDB_MODE must be online, offline, or disabled")
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is enabled but wandb is not installed"
            ) from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(self.output_dir))
        run_id = (
            self.run_id
            or os.environ.get("WANDB_RUN_ID")
            or os.environ.get("RUNTIME_ID")
            or self._saved_run_id()
        )
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            id=run_id,
            resume="allow" if run_id else None,
            mode=self.mode,
            dir=str(self.output_dir),
            group=self.group,
            tags=list(self.tags) or None,
            config=self.config,
        )
        self.run_id = str(self._run.id)
        self._write_metadata()

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        if self._run is not None:
            self._run.log(dict(metrics), step=int(step))

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "wandb_config.json"

    def _saved_run_id(self) -> str | None:
        if not self.metadata_path.is_file():
            return None
        value = json.loads(self.metadata_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("wandb_config.json must contain an object")
        saved = value.get("run_id")
        return str(saved) if saved else None

    def _write_metadata(self) -> None:
        payload = {
            "project": self.project,
            "run_id": self.run_id or "",
            "name": self.name,
            "mode": self.mode,
        }
        temporary = self.metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.metadata_path)


__all__ = ["WandbRunLogger"]
