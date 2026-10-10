"""Console and Weights & Biases logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(*, rank: int = 0, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mmabc")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(f"[%(asctime)s r{rank}] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    # Non-zero ranks only surface warnings, so eight nodes do not multiply every
    # progress line by 64.
    logger.setLevel(level if rank == 0 else logging.WARNING)
    logger.propagate = False
    return logger


class MetricLogger:
    def __init__(
        self,
        *,
        log_every: int,
        wandb_project: str | None,
        run_name: str,
        enabled: bool,
        config: dict | None = None,
        wandb_entity: str | None = None,
        id_file: Path | None = None,
    ) -> None:
        self.log_every = log_every
        self.enabled = enabled
        self._logger = logging.getLogger("mmabc")
        self._wandb = None
        if enabled and wandb_project:
            try:
                import wandb

                # A stable id per checkpoint root, so an auto-resumed run keeps
                # appending to the same W&B run instead of starting a new one.
                run_id = None
                if id_file is not None:
                    id_file = Path(id_file)
                    run_id = id_file.read_text().strip() if id_file.exists() else wandb.util.generate_id()
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity or None,
                    name=run_name,
                    config=config,
                    id=run_id,
                    resume="allow" if run_id else None,
                    dir=str(id_file.parent) if id_file is not None else None,
                )
                if id_file is not None:
                    id_file.write_text(wandb.run.id)
                self._wandb = wandb
            except Exception as exc:  # pragma: no cover
                self._logger.warning("wandb disabled: %s", exc)

    def log(self, step: int, stats: dict[str, float]) -> None:
        if not self.enabled:
            return
        parts = []
        for key in (
            "loss",
            "loss_action",
            "loss_manip",
            "loss_aux",
            "loss_future",
            "recon_l1_real",
            "grad_norm",
            "s_per_step",
            "t_data_ms",
            "t_compute_ms",
            "t_sync_ms",
            "samples_per_s",
            "aux_active_frac",
            "mem_gb",
        ):
            if key in stats:
                parts.append(f"{key}={stats[key]:.4g}")
        self._logger.info("step %6d  %s", step, "  ".join(parts))
        if self._wandb is not None:
            self._wandb.log(stats, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
