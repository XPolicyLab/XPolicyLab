"""Opt-in critical-window sampling; evaluation always retains its fixed windows."""

from __future__ import annotations

from .critical_phase_sampling import CriticalPhaseIndex, row_key
from .video_event_dataset import VideoEventDataset


class CriticalPhaseVideoEventDataset(VideoEventDataset):
    def __init__(
        self,
        *,
        critical_phase_index=None,
        critical_phase_weight=1.0,
        critical_phase_allow_unreviewed=False,
        critical_phase_prefix=16,
        critical_phase_mode="binary",
        critical_phase_mixture=0.5,
        critical_phase_max_density_ratio=3.0,
        **kwargs,
    ):
        self._critical_phase = None
        super().__init__(**kwargs)
        # runtime's heldout_action evaluator clones data.train and changes its
        # metadata/is_training_set. Do not bind a training index to that clone.
        if not self.is_training_set:
            return
        if critical_phase_index is None:
            if critical_phase_weight != 1.0 or critical_phase_mode != "binary":
                raise ValueError("Critical phase sampling requires critical_phase_index")
            return
        if (
            self.segment_sampling_mode != "step_uniform"
            or self.skip_padding_as_possible
            or kwargs.get("val_set_proportion", 0.05) > 1e-6
            or len(self.video_path_columns) != 1
            or self.video_hdf5_path_column
        ):
            raise ValueError(
                "Critical phase requires step_uniform, explicit split, single video, and no padding retries"
            )
        if "fps" not in self.metadata or not self.metadata.fps.eq(25).all():
            raise ValueError("Critical phase requires verified 25 Hz metadata")
        if not 1 <= int(critical_phase_prefix) <= self.action_horizon:
            raise ValueError("critical_phase_prefix must be within action horizon")
        contract = dict(
            action_horizon=self.action_horizon,
            stride=self.global_sample_stride,
            action_offset=self.action_time_offset,
            prefix=int(critical_phase_prefix),
            fps=25,
        )
        self._critical_phase = CriticalPhaseIndex(
            critical_phase_index,
            metadata_paths=self.metadata_paths,
            contract=contract,
            weight=critical_phase_weight,
            allow_unreviewed=critical_phase_allow_unreviewed,
            mode=critical_phase_mode,
            mixture=critical_phase_mixture,
            max_density_ratio=critical_phase_max_density_ratio,
        )
        keys = [
            row_key(
                row[self.parquet_path_column],
                row[self.video_path_columns[0]],
                self.table_start_steps[i],
                self.table_start_steps[i] + self.trajectory_lengths[i],
                self.video_start_steps[i],
                self.video_start_steps[i] + self.trajectory_lengths[i],
                self._get_instruction(row),
            )
            for i, (_, row) in enumerate(self.metadata.iterrows())
        ]
        self._critical_phase_rows = self._critical_phase.bind(keys, self.trajectory_lengths)

    def _resolve_step_index_uniform(self, idx):
        return super()._resolve_step_index(idx)

    def _resolve_step_index(self, idx):
        row, start, absolute = self._resolve_step_index_uniform(idx)
        if self._critical_phase is None:
            return row, start, absolute
        logical_idx = int(idx) if idx >= 0 else int(idx) + self.total_steps
        start = self._critical_phase.choose(
            self._critical_phase_rows[row],
            start,
            seed=self.seed,
            epoch=self.sampling_epoch,
            sample_index=logical_idx,
        )
        return row, start, int(self.trajectory_start_steps[row] + start)

    def __getitem__(self, idx):
        if self._critical_phase is None:
            return super().__getitem__(idx)
        try:
            return self._get(idx)
        except Exception as error:
            raise RuntimeError(f"Critical phase sample {idx} failed; no random fallback") from error
