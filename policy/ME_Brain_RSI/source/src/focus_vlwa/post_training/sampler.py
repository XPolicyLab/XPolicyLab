"""Deterministic distributed sample cursors for checkpoint resume."""

import itertools

import torch


def batch_change_resume_cursor(step, anchor_step, old_batch, new_batch, samples_per_rank):
    """Preserve the old epoch cursor, then count updates with the new batch size."""
    if step < anchor_step or anchor_step < 0 or min(old_batch, new_batch) < 1:
        raise ValueError("Invalid batch-change resume anchor")
    old_epoch_batches = samples_per_rank // old_batch
    new_epoch_batches = samples_per_rank // new_batch
    if min(old_epoch_batches, new_epoch_batches) < 1:
        raise ValueError("Each sampler epoch must contain at least one full batch")
    epoch, old_offset = divmod(anchor_step, old_epoch_batches)
    sample_offset = old_offset * old_batch
    remaining_updates = step - anchor_step
    first_epoch_batches = (samples_per_rank - sample_offset) // new_batch
    if remaining_updates < first_epoch_batches:
        return epoch, sample_offset + remaining_updates * new_batch
    extra_epochs, batch_offset = divmod(remaining_updates - first_epoch_batches, new_epoch_batches)
    return epoch + 1 + extra_epochs, batch_offset * new_batch


class _ResumableDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """Skip already-consumed indices without decoding their samples."""

    start_index = 0

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_index, None)
