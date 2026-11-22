"""Physical expert layouts and logical video-action causal visibility."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class InterleaveLayout:
    """IVAVA chunks with physical ``[I,V0,V1|A0,A1]`` storage."""

    num_chunks: int
    condition_video_tokens: int
    video_tokens_per_chunk: int
    action_tokens_per_chunk: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_chunks", self.num_chunks),
            ("condition_video_tokens", self.condition_video_tokens),
            ("video_tokens_per_chunk", self.video_tokens_per_chunk),
            ("action_tokens_per_chunk", self.action_tokens_per_chunk),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def num_video_tokens(self) -> int:
        return (
            self.condition_video_tokens + self.num_chunks * self.video_tokens_per_chunk
        )

    @property
    def num_action_tokens(self) -> int:
        return self.num_chunks * self.action_tokens_per_chunk

    @property
    def sequence_length(self) -> int:
        return self.num_video_tokens + self.num_action_tokens

    def logical_to_physical(self, device: torch.device | None = None) -> torch.Tensor:
        """Gather physical storage into ``I,A0,V0,A1,V1`` logical order."""

        parts = [torch.arange(0, self.condition_video_tokens)]
        action_base = self.num_video_tokens
        for chunk in range(self.num_chunks):
            v0 = self.condition_video_tokens + chunk * self.video_tokens_per_chunk
            a0 = action_base + chunk * self.action_tokens_per_chunk
            parts.append(torch.arange(a0, a0 + self.action_tokens_per_chunk))
            parts.append(torch.arange(v0, v0 + self.video_tokens_per_chunk))
        return torch.cat(parts).to(device=device)

    def physical_to_logical(self, device: torch.device | None = None) -> torch.Tensor:
        permutation = self.logical_to_physical(device=device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            permutation.numel(), device=permutation.device
        )
        return inverse

    def logical_ranks(self, device: torch.device | None = None) -> torch.Tensor:
        ranks = [torch.zeros(self.condition_video_tokens, dtype=torch.long)]
        for chunk in range(self.num_chunks):
            ranks.append(
                torch.full(
                    (self.action_tokens_per_chunk,), 2 * chunk + 1, dtype=torch.long
                )
            )
            ranks.append(
                torch.full(
                    (self.video_tokens_per_chunk,), 2 * chunk + 2, dtype=torch.long
                )
            )
        return torch.cat(ranks).to(device=device)

    def attention_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """Return bool `[S,S]`, indexed as `[query,key]`, True means visible."""

        ranks = self.logical_ranks(device=device)
        return ranks[:, None] >= ranks[None, :]

    def flex_block_mask(self, device: torch.device):
        return _cached_flex_block_mask(self, device.type, device.index)

    def pack(self, video: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if video.shape[-2] != self.num_video_tokens:
            raise ValueError(
                f"video token length {video.shape[-2]} != {self.num_video_tokens}"
            )
        if action.shape[-2] != self.num_action_tokens:
            raise ValueError(
                f"action token length {action.shape[-2]} != {self.num_action_tokens}"
            )
        physical = torch.cat((video, action), dim=-2)
        return physical.index_select(
            -2, self.logical_to_physical(device=physical.device)
        )

    def unpack(self, logical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logical.shape[-2] != self.sequence_length:
            raise ValueError(
                f"logical token length {logical.shape[-2]} != {self.sequence_length}"
            )
        physical = logical.index_select(
            -2, self.physical_to_logical(device=logical.device)
        )
        return (
            physical[..., : self.num_video_tokens, :],
            physical[..., self.num_video_tokens :, :],
        )


@dataclass(frozen=True)
class ActionStageLayout:
    """Prefix-only layout ending at one active Action stage.

    Physical expert storage contains only the tokens that can affect the
    requested Action stage:

    - A0: video ``[I]``, action ``[A0]``; logical ``I,A0``.
    - A1: video ``[I,V0]``, action ``[A0,A1]``; logical ``I,A0,V0,A1``.

    ``num_chunks`` remains the model's full planning-window chunk count.  The
    active chunk selects how much of that window is present in this layout.
    """

    num_chunks: int
    active_action_chunk: int
    condition_video_tokens: int
    video_tokens_per_chunk: int
    action_tokens_per_chunk: int

    def __post_init__(self) -> None:
        for name, value in (
            ("num_chunks", self.num_chunks),
            ("condition_video_tokens", self.condition_video_tokens),
            ("video_tokens_per_chunk", self.video_tokens_per_chunk),
            ("action_tokens_per_chunk", self.action_tokens_per_chunk),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0 <= self.active_action_chunk < self.num_chunks:
            raise ValueError(
                "active_action_chunk must be inside the planning window, got "
                f"{self.active_action_chunk} for {self.num_chunks} chunks"
            )

    @property
    def active_rank(self) -> int:
        return 2 * self.active_action_chunk + 1

    @property
    def num_video_tokens(self) -> int:
        return (
            self.condition_video_tokens
            + self.active_action_chunk * self.video_tokens_per_chunk
        )

    @property
    def num_action_tokens(self) -> int:
        return (self.active_action_chunk + 1) * self.action_tokens_per_chunk

    @property
    def sequence_length(self) -> int:
        return self.num_video_tokens + self.num_action_tokens

    def logical_to_physical(self, device: torch.device | None = None) -> torch.Tensor:
        parts = [torch.arange(0, self.condition_video_tokens)]
        action_base = self.num_video_tokens
        for chunk in range(self.active_action_chunk + 1):
            action_start = action_base + chunk * self.action_tokens_per_chunk
            parts.append(
                torch.arange(action_start, action_start + self.action_tokens_per_chunk)
            )
            if chunk < self.active_action_chunk:
                video_start = (
                    self.condition_video_tokens + chunk * self.video_tokens_per_chunk
                )
                parts.append(
                    torch.arange(video_start, video_start + self.video_tokens_per_chunk)
                )
        return torch.cat(parts).to(device=device)

    def physical_to_logical(self, device: torch.device | None = None) -> torch.Tensor:
        permutation = self.logical_to_physical(device=device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            permutation.numel(), device=permutation.device
        )
        return inverse

    def logical_ranks(self, device: torch.device | None = None) -> torch.Tensor:
        ranks = [torch.zeros(self.condition_video_tokens, dtype=torch.long)]
        for chunk in range(self.active_action_chunk + 1):
            ranks.append(
                torch.full(
                    (self.action_tokens_per_chunk,), 2 * chunk + 1, dtype=torch.long
                )
            )
            if chunk < self.active_action_chunk:
                ranks.append(
                    torch.full(
                        (self.video_tokens_per_chunk,), 2 * chunk + 2, dtype=torch.long
                    )
                )
        return torch.cat(ranks).to(device=device)

    def attention_mask(self, device: torch.device | None = None) -> torch.Tensor:
        ranks = self.logical_ranks(device=device)
        return ranks[:, None] >= ranks[None, :]

    def flex_block_mask(self, device: torch.device):
        return _cached_flex_block_mask(self, device.type, device.index)

    def pack(self, video: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if video.shape[-2] != self.num_video_tokens:
            raise ValueError(
                f"video token length {video.shape[-2]} != {self.num_video_tokens}"
            )
        if action.shape[-2] != self.num_action_tokens:
            raise ValueError(
                f"action token length {action.shape[-2]} != {self.num_action_tokens}"
            )
        physical = torch.cat((video, action), dim=-2)
        return physical.index_select(
            -2, self.logical_to_physical(device=physical.device)
        )

    def unpack(self, logical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logical.shape[-2] != self.sequence_length:
            raise ValueError(
                f"logical token length {logical.shape[-2]} != {self.sequence_length}"
            )
        physical = logical.index_select(
            -2, self.physical_to_logical(device=logical.device)
        )
        return (
            physical[..., : self.num_video_tokens, :],
            physical[..., self.num_video_tokens :, :],
        )


@dataclass(frozen=True)
class ParallelTeacherForcingLayout:
    """Packed clean/noisy teacher forcing for ``I,A0,V0,...``.

    Each target stage has a noisy prediction branch.  Every stage except the
    final ``V`` stage also has a clean branch that becomes context for later
    predictions.  The final clean video would have no consumer and is omitted.

    Physical expert storage is kept contiguous per modality:

    - Video: ``[I,V0_clean,V0_noisy,...,V_last_noisy]``
    - Action baseline: ``[A0_clean,A0_noisy,...,A_last_clean,A_last_noisy]``
    - Action robust A0: ``[A0_clean,A0_robust,A0_noisy,A1_clean,A1_noisy]``

    Attention packs these tensors into stage-paired logical order.  A noisy
    branch sees only clean earlier stages and itself; a clean branch sees only
    clean history through its own stage.  This makes one packed forward exactly
    equivalent to summing the independent clean-prefix/noisy-active forwards.
    """

    num_chunks: int
    condition_video_tokens: int
    video_tokens_per_chunk: int
    action_tokens_per_chunk: int
    robust_a0: bool = False

    CONDITION = 0
    CLEAN = 1
    NOISY = 2
    ROBUST = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("num_chunks", self.num_chunks),
            ("condition_video_tokens", self.condition_video_tokens),
            ("video_tokens_per_chunk", self.video_tokens_per_chunk),
            ("action_tokens_per_chunk", self.action_tokens_per_chunk),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.robust_a0 and self.num_chunks != 2:
            raise ValueError("robust_a0 requires the two-chunk I-A0-V0-A1-V1 layout")

    @property
    def standard_video_tokens(self) -> int:
        return (
            self.condition_video_tokens + self.num_chunks * self.video_tokens_per_chunk
        )

    @property
    def standard_action_tokens(self) -> int:
        return self.num_chunks * self.action_tokens_per_chunk

    @property
    def num_video_tokens(self) -> int:
        # Every video target is noisy; all but the final one also appear clean.
        return (
            self.condition_video_tokens
            + (2 * self.num_chunks - 1) * self.video_tokens_per_chunk
        )

    @property
    def num_action_tokens(self) -> int:
        copies = 2 * self.num_chunks + int(self.robust_a0)
        return copies * self.action_tokens_per_chunk

    @property
    def sequence_length(self) -> int:
        return self.num_video_tokens + self.num_action_tokens

    def _video_blocks(self) -> dict[tuple[int, int], tuple[int, int]]:
        blocks: dict[tuple[int, int], tuple[int, int]] = {
            (self.CONDITION, -1): (0, self.condition_video_tokens)
        }
        offset = self.condition_video_tokens
        for chunk in range(self.num_chunks):
            video_stage = 2 * chunk + 1
            if chunk < self.num_chunks - 1:
                blocks[(self.CLEAN, video_stage)] = (
                    offset,
                    offset + self.video_tokens_per_chunk,
                )
                offset += self.video_tokens_per_chunk
            blocks[(self.NOISY, video_stage)] = (
                offset,
                offset + self.video_tokens_per_chunk,
            )
            offset += self.video_tokens_per_chunk
        if offset != self.num_video_tokens:
            raise AssertionError("parallel video layout length mismatch")
        return blocks

    def _action_blocks(self) -> dict[tuple[int, int], tuple[int, int]]:
        blocks: dict[tuple[int, int], tuple[int, int]] = {}
        offset = self.num_video_tokens
        for chunk in range(self.num_chunks):
            action_stage = 2 * chunk
            roles = [self.CLEAN]
            if self.robust_a0 and chunk == 0:
                roles.append(self.ROBUST)
            roles.append(self.NOISY)
            for role in roles:
                blocks[(role, action_stage)] = (
                    offset,
                    offset + self.action_tokens_per_chunk,
                )
                offset += self.action_tokens_per_chunk
        if offset != self.sequence_length:
            raise AssertionError("parallel action layout length mismatch")
        return blocks

    def logical_to_physical(self, device: torch.device | None = None) -> torch.Tensor:
        video = self._video_blocks()
        action = self._action_blocks()
        parts = [torch.arange(*video[(self.CONDITION, -1)])]
        for chunk in range(self.num_chunks):
            action_stage = 2 * chunk
            video_stage = action_stage + 1
            roles = [self.CLEAN]
            if self.robust_a0 and chunk == 0:
                roles.append(self.ROBUST)
            roles.append(self.NOISY)
            for role in roles:
                parts.append(torch.arange(*action[(role, action_stage)]))
            if chunk < self.num_chunks - 1:
                parts.append(torch.arange(*video[(self.CLEAN, video_stage)]))
            parts.append(torch.arange(*video[(self.NOISY, video_stage)]))
        return torch.cat(parts).to(device=device)

    def physical_to_logical(self, device: torch.device | None = None) -> torch.Tensor:
        permutation = self.logical_to_physical(device=device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(
            permutation.numel(), device=permutation.device
        )
        return inverse

    def logical_roles(self, device: torch.device | None = None) -> torch.Tensor:
        roles = [
            torch.full((self.condition_video_tokens,), self.CONDITION, dtype=torch.long)
        ]
        for chunk in range(self.num_chunks):
            roles.append(
                torch.full(
                    (self.action_tokens_per_chunk,), self.CLEAN, dtype=torch.long
                )
            )
            if self.robust_a0 and chunk == 0:
                roles.append(
                    torch.full(
                        (self.action_tokens_per_chunk,), self.ROBUST, dtype=torch.long
                    )
                )
            roles.append(
                torch.full(
                    (self.action_tokens_per_chunk,), self.NOISY, dtype=torch.long
                )
            )
            if chunk < self.num_chunks - 1:
                roles.append(
                    torch.full(
                        (self.video_tokens_per_chunk,), self.CLEAN, dtype=torch.long
                    )
                )
            roles.append(
                torch.full((self.video_tokens_per_chunk,), self.NOISY, dtype=torch.long)
            )
        return torch.cat(roles).to(device=device)

    def logical_stages(self, device: torch.device | None = None) -> torch.Tensor:
        stages = [torch.full((self.condition_video_tokens,), -1, dtype=torch.long)]
        for chunk in range(self.num_chunks):
            action_stage = 2 * chunk
            video_stage = action_stage + 1
            stages.append(
                torch.full(
                    (self.action_tokens_per_chunk,), action_stage, dtype=torch.long
                )
            )
            if self.robust_a0 and chunk == 0:
                stages.append(
                    torch.full(
                        (self.action_tokens_per_chunk,), action_stage, dtype=torch.long
                    )
                )
            stages.append(
                torch.full(
                    (self.action_tokens_per_chunk,), action_stage, dtype=torch.long
                )
            )
            if chunk < self.num_chunks - 1:
                stages.append(
                    torch.full(
                        (self.video_tokens_per_chunk,), video_stage, dtype=torch.long
                    )
                )
            stages.append(
                torch.full(
                    (self.video_tokens_per_chunk,), video_stage, dtype=torch.long
                )
            )
        return torch.cat(stages).to(device=device)

    def attention_mask(self, device: torch.device | None = None) -> torch.Tensor:
        roles = self.logical_roles(device=device)
        stages = self.logical_stages(device=device)
        query_role = roles[:, None]
        key_role = roles[None, :]
        query_stage = stages[:, None]
        key_stage = stages[None, :]
        condition = key_role == self.CONDITION
        clean_chain = (key_role == self.CLEAN) & (
            ((query_role == self.CLEAN) & (key_stage <= query_stage))
            | ((query_role == self.NOISY) & (key_stage < query_stage))
        )
        noisy_self = (
            (query_role == self.NOISY)
            & (key_role == self.NOISY)
            & (query_stage == key_stage)
        )
        if not self.robust_a0:
            return condition | clean_chain | noisy_self
        # Only the noisy A1 prediction consumes the corrupted history view.
        # Clean A1 must retain the clean chain because it is V1's physical
        # teacher-forced context.  This prevents corruption from reaching
        # either V0 directly or V1 indirectly through clean A1.
        noisy_a1_query = (query_role == self.NOISY) & (query_stage == 2)
        clean_a0_key = (key_role == self.CLEAN) & (key_stage == 0)
        clean_chain = clean_chain & ~(noisy_a1_query & clean_a0_key)
        robust_chain = (
            noisy_a1_query & (key_role == self.ROBUST) & (key_stage == 0)
        )
        robust_self = (
            (query_role == self.ROBUST)
            & (key_role == self.ROBUST)
            & (query_stage == key_stage)
        )
        return condition | clean_chain | robust_chain | robust_self | noisy_self

    def flex_block_mask(self, device: torch.device):
        return _cached_flex_block_mask(self, device.type, device.index)

    def pack(self, video: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if video.shape[-2] != self.num_video_tokens:
            raise ValueError(
                f"video token length {video.shape[-2]} != {self.num_video_tokens}"
            )
        if action.shape[-2] != self.num_action_tokens:
            raise ValueError(
                f"action token length {action.shape[-2]} != {self.num_action_tokens}"
            )
        physical = torch.cat((video, action), dim=-2)
        return physical.index_select(-2, self.logical_to_physical(physical.device))

    def unpack(self, logical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logical.shape[-2] != self.sequence_length:
            raise ValueError(
                f"logical token length {logical.shape[-2]} != {self.sequence_length}"
            )
        physical = logical.index_select(-2, self.physical_to_logical(logical.device))
        return (
            physical[..., : self.num_video_tokens, :],
            physical[..., self.num_video_tokens :, :],
        )

    def expand_video(self, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """Build physical ``[I,V0c,V0n,...,Vlast_n]`` storage."""

        if clean.shape != noisy.shape or clean.shape[-2] != self.standard_video_tokens:
            raise ValueError(
                "clean/noisy video must share the standard [I,V...] length"
            )
        condition = clean[..., : self.condition_video_tokens, :]
        clean_future = clean[..., self.condition_video_tokens :, :]
        noisy_future = noisy[..., self.condition_video_tokens :, :]
        clean_chunks = clean_future.split(self.video_tokens_per_chunk, dim=-2)
        noisy_chunks = noisy_future.split(self.video_tokens_per_chunk, dim=-2)
        parts = [condition]
        for chunk in range(self.num_chunks):
            if chunk < self.num_chunks - 1:
                parts.append(clean_chunks[chunk])
            parts.append(noisy_chunks[chunk])
        return torch.cat(parts, dim=-2)

    def expand_action(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        robust_a0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build physical clean/robust/noisy Action storage.

        The compatibility layout remains ``[A0c,A0n,A1c,A1n]``.  The robust
        layout is ``[A0c,A0r,A0n,A1c,A1n]``.  Position/timestep/FiLM callers
        may omit ``robust_a0`` to duplicate clean A0 metadata; Action-value
        callers provide the explicitly corrupted A0 view.
        """

        if clean.shape != noisy.shape or clean.shape[-2] != self.standard_action_tokens:
            raise ValueError(
                "clean/noisy action must share the standard [A0,A1,...] length"
            )
        clean_chunks = clean.split(self.action_tokens_per_chunk, dim=-2)
        noisy_chunks = noisy.split(self.action_tokens_per_chunk, dim=-2)
        if self.robust_a0:
            if robust_a0 is None:
                robust_a0 = clean_chunks[0]
            if tuple(robust_a0.shape) != tuple(clean_chunks[0].shape):
                raise ValueError(
                    "robust_a0 must match one clean Action chunk, got "
                    f"{tuple(robust_a0.shape)} != {tuple(clean_chunks[0].shape)}"
                )
        elif robust_a0 is not None:
            raise ValueError("robust_a0 values require layout.robust_a0=true")
        parts = []
        for chunk, (clean_chunk, noisy_chunk) in enumerate(
            zip(clean_chunks, noisy_chunks)
        ):
            parts.append(clean_chunk)
            if self.robust_a0 and chunk == 0:
                if robust_a0 is None:
                    raise AssertionError("robust A0 was not resolved")
                parts.append(robust_a0)
            parts.append(noisy_chunk)
        return torch.cat(parts, dim=-2)

    def noisy_video(self, physical: torch.Tensor) -> torch.Tensor:
        """Select noisy future video chunks, excluding condition tokens."""

        if physical.shape[-2] != self.num_video_tokens:
            raise ValueError("physical video length does not match packed layout")
        blocks = self._video_blocks()
        return torch.cat(
            [
                physical[..., slice(*blocks[(self.NOISY, 2 * chunk + 1)]), :]
                for chunk in range(self.num_chunks)
            ],
            dim=-2,
        )

    def noisy_action(self, physical: torch.Tensor) -> torch.Tensor:
        """Select noisy action prediction chunks in temporal order."""

        if physical.shape[-2] != self.num_action_tokens:
            raise ValueError("physical action length does not match packed layout")
        # Action block offsets include the preceding physical video tokens.
        blocks = self._action_blocks()
        return torch.cat(
            [
                physical[
                    ...,
                    slice(
                        blocks[(self.NOISY, 2 * chunk)][0] - self.num_video_tokens,
                        blocks[(self.NOISY, 2 * chunk)][1] - self.num_video_tokens,
                    ),
                    :,
                ]
                for chunk in range(self.num_chunks)
            ],
            dim=-2,
        )


@dataclass(frozen=True)
class VideoOnlyParallelTeacherForcingLayout(ParallelTeacherForcingLayout):
    """Packed ``I,V0,...,V{E-1}`` forcing with isolated dummy Action tokens.

    Storage deliberately remains shape-compatible with
    :class:`ParallelTeacherForcingLayout`, so video-only Stage-1 checkpoints
    use the exact same VAP parameter/schema as paired IVAVA Stage 2.  Action
    tensors are placeholders used only to keep DMuon/FSDP module traversal
    stable.  They are never visible to a Video query and receive no loss.

    The effective video factorization is therefore strictly::

        I -> V0; I + clean(V0) -> V1; ...; I + clean(V<j) -> Vj

    rather than pretending that a missing action is a physical zero action.
    """

    VIDEO = 0
    ACTION = 1

    def logical_modalities(self, device: torch.device | None = None) -> torch.Tensor:
        modalities = [
            torch.full((self.condition_video_tokens,), self.VIDEO, dtype=torch.long)
        ]
        for chunk in range(self.num_chunks):
            modalities.extend(
                [
                    torch.full(
                        (self.action_tokens_per_chunk,),
                        self.ACTION,
                        dtype=torch.long,
                    ),
                    torch.full(
                        (self.action_tokens_per_chunk,),
                        self.ACTION,
                        dtype=torch.long,
                    ),
                ]
            )
            if chunk < self.num_chunks - 1:
                modalities.append(
                    torch.full(
                        (self.video_tokens_per_chunk,),
                        self.VIDEO,
                        dtype=torch.long,
                    )
                )
            modalities.append(
                torch.full(
                    (self.video_tokens_per_chunk,),
                    self.VIDEO,
                    dtype=torch.long,
                )
            )
        return torch.cat(modalities).to(device=device)

    def attention_mask(self, device: torch.device | None = None) -> torch.Tensor:
        roles = self.logical_roles(device=device)
        stages = self.logical_stages(device=device)
        modalities = self.logical_modalities(device=device)
        query_role = roles[:, None]
        key_role = roles[None, :]
        query_stage = stages[:, None]
        key_stage = stages[None, :]
        same_modality = modalities[:, None] == modalities[None, :]
        condition = key_role == self.CONDITION
        clean_chain = (
            same_modality
            & (key_role == self.CLEAN)
            & (
                ((query_role == self.CLEAN) & (key_stage <= query_stage))
                | ((query_role == self.NOISY) & (key_stage < query_stage))
            )
        )
        noisy_self = (
            same_modality
            & (query_role == self.NOISY)
            & (key_role == self.NOISY)
            & (query_stage == key_stage)
        )
        # The initial image is valid context for both streams. Dummy Action
        # outputs are later tied to the loss with coefficient zero solely to
        # keep distributed optimizer hooks deterministic.
        return condition | clean_chain | noisy_self


AttentionLayout = (
    ActionStageLayout
    | InterleaveLayout
    | ParallelTeacherForcingLayout
    | VideoOnlyParallelTeacherForcingLayout
)


@lru_cache(maxsize=32)
def _cached_flex_block_mask(
    layout: AttentionLayout, device_type: str, device_index: int | None
):
    """Compile and cache a block-sparse mask without materializing ``S×S``."""

    from torch.nn.attention.flex_attention import create_block_mask

    device = torch.device(device_type, device_index)
    if isinstance(layout, (ActionStageLayout, InterleaveLayout)):
        ranks = layout.logical_ranks(device=device)

        def mask_mod(batch, head, query_index, key_index):
            del batch, head
            return ranks[query_index] >= ranks[key_index]

    elif isinstance(layout, VideoOnlyParallelTeacherForcingLayout):
        roles = layout.logical_roles(device=device)
        stages = layout.logical_stages(device=device)
        modalities = layout.logical_modalities(device=device)

        def mask_mod(batch, head, query_index, key_index):
            del batch, head
            query_role = roles[query_index]
            key_role = roles[key_index]
            query_stage = stages[query_index]
            key_stage = stages[key_index]
            same_modality = modalities[query_index] == modalities[key_index]
            condition = key_role == layout.CONDITION
            clean_chain = (
                same_modality
                & (key_role == layout.CLEAN)
                & (
                    ((query_role == layout.CLEAN) & (key_stage <= query_stage))
                    | ((query_role == layout.NOISY) & (key_stage < query_stage))
                )
            )
            noisy_self = (
                same_modality
                & (query_role == layout.NOISY)
                & (key_role == layout.NOISY)
                & (query_stage == key_stage)
            )
            return condition | clean_chain | noisy_self

    else:
        roles = layout.logical_roles(device=device)
        stages = layout.logical_stages(device=device)

        if layout.robust_a0:

            def mask_mod(batch, head, query_index, key_index):
                del batch, head
                query_role = roles[query_index]
                key_role = roles[key_index]
                query_stage = stages[query_index]
                key_stage = stages[key_index]
                condition = key_role == layout.CONDITION
                clean_chain = (key_role == layout.CLEAN) & (
                    ((query_role == layout.CLEAN) & (key_stage <= query_stage))
                    | ((query_role == layout.NOISY) & (key_stage < query_stage))
                )
                noisy_a1_query = (query_role == layout.NOISY) & (query_stage == 2)
                clean_a0_key = (key_role == layout.CLEAN) & (key_stage == 0)
                clean_chain = clean_chain & ~(noisy_a1_query & clean_a0_key)
                robust_chain = (
                    noisy_a1_query
                    & (key_role == layout.ROBUST)
                    & (key_stage == 0)
                )
                robust_self = (
                    (query_role == layout.ROBUST)
                    & (key_role == layout.ROBUST)
                    & (query_stage == key_stage)
                )
                noisy_self = (
                    (query_role == layout.NOISY)
                    & (key_role == layout.NOISY)
                    & (query_stage == key_stage)
                )
                return condition | clean_chain | robust_chain | robust_self | noisy_self

        else:

            def mask_mod(batch, head, query_index, key_index):
                del batch, head
                query_role = roles[query_index]
                key_role = roles[key_index]
                query_stage = stages[query_index]
                key_stage = stages[key_index]
                condition = key_role == layout.CONDITION
                clean_chain = (key_role == layout.CLEAN) & (
                    ((query_role == layout.CLEAN) & (key_stage <= query_stage))
                    | ((query_role == layout.NOISY) & (key_stage < query_stage))
                )
                noisy_self = (
                    (query_role == layout.NOISY)
                    & (key_role == layout.NOISY)
                    & (query_stage == key_stage)
                )
                return condition | clean_chain | noisy_self

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=layout.sequence_length,
        KV_LEN=layout.sequence_length,
        device=device,
        BLOCK_SIZE=128,
        _compile=True,
    )


__all__ = [
    "ActionStageLayout",
    "AttentionLayout",
    "InterleaveLayout",
    "ParallelTeacherForcingLayout",
    "VideoOnlyParallelTeacherForcingLayout",
]
