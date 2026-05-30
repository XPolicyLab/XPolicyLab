from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union, List

import torch
from pydantic import Field, BaseModel

from .base import ModalityTransform

ArmSide = Literal["left", "right"]

class BasePadTransform(ModalityTransform):
    """
    Pad state/action features to fixed dimensions with masks.
    """

    arm_state_dim: int = Field(..., description="Per-arm state feature dim.")
    arm_action_dim: int = Field(..., description="Per-arm action feature dim.")

    max_state_dim: int = Field(..., description="Final padded state dim.")
    max_action_dim: int = Field(..., description="Final padded action dim.")

    single_arm_placement: ArmSide = Field(default="right")

    pad_value_state: float = 0.0
    pad_value_action: float = 0.0

    apply_to: list[str] = Field(default_factory=list)

    def _pad_last_dim_with_mask(
        self,
        x: torch.Tensor,
        max_dim: int,
        pad_value: float,
        base_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        dev = x.device

        if base_mask is None:
            base_mask = torch.ones(D, dtype=torch.bool, device=dev)
        else:
            assert base_mask.shape == (D,), f"mask shape mismatch: {base_mask.shape} vs {(D,)}"

        if D == max_dim:
            return x, base_mask
        if D > max_dim:
            return x[:, :max_dim], base_mask[:max_dim]

        pad = torch.full((T, max_dim - D), pad_value, dtype=x.dtype, device=dev)
        mask_pad = torch.zeros(max_dim - D, dtype=torch.bool, device=dev)
        return torch.cat([x, pad], dim=-1), torch.cat([base_mask, mask_pad], dim=0)

    def _build_canonical_layout(
        self,
        x: torch.Tensor,
        per_arm_dim: int,
        pad_value: float,
        kind: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Default behavior: do not change layout, only mark all existing dims as valid.
        """
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        D = x.shape[1]
        mask = torch.ones(D, dtype=torch.bool, device=x.device)
        return x, mask

    def _transform_one(
        self,
        x: torch.Tensor,
        per_arm_dim: int,
        max_dim: int,
        pad_value: float,
        kind: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_canonical, mask_canonical = self._build_canonical_layout(
            x=x,
            per_arm_dim=per_arm_dim,
            pad_value=pad_value,
            kind=kind,
        )
        return self._pad_last_dim_with_mask(
            x=x_canonical,
            max_dim=max_dim,
            pad_value=pad_value,
            base_mask=mask_canonical,
        )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if "state" in data and data["state"] is not None:
            state = data["state"]
            assert isinstance(state, torch.Tensor), f"Expected torch.Tensor for state, got {type(state)}"
            state_out, state_mask = self._transform_one(
                x=state,
                per_arm_dim=self.arm_state_dim,
                max_dim=self.max_state_dim,
                pad_value=self.pad_value_state,
                kind="state",
            )
            data["state"] = state_out
            data["state_mask"] = state_mask

        if "action" in data and data["action"] is not None:
            action = data["action"]
            assert isinstance(action, torch.Tensor), f"Expected torch.Tensor for action, got {type(action)}"
            action_out, action_mask = self._transform_one(
                x=action,
                per_arm_dim=self.arm_action_dim,
                max_dim=self.max_action_dim,
                pad_value=self.pad_value_action,
                kind="action",
            )
            data["action"] = action_out
            data["action_mask"] = action_mask

        return data

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.apply(data)


class BimanualPadTransform(BasePadTransform):
    """
    Convert state/action to a canonical bimanual layout, then pad with masks.
    """

    def _build_canonical_layout(
        self,
        x: torch.Tensor,
        per_arm_dim: int,
        pad_value: float,
        kind: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        dev = x.device

        bimanual_dim = 2 * per_arm_dim

        if D == per_arm_dim:
            base = torch.full((T, bimanual_dim), pad_value, dtype=x.dtype, device=dev)
            mask = torch.zeros(bimanual_dim, dtype=torch.bool, device=dev)

            if self.single_arm_placement == "left":
                base[:, :per_arm_dim] = x
                mask[:per_arm_dim] = True
            else:
                base[:, per_arm_dim:] = x
                mask[per_arm_dim:] = True

            return base, mask

        if D == bimanual_dim:
            mask = torch.ones(bimanual_dim, dtype=torch.bool, device=dev)
            return x, mask

        return super()._build_canonical_layout(
            x=x,
            per_arm_dim=per_arm_dim,
            pad_value=pad_value,
            kind=kind,
        )


class BimanualPadAndGripperPadTransform(BasePadTransform):
    """
    Convert state/action to a canonical bimanual layout with one gripper slot per arm.
    """

    gripper_pad_value_state: float = 0.0
    gripper_pad_value_action: float = 0.0

    def _build_canonical_layout(
        self,
        x: torch.Tensor,
        per_arm_dim: int,
        pad_value: float,
        kind: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 2, f"Expected [T, D], got {tuple(x.shape)}"
        T, D = x.shape
        dev = x.device

        if per_arm_dim < 2:
            raise ValueError(f"per_arm_dim must be >= 2, got {per_arm_dim}")

        gripper_pad_value = (
            self.gripper_pad_value_state if kind == "state" else self.gripper_pad_value_action
        )

        arm_no_g = per_arm_dim - 1
        expect_single = arm_no_g
        expect_dual = 2 * arm_no_g
        canonical_dim = 2 * per_arm_dim

        if D not in (expect_single, expect_dual):
            raise ValueError(
                f"Input dim D={D} mismatch. Expected {expect_single} (single, no gripper) "
                f"or {expect_dual} (dual, no gripper) for per_arm_dim={per_arm_dim}."
            )

        base = torch.full((T, canonical_dim), pad_value, dtype=x.dtype, device=dev)
        mask = torch.zeros(canonical_dim, dtype=torch.bool, device=dev)

        left_gripper_idx = arm_no_g
        right_gripper_idx = per_arm_dim + arm_no_g
        base[:, left_gripper_idx] = gripper_pad_value
        base[:, right_gripper_idx] = gripper_pad_value

        if D == expect_single:
            if self.single_arm_placement == "left":
                base[:, :arm_no_g] = x
                mask[:arm_no_g] = True
            else:
                base[:, per_arm_dim : per_arm_dim + arm_no_g] = x
                mask[per_arm_dim : per_arm_dim + arm_no_g] = True
        else:
            xL = x[:, :arm_no_g]
            xR = x[:, arm_no_g : 2 * arm_no_g]
            base[:, :arm_no_g] = xL
            base[:, per_arm_dim : per_arm_dim + arm_no_g] = xR
            mask[:arm_no_g] = True
            mask[per_arm_dim : per_arm_dim + arm_no_g] = True

        return base, mask