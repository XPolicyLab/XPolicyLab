from __future__ import annotations

import pytest
import torch
from torch import nn

from worldscape_policy.training.freezing import FreezePolicy, FreezeRule
from worldscape_policy.training.objective import (
    ActionFlowLoss,
    AlignmentLoss,
    CompositeObjective,
    ObjectiveInputs,
    PlanningCELoss,
    SemanticForcingLoss,
    VideoFlowLoss,
)
from worldscape_policy.training.prompt_schedule import PromptSchedule, Stage


def test_video_flow_matches_legacy_reduction() -> None:
    prediction = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    target = torch.zeros_like(prediction)
    weight = torch.tensor([[1.0, 3.0]])

    result = VideoFlowLoss()(prediction, target, weight)
    legacy = (
        (prediction.float() - target.float()).square().mean(dim=(1, 3, 4))
        * weight
    ).mean()

    torch.testing.assert_close(result.loss, legacy)
    assert set(result.metrics) == {"loss", "unweighted_mse"}


def test_action_flow_matches_masked_legacy_reduction() -> None:
    prediction = torch.tensor(
        [[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]], [[8.0, 8.0, 8.0], [8.0, 8.0, 8.0]]]
    )
    target = torch.zeros_like(prediction)
    action_mask = torch.ones_like(prediction)
    dim_mask = torch.tensor([1.0, 1.0, 0.0])
    has_real_action = torch.tensor([True, False])
    weight = torch.tensor([[1.0, 2.0], [1.0, 1.0]])

    result = ActionFlowLoss()(
        prediction,
        target,
        weight=weight,
        mask=action_mask,
        dim_mask=dim_mask,
        has_real_action=has_real_action,
    )
    combined = action_mask * dim_mask
    error = (prediction - target).square() * combined
    error = error * has_real_action.reshape(2, 1, 1)
    legacy = (error.mean(dim=2) * weight).mean()

    torch.testing.assert_close(result.loss, legacy)
    assert result.metrics["guard_hit"].item() == 0


def test_action_flow_sanitizes_and_guards_outliers() -> None:
    nonfinite = ActionFlowLoss(guard_threshold=10.0)(
        torch.tensor([[[float("nan"), 1.0]]]),
        torch.zeros(1, 1, 2),
    )
    assert torch.isfinite(nonfinite.loss)
    assert nonfinite.metrics["nonfinite_fraction"].item() == 0.5

    guarded = ActionFlowLoss(guard_threshold=1.0)(
        torch.full((1, 1, 2), 10.0),
        torch.zeros(1, 1, 2),
    )
    assert guarded.loss.item() == 0
    assert guarded.skipped
    assert guarded.metrics["guard_hit"].item() == 1


def test_alignment_and_semantic_forcing_have_teacher_only_gradient() -> None:
    prediction = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]]], requires_grad=True
    )
    target = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0]]], requires_grad=True
    )
    alignment = AlignmentLoss(cosine_weight=1.0, mse_weight=0.5)(
        prediction, target
    )
    torch.testing.assert_close(alignment.loss, torch.tensor(1.5))

    semantic = SemanticForcingLoss(cosine_weight=1.0, mse_weight=0.5)(
        prediction, target
    )
    semantic.loss.backward()
    assert prediction.grad is not None
    assert target.grad is None


def test_planning_ce_truncates_and_ignores_labels_like_legacy() -> None:
    logits = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[0, -100]])

    result = PlanningCELoss()(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 2),
        labels.reshape(-1),
        ignore_index=-100,
    )

    torch.testing.assert_close(result.loss, expected)
    assert result.metrics["valid_tokens"].item() == 1


def test_planning_ce_fails_closed_without_supervised_tokens() -> None:
    with pytest.raises(ValueError, match="no supervised tokens"):
        PlanningCELoss()(
            torch.zeros(1, 2, 3),
            torch.full((1, 2), -100),
        )


def test_composite_uses_plan_weights() -> None:
    inputs = ObjectiveInputs(
        video_prediction=torch.ones(1, 1, 1),
        video_target=torch.zeros(1, 1, 1),
        action_prediction=torch.full((1, 1, 1), 2.0),
        action_target=torch.zeros(1, 1, 1),
        semantic_prediction=torch.tensor([[1.0, 0.0]]),
        semantic_target=torch.tensor([[0.0, 1.0]]),
        planning_logits=torch.tensor([[[2.0, 0.0]]]),
        planning_labels=torch.tensor([[0]]),
    )
    objective = CompositeObjective(
        world_flow_weight=2.0,
        semantic_forcing_weight=3.0,
        planning_ce_weight=4.0,
    )

    result = objective(inputs)
    expected = (
        result.terms["action_flow"].loss
        + 2.0 * result.terms["video_flow"].loss
        + 3.0 * result.terms["semantic_forcing"].loss
        + 4.0 * result.terms["planning_ce"].loss
    )

    torch.testing.assert_close(result.loss, expected)
    assert set(result.terms) == {
        "action_flow",
        "video_flow",
        "semantic_forcing",
        "planning_ce",
    }


def test_prompt_schedule_is_batchwise_reproducible_and_boundary_safe() -> None:
    schedule = PromptSchedule(
        [Stage(0.5, 0.0, "first"), Stage(1.0, 1.0, "second")]
    )
    first = schedule.sample(8, 0.49)
    second = schedule.sample(8, 0.5)

    assert not first.mode_mask.any()
    assert second.mode_mask.all()
    assert second.stage.name == "second"

    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    plan = PromptSchedule.plan_default()
    torch.testing.assert_close(
        plan.sample(128, 0.8, generator=generator_a).mode_mask,
        plan.sample(128, 0.8, generator=generator_b).mode_mask,
    )
    legacy_batch = plan.sample(
        128,
        0.8,
        generator=torch.Generator().manual_seed(7),
        per_sample=False,
    )
    assert bool(legacy_batch.mode_mask.all()) or not bool(
        legacy_batch.mode_mask.any()
    )


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.head = nn.Linear(2, 1)


def test_freeze_policy_uses_longest_native_module_path_and_reports() -> None:
    model = _ToyPolicy()
    policy = FreezePolicy(
        [
            FreezeRule("encoder", frozen=True, initialization_source="pretrained"),
            FreezeRule(
                "encoder.1",
                frozen=False,
                optimizer_group="adapter",
                initialization_source="new",
            ),
            FreezeRule("head", frozen=False, optimizer_group="head"),
        ]
    )

    report = policy.apply(model, unused_module_paths=("encoder.1",))

    assert not model.encoder[0].weight.requires_grad
    assert model.encoder[1].weight.requires_grad
    assert model.head.weight.requires_grad
    assert "encoder.0.weight" in report.frozen_names
    assert report.optimizer_groups["adapter"] == (
        "encoder.1.weight",
        "encoder.1.bias",
    )
    assert report.unused_trainable_names == (
        "encoder.1.weight",
        "encoder.1.bias",
    )
    assert report.as_dict()["initialization_sources"]["encoder.1.weight"] == "new"


def test_freeze_policy_reports_non_strict_missing_paths() -> None:
    report = FreezePolicy(
        [FreezeRule("missing", frozen=True)], strict=False
    ).apply(_ToyPolicy())

    assert report.unmatched_rules == ("missing",)
