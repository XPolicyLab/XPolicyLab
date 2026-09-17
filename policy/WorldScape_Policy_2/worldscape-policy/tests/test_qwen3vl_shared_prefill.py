from types import SimpleNamespace

import pytest
import torch

from worldscape_policy.conditioning.prompt_format import render_instruction_template
from worldscape_policy.conditioning.vlm.qwen3vl import (
    QwenPlanningEncoder,
)


def test_cot_prompt_does_not_duplicate_terminal_punctuation():
    template = "Instructions: {task}. Predict the next subtask."

    assert render_instruction_template(template, "Fold the shirt.") == (
        "Instructions: Fold the shirt. Predict the next subtask."
    )
    assert render_instruction_template(template, "Fold the shirt") == (
        "Instructions: Fold the shirt. Predict the next subtask."
    )


class _FakeTokenizer:
    def __init__(self):
        self.encoded = []
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        self.encoded.append(text)
        return [7, 8]


class _FakeVLM(torch.nn.Module):
    def __init__(self, hidden_states, scores):
        super().__init__()
        self.hidden_states = hidden_states
        self.scores = scores
        self.processor = SimpleNamespace(tokenizer=_FakeTokenizer())
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(
            sequences=torch.tensor([[10, 11, 12, 13]]),
            hidden_states=self.hidden_states,
            scores=self.scores,
        )


class _TrainableFakeVLM(_FakeVLM):
    def __init__(self, hidden_states, scores):
        super().__init__(hidden_states, scores)
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, **kwargs):
        del kwargs
        batch, sequence_length = input_ids.shape
        hidden_dim = self.hidden_states[0][-1].shape[-1]
        logits = self.logit_scale * torch.ones(
            batch, sequence_length, 5, device=input_ids.device
        )
        hidden = self.logit_scale * torch.ones(
            batch, sequence_length, hidden_dim, device=input_ids.device
        )
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


def test_shared_prefill_collects_exact_generated_token_states_without_replay():
    batch_size = 1
    hidden_dim = 3
    num_planning_tokens = 2

    prefill_last = torch.full((batch_size, 4, hidden_dim), 10.0)
    y1_hidden = torch.full((batch_size, 1, hidden_dim), 21.0)
    y2_hidden = torch.full((batch_size, 1, hidden_dim), 22.0)
    hidden_states = (
        (torch.zeros_like(prefill_last), prefill_last),
        (torch.zeros_like(y1_hidden), y1_hidden),
        (torch.zeros_like(y2_hidden), y2_hidden),
    )
    scores = (
        torch.full((batch_size, 5), 31.0),
        torch.full((batch_size, 5), 32.0),
        torch.full((batch_size, 5), 33.0),
    )

    backbone = object.__new__(QwenPlanningEncoder)
    torch.nn.Module.__init__(backbone)
    backbone.planning_num_tokens = num_planning_tokens
    backbone.vlm_token_mode = "last"
    backbone.vlm = _FakeVLM(hidden_states=hidden_states, scores=scores)

    perception, planning, logits, labels = backbone._generate_planning_features(
        vlm_inputs={"input_ids": torch.tensor([[1, 2]])},
        planning_labels_text=["plan"],
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(perception, prefill_last)
    torch.testing.assert_close(planning, torch.cat((y1_hidden, y2_hidden), dim=1))
    torch.testing.assert_close(logits[:, 0], scores[0])
    torch.testing.assert_close(logits[:, 1], scores[1])
    assert labels.tolist() == [[7, 8]]

    assert backbone.vlm.generate_kwargs["max_new_tokens"] == num_planning_tokens + 1
    assert backbone.vlm.generate_kwargs["min_new_tokens"] == num_planning_tokens + 1
    assert backbone.vlm.generate_kwargs["use_cache"] is True
    assert backbone.vlm.generate_kwargs["output_hidden_states"] is True


def test_planning_supervision_teacher_forces_ground_truth_tokens():
    batch_size = 1
    hidden_dim = 3
    num_planning_tokens = 2
    hidden_states = (
        (torch.zeros(batch_size, 2, hidden_dim),),
        (torch.zeros(batch_size, 1, hidden_dim),),
        (torch.zeros(batch_size, 1, hidden_dim),),
    )
    backbone = object.__new__(QwenPlanningEncoder)
    torch.nn.Module.__init__(backbone)
    backbone.planning_num_tokens = num_planning_tokens
    backbone.vlm_token_mode = "last"
    backbone.vlm = _TrainableFakeVLM(
        hidden_states=hidden_states,
        scores=(torch.zeros(batch_size, 5),) * 3,
    )

    perception, planning, logits, labels = backbone._generate_planning_features(
        vlm_inputs={
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        planning_labels_text=["plan"],
        device=torch.device("cpu"),
        planning_supervision=True,
    )

    assert logits is not None
    assert logits.requires_grad
    assert backbone.vlm.generate_kwargs is None
    assert labels.tolist() == [[7, 8]]
    assert perception.shape == (1, 2, hidden_dim)
    assert planning.shape == (1, num_planning_tokens, hidden_dim)
    logits.sum().backward()
    assert backbone.vlm.logit_scale.grad is not None


def test_planning_supervision_rejects_frozen_vlm():
    hidden_states = (
        (torch.zeros(1, 2, 3),),
        (torch.zeros(1, 1, 3),),
    )
    backbone = object.__new__(QwenPlanningEncoder)
    torch.nn.Module.__init__(backbone)
    backbone.planning_num_tokens = 1
    backbone.vlm_token_mode = "last"
    backbone.vlm = _TrainableFakeVLM(
        hidden_states=hidden_states,
        scores=(torch.zeros(1, 5),) * 2,
    )
    backbone.vlm.requires_grad_(False)

    with pytest.raises(ValueError, match="requires an unfrozen VLM"):
        backbone._generate_planning_features(
            vlm_inputs={"input_ids": torch.tensor([[1, 2]])},
            planning_labels_text=["plan"],
            device=torch.device("cpu"),
            planning_supervision=True,
        )


def test_mixed_optional_planning_labels_use_ignore_index():
    batch_size = 2
    hidden_dim = 3
    hidden_states = (
        (torch.zeros(batch_size, 2, hidden_dim),),
        (torch.zeros(batch_size, 1, hidden_dim),),
        (torch.zeros(batch_size, 1, hidden_dim),),
    )
    backbone = object.__new__(QwenPlanningEncoder)
    torch.nn.Module.__init__(backbone)
    backbone.planning_num_tokens = 2
    backbone.vlm_token_mode = "last"
    backbone.vlm = _FakeVLM(
        hidden_states=hidden_states,
        scores=(torch.zeros(batch_size, 5),) * 3,
    )

    _, _, _, labels = backbone._generate_planning_features(
        vlm_inputs={"input_ids": torch.tensor([[1, 2], [3, 4]])},
        planning_labels_text=[None, "plan"],
        device=torch.device("cpu"),
    )

    assert labels.tolist() == [[-100, -100], [7, 8]]
    assert backbone.vlm.processor.tokenizer.encoded == ["plan"]


def test_explicit_image_ranges_convert_to_expected_uint8_values():
    backbone = object.__new__(QwenPlanningEncoder)
    torch.nn.Module.__init__(backbone)

    zero_one = torch.tensor(
        [[[[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]]]]
    )
    minus_one_one = torch.tensor(
        [[[[[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]]]
    )

    zero_one_image = backbone._to_pil_images(
        zero_one, input_range="zero_one"
    )[0][0]
    minus_one_one_image = backbone._to_pil_images(
        minus_one_one, input_range="minus_one_one"
    )[0][0]
    assert zero_one_image.size == (224, 224)
    assert minus_one_one_image.size == (224, 224)
    assert zero_one_image.tobytes() == minus_one_one_image.tobytes()
    # The 1x3 source is aspect-preserving letterboxed to 75x224.
    assert zero_one_image.getpixel((112, 0)) == (0, 0, 0)
    center = zero_one_image.getpixel((112, 111))
    assert all(abs(channel - 128) <= 1 for channel in center)

    with pytest.raises(ValueError, match=r"Images must be in \[0.0, 1\]"):
        backbone._to_pil_images(minus_one_one, input_range="zero_one")
    with pytest.raises(TypeError, match="requires torch.uint8"):
        backbone._to_pil_images(zero_one, input_range="uint8")


def test_task_embedding_uses_raw_high_level_without_cot_prompt():
    calls: list[tuple[list[str], bool]] = []

    class _PromptTrackingVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                CoT_prompt="PLAN:{task}",
            )
            self.processor = SimpleNamespace(tokenizer=_FakeTokenizer())
            self.model = SimpleNamespace(
                device=torch.device("cpu"),
                get_input_embeddings=lambda: torch.nn.Embedding(16, 4),
            )

        def build_qwenvl_inputs(self, images, instructions, apply_cot_prompt=True, **kwargs):
            del images, kwargs
            calls.append((list(instructions), apply_cot_prompt))
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

        def generate(self, **kwargs):
            del kwargs
            hidden = torch.zeros(1, 4, 3)
            return SimpleNamespace(
                hidden_states=(
                    (hidden,),
                    (torch.ones(1, 1, 3),),
                ),
                scores=(torch.zeros(1, 5),),
            )

        def forward(self, **kwargs):
            del kwargs
            hidden = torch.zeros(1, 4, 3)
            return SimpleNamespace(hidden_states=(hidden,))

    backbone = object.__new__(QwenPlanningEncoder)
    torch.nn.Module.__init__(backbone)
    backbone.planning_num_tokens = 1
    backbone.vlm_token_mode = "last"
    backbone.enable_planning_branch = True
    backbone.vlm = _PromptTrackingVLM()
    backbone._encode_vlm_last = lambda inputs: torch.zeros(1, 4, 3)
    backbone._pool_vlm_hidden_states = lambda hidden_states: hidden_states[-1]

    images = torch.zeros(1, 1, 3, 224, 224)
    backbone.encode_planning(
        images=images,
        planning_text=["fold shirt"],
        negative_text=None,
        training=False,
    )

    assert (["fold shirt"], True) in calls
    assert (["fold shirt"], False) in calls

