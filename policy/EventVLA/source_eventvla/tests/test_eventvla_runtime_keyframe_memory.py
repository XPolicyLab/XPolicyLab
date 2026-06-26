import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from eventvla.model.framework.EventVLA import EventVLA
except Exception as exc:  # pragma: no cover - only used when optional train deps are absent.
    torch = None
    OmegaConf = None
    Image = None
    EventVLA = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from eventvla.training.train_eventvla import VLATrainer
except Exception as exc:  # pragma: no cover - only used when optional trainer deps are absent.
    VLATrainer = None
    TRAINER_IMPORT_ERROR = exc
else:
    TRAINER_IMPORT_ERROR = None


def _make_model(source="predict", schedule=None, teacher_prob_start=1.0, teacher_prob_end=0.0):
    model = object.__new__(EventVLA)
    model.training = True
    model.config = OmegaConf.create(
        {
            "datasets": {
                "vla_data": {
                    "keyframe_image_memory": {
                        "include_names": ["main"],
                        "exclude_name_patterns": ["wrist"],
                    }
                }
            }
        }
    )
    model.max_keyframe_images = 4
    model.keyframe_train_memory_source = source
    model.keyframe_eval_memory_source = "predict"
    model.keyframe_train_memory_schedule = schedule or source
    model.keyframe_schedule_mix_granularity = "sample"
    model.keyframe_schedule_completed_steps = 0
    model.keyframe_schedule_max_train_steps = 0
    model.keyframe_schedule_warmup_steps = 0
    model.keyframe_schedule_transition_steps = 0
    model.keyframe_schedule_teacher_prob_start = teacher_prob_start
    model.keyframe_schedule_teacher_prob_end = teacher_prob_end
    model.keyframe_schedule_progress = 1.0
    model.keyframe_memory_teacher_prob = 0.0
    model.event_future_min_offset = 1
    model.event_commit_threshold = 0.55
    model.enable_keyframe_inference_event_filter = True
    model.keyframe_nms_window = 1
    model.keyframe_cooldown_steps = 5
    model._last_keyframe_input_teacher_selector = None
    model._last_keyframe_input_metrics = {}
    model._runtime_keyframe_image_bank = []
    model._runtime_pending_keyframe_writes = []
    model._runtime_slot_episode_ids = []
    model._inference_event_state = []
    return model


def _example(step, episode_id="ep0"):
    return {
        "dataset_index": 0,
        "trajectory_id": 7,
        "episode_id": episode_id,
        "timestep": step,
        "lang": "press the button",
        "memory_keyframe_images": [],
        "memory_keyframe_image_metas": [],
        "memory_keyframe_steps": [],
    }


@unittest.skipIf(IMPORT_ERROR is not None, f"missing EventVLA test dependencies: {IMPORT_ERROR}")
class RuntimeKeyframeMemoryTest(unittest.TestCase):
    def test_predicted_pending_exact_fetch_becomes_predict_input(self):
        model = _make_model(source="predict")
        batch0 = [_example(100)]

        registered = model._register_predict_keyframe_writes(
            examples=batch0,
            pred_event_offset=torch.tensor([37]),
            pred_event_confidence=torch.tensor([0.9]),
            predicted_should_commit=torch.tensor([True]),
            keyframe_annotation_mask=torch.tensor([True]),
        )
        self.assertEqual(registered, 1)
        self.assertEqual(model._runtime_pending_keyframe_writes[0][0]["target_step"], 137)

        self.assertEqual(model.collect_due_predict_exact_fetch_requests([_example(120)]), [])
        requests = model.collect_due_predict_exact_fetch_requests([_example(150)])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["slot_idx"], 0)
        self.assertEqual(requests[0]["target_step"], 137)
        self.assertEqual(model._runtime_pending_keyframe_writes[0], [])

        history_image = Image.new("RGB", (4, 4), "blue")
        current_image = Image.new("RGB", (4, 4), "red")
        batch1 = [_example(150)]
        batch1[0]["runtime_memory_exact_fetches"] = [
            {
                **requests[0],
                "images": [history_image, current_image],
                "image_metas": [
                    {"time_role": "history", "delta": -15, "view": "main"},
                    {"time_role": "current", "delta": 0, "view": "main"},
                ],
            }
        ]

        model._resolve_training_keyframe_inputs(batch1)
        self.assertEqual(batch1[0]["keyframe_input_memory_source"], "predict")
        self.assertEqual(batch1[0]["memory_keyframe_steps"], [137])
        self.assertEqual(batch1[0]["memory_keyframe_images"], [current_image])
        self.assertEqual(batch1[0]["keyframe_input_runtime_steps"], [137])
        self.assertEqual(batch1[0]["runtime_memory_exact_fetch_consumed"], 1)

    def test_exact_fetch_consumes_into_correct_batch_slot(self):
        model = _make_model(source="predict")
        slot0_image = Image.new("RGB", (4, 4), "green")
        slot1_image = Image.new("RGB", (4, 4), "red")
        batch = [_example(200, episode_id="ep0"), _example(200, episode_id="ep1")]
        batch[0]["runtime_memory_exact_fetches"] = [
            {
                "slot_idx": 1,
                "dataset_index": 0,
                "trajectory_id": 8,
                "episode_id": "ep1",
                "sample_step": 100,
                "target_step": 137,
                "confidence": 0.8,
                "source": "predict_exact",
                "images": [slot0_image, slot1_image],
                "image_metas": [
                    {"time_role": "history", "delta": -15, "view": "main"},
                    {"time_role": "current", "delta": 0, "view": "main"},
                ],
            }
        ]

        model._resolve_training_keyframe_inputs(batch)
        self.assertEqual(batch[0]["memory_keyframe_steps"], [])
        self.assertEqual(batch[1]["memory_keyframe_steps"], [137])
        self.assertEqual(batch[1]["memory_keyframe_images"], [slot1_image])
        self.assertEqual(model._runtime_keyframe_image_bank[1][0]["step"], 137)

    def test_teacher_source_keeps_dataloader_memory_even_when_runtime_bank_exists(self):
        model = _make_model(source="teacher")
        gt_image = Image.new("RGB", (4, 4), "green")
        runtime_image = Image.new("RGB", (4, 4), "red")
        model._append_runtime_keyframe_entry(
            0,
            {
                "step": 99,
                "images": [runtime_image],
                "metas": [{"role": "memory_keyframe", "source_timestep": 99}],
                "episode_id": "ep0",
            },
        )
        batch = [_example(150)]
        batch[0]["memory_keyframe_images"] = [gt_image]
        batch[0]["memory_keyframe_image_metas"] = [{"role": "memory_keyframe", "source_timestep": 11}]
        batch[0]["memory_keyframe_steps"] = [11]

        model._resolve_training_keyframe_inputs(batch)
        self.assertEqual(batch[0]["keyframe_input_memory_source"], "teacher")
        self.assertEqual(batch[0]["memory_keyframe_steps"], [11])
        self.assertEqual(batch[0]["memory_keyframe_images"], [gt_image])
        self.assertEqual(batch[0]["keyframe_input_runtime_steps"], [99])

    def test_teacher_to_predict_late_stage_uses_runtime_bank(self):
        model = _make_model(
            source="teacher_to_predict",
            schedule="teacher_to_predict",
            teacher_prob_start=0.0,
            teacher_prob_end=0.0,
        )
        gt_image = Image.new("RGB", (4, 4), "green")
        runtime_image = Image.new("RGB", (4, 4), "red")
        model._append_runtime_keyframe_entry(
            0,
            {
                "step": 137,
                "images": [runtime_image],
                "metas": [{"role": "memory_keyframe", "source_timestep": 137}],
                "episode_id": "ep0",
            },
        )
        batch = [_example(200)]
        batch[0]["memory_keyframe_images"] = [gt_image]
        batch[0]["memory_keyframe_image_metas"] = [{"role": "memory_keyframe", "source_timestep": 42}]
        batch[0]["memory_keyframe_steps"] = [42]

        model._resolve_training_keyframe_inputs(batch)
        self.assertEqual(batch[0]["keyframe_input_memory_source"], "predict")
        self.assertEqual(batch[0]["memory_keyframe_steps"], [137])
        self.assertEqual(batch[0]["memory_keyframe_images"], [runtime_image])
        self.assertEqual(model._last_keyframe_input_metrics["keyframe_input_teacher_prob"], 0.0)
        self.assertEqual(model._last_keyframe_input_metrics["keyframe_input_predict_usage"], 1.0)

    def test_episode_reset_clears_runtime_bank_and_pending_writes(self):
        model = _make_model(source="predict")
        model._append_runtime_keyframe_entry(
            0,
            {
                "step": 10,
                "images": [Image.new("RGB", (4, 4), "red")],
                "metas": [{"role": "memory_keyframe", "source_timestep": 10}],
                "episode_id": "ep0",
            },
        )
        model._runtime_pending_keyframe_writes = [[{"episode_id": "ep0", "target_step": 20}]]
        model._runtime_slot_episode_ids = ["ep0"]

        model.reset_memory_by_mask(torch.tensor([True]), episode_ids=["ep1"])
        self.assertEqual(model._runtime_keyframe_image_bank[0], [])
        self.assertEqual(model._runtime_pending_keyframe_writes[0], [])
        self.assertEqual(model._runtime_slot_episode_ids[0], "ep1")
        self.assertIsNone(model._inference_event_state[0]["pending_step"])

    def test_inference_filter_keeps_pending_against_lower_confidence_cluster(self):
        model = _make_model(source="predict")
        model.training = False
        model.keyframe_cooldown_steps = 5

        first = torch.tensor([[0.0, 0.1, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1]])
        event_offset, event_confidence, should_commit, *_ = model._select_inference_chunk_event(
            first,
            examples=[_example(10)],
        )
        self.assertTrue(bool(should_commit[0]))
        self.assertEqual(int(event_offset[0]), 3)
        self.assertAlmostEqual(float(event_confidence[0]), 0.8, places=6)
        self.assertEqual(model._inference_event_state[0]["pending_step"], 13)

        lower_same_cluster = torch.tensor([[0.0, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1]])
        event_offset, _, should_commit, *_, suppressed = model._select_inference_chunk_event(
            lower_same_cluster,
            examples=[_example(11)],
        )
        self.assertFalse(bool(should_commit[0]))
        self.assertTrue(bool(suppressed[0]))
        self.assertEqual(int(event_offset[0]), -1)
        self.assertEqual(model._inference_event_state[0]["pending_step"], 13)

        higher_same_cluster = torch.tensor([[0.0, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]])
        event_offset, event_confidence, should_commit, *_ = model._select_inference_chunk_event(
            higher_same_cluster,
            examples=[_example(11)],
        )
        self.assertTrue(bool(should_commit[0]))
        self.assertEqual(int(event_offset[0]), 4)
        self.assertAlmostEqual(float(event_confidence[0]), 0.9, places=6)
        self.assertEqual(model._inference_event_state[0]["pending_step"], 15)

    def test_inference_filter_selects_next_nms_peak_outside_cooldown(self):
        model = _make_model(source="predict")
        model.training = False
        model.keyframe_nms_window = 1
        model.keyframe_cooldown_steps = 2
        model._ensure_runtime_slots_for_examples([_example(10)])
        model._inference_event_state[0]["last_committed_step"] = 12

        probs = torch.tensor([[0.0, 0.1, 0.92, 0.2, 0.1, 0.1, 0.1, 0.75, 0.1]])
        event_offset, event_confidence, should_commit, *_ = model._select_inference_chunk_event(
            probs,
            examples=[_example(10)],
        )

        self.assertTrue(bool(should_commit[0]))
        self.assertEqual(int(event_offset[0]), 7)
        self.assertAlmostEqual(float(event_confidence[0]), 0.75, places=6)
        self.assertEqual(model._inference_event_state[0]["pending_step"], 17)


@unittest.skipIf(TRAINER_IMPORT_ERROR is not None, f"missing trainer test dependencies: {TRAINER_IMPORT_ERROR}")
class TrainerExactFetchHookTest(unittest.TestCase):
    def test_attach_predict_exact_fetches_round_trips_due_request(self):
        class FakeModel:
            def collect_due_predict_exact_fetch_requests(self, batch):
                return [
                    {
                        "request_id": "r1",
                        "slot_idx": 1,
                        "dataset_index": 0,
                        "trajectory_id": 7,
                        "episode_id": "ep1",
                        "sample_step": 100,
                        "target_step": 137,
                        "confidence": 0.9,
                        "source": "predict_exact",
                    }
                ]

        class FakeDataset:
            def get_memory_image_at_step(self, request):
                return {
                    "request_id": request["request_id"],
                    "slot_idx": request["slot_idx"],
                    "dataset_index": request["dataset_index"],
                    "trajectory_id": request["trajectory_id"],
                    "episode_id": request["episode_id"],
                    "sample_step": request["sample_step"],
                    "target_step": request["target_step"],
                    "confidence": request["confidence"],
                    "source": request["source"],
                    "images": [Image.new("RGB", (4, 4), "red")],
                    "image_metas": [{"time_role": "current", "delta": 0, "view": "main"}],
                }

        trainer = object.__new__(VLATrainer)
        trainer.model = FakeModel()
        trainer.vla_train_dataloader = SimpleNamespace(dataset=FakeDataset())
        batch = [{"runtime_memory_exact_fetches": ["stale"]}, {}]

        trainer._attach_predict_exact_fetches(batch)
        self.assertEqual(batch[0]["predict_exact_fetch_request_count"], 1)
        self.assertEqual(batch[0]["predict_exact_fetch_success"], 1)
        self.assertEqual(batch[0]["predict_exact_fetch_missing"], 0)
        self.assertEqual(batch[0]["runtime_memory_exact_fetches"][0]["slot_idx"], 1)
        self.assertEqual(batch[0]["runtime_memory_exact_fetches"][0]["target_step"], 137)


if __name__ == "__main__":
    unittest.main()
