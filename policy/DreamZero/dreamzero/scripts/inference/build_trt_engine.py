"""Build a TensorRT engine from a DreamZero checkpoint.

Must be launched via build_trt_engine.sh (or with ENABLE_TENSORRT=true already
set) so that flash-attention compatibility mode is active before any groot model
modules are imported.

Launched via torchrun so that RANK / WORLD_SIZE / MASTER_* env vars exist for
GrootSimPolicy's distributed initialisation.

Calibration:
  For quantized precisions (nvfp4, fp8), ModelOpt calibrates quantization
  parameters by observing activation statistics during forward passes.  Using
  real dataset trajectories produces a significantly more accurate engine than
  random dummy inputs.  Pass --dataset-path to enable real calibration.
"""

import os
import sys
import argparse
import logging
from types import SimpleNamespace

# Verify ENABLE_TENSORRT was exported before any groot imports occur.
if os.getenv("ENABLE_TENSORRT", "").lower() != "true":
    print(
        "ERROR: ENABLE_TENSORRT must be 'true' before importing this script.\n"
        "Use build_trt_engine.sh instead of calling this script directly.",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.distributed as dist
from tianshou.data import Batch
from torch.distributed.device_mesh import init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.control.tensorrt_utils import (
    wan_trt_quantize_and_load_engine,
    create_wan_test_inputs,
)

# DreamZero-DROID uses the ar_14B_droid model type in tensorrt_utils.
_MODEL_TYPE = "ar_14B_droid"


def _init_single_gpu_mesh():
    """Initialise a single-GPU device mesh (launched via torchrun --nproc_per_node=1)."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )
    return mesh


def _make_dummy_forward_loop():
    """Fallback calibration using random dummy inputs.

    Acceptable for fp16 (no quantization), but may reduce accuracy for
    nvfp4/fp8 since the activation distribution differs from real data.
    Prefer _make_dataset_forward_loop when a dataset is available.
    """
    def forward_loop(model):
        trt_forward = getattr(model, "_forward_inference_trt_droid", model.forward)
        test_inputs = create_wan_test_inputs(None, device="cuda", model_type=_MODEL_TYPE)
        for _ in range(16):
            with torch.no_grad():
                trt_forward(*test_inputs)

    return forward_loop


def _make_dataset_forward_loop(policy, dataset_path: str, num_calibration_trajs: int = 2):
    """Real-data calibration loop — mirrors the internal droid_video_pred.sh approach.

    Loads ``num_calibration_trajs`` trajectories from the LeRobot dataset and
    runs ``policy.lazy_joint_forward_causal`` at each action-horizon step,
    exercising the DiT model with realistic activation distributions.
    """
    from groot.vla.data.dataset.lerobot import LeRobotSingleDataset

    def forward_loop(model):
        logger.info(
            "Calibration: loading dataset from %s (%d trajs)", dataset_path, num_calibration_trajs
        )
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=policy.modality_configs,
            embodiment_tag=policy.embodiment_tag,
            video_backend="torchvision_av",
            video_backend_kwargs=None,
            transforms=None,        # policy.lazy_joint_forward_causal applies transforms
            use_global_metadata=False,
        )

        action_horizon = policy.trained_model.action_head.action_horizon
        num_frame_per_block = policy.trained_model.action_head.num_frame_per_block
        torch._dynamo.config.recompile_limit = 500

        for traj_id in range(min(num_calibration_trajs, len(dataset.trajectory_lengths))):
            logger.info("Calibration trajectory %d / %d", traj_id + 1, num_calibration_trajs)
            traj_len = int(dataset.trajectory_lengths[traj_id])
            latent_video = None

            # Step through the trajectory at action-horizon intervals (same cadence as
            # real inference) for up to 5 chunks — enough to cover the KV-cache build-up
            # and the cached inference path that the TRT engine will handle.
            max_steps = min(traj_len, 5 * action_horizon)
            for step in range(0, max_steps, action_horizon):
                # Clamp delta indices to valid range for this trajectory.
                indices = {
                    k: np.clip(v + step, 0, traj_len - 1)
                    for k, v in dataset.delta_indices.items()
                }
                data_point = dataset.get_step_data(traj_id, indices)
                batch = Batch(obs=data_point)

                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = policy.lazy_joint_forward_causal(
                        batch, latent_video=latent_video
                    )
                dist.barrier()

                # Feed the last generated frame back as context for the next step,
                # matching autoregressive inference behaviour.
                if video_pred is not None:
                    latent_video = video_pred[:, :, -num_frame_per_block:]

            # Reset AR state between trajectories.
            policy.trained_model.action_head.current_start_frame = 0
            policy.trained_model.action_head.kv_cache1 = None
            policy.trained_model.action_head.kv_cache_neg = None
            policy.trained_model.action_head.crossattn_cache = None
            policy.trained_model.action_head.crossattn_cache_neg = None

    return forward_loop


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine for the DreamZero DiT model."
    )
    parser.add_argument("--model-path", required=True, help="Path to checkpoint directory.")
    parser.add_argument(
        "--tensorrt",
        required=True,
        choices=["nvfp4", "fp8", "fp16"],
        help="TensorRT quantization / precision format.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Path to a LeRobot-format DROID dataset for real calibration. "
            "Strongly recommended for nvfp4/fp8 — random dummy inputs are used as "
            "fallback but may reduce quantization accuracy."
        ),
    )
    parser.add_argument(
        "--num-calibration-trajs",
        type=int,
        default=2,
        help="Number of dataset trajectories used for calibration (default: 2).",
    )
    args = parser.parse_args()

    if args.tensorrt in ("nvfp4", "fp8") and args.dataset_path is None:
        logger.warning(
            "No --dataset-path provided for %s quantization. "
            "Falling back to random dummy inputs — this may reduce engine accuracy. "
            "Re-run with --dataset-path <path/to/droid_lerobot> for best results.",
            args.tensorrt,
        )

    engine_dir = os.path.join(args.model_path, "tensorrt", "wan")
    engine_path = os.path.join(engine_dir, f"WanModel_{args.tensorrt}.trt")
    onnx_path = os.path.join(engine_dir, f"CausalWanModel.onnx")
    os.makedirs(engine_dir, exist_ok=True)

    if os.path.exists(engine_path):
        logger.info("TRT engine already exists: %s", engine_path)
        logger.info("Delete it first if you want to rebuild.")
        return

    logger.info("Loading DreamZero policy from : %s", args.model_path)
    logger.info("Target engine path            : %s", engine_path)
    logger.info("Quantization precision        : %s", args.tensorrt)

    device_mesh = _init_single_gpu_mesh()

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    # Build calibration forward loop — prefer real data for quantized precisions.
    if args.dataset_path is not None:
        forward_loop = _make_dataset_forward_loop(
            policy, args.dataset_path, args.num_calibration_trajs
        )
        logger.info(
            "Calibration: using %d real trajectories from %s",
            args.num_calibration_trajs,
            args.dataset_path,
        )
    else:
        forward_loop = _make_dummy_forward_loop()
        logger.info("Calibration: using random dummy inputs (no --dataset-path given).")

    # cfg mimics the Hydra config used by the internal eval script.
    cfg = SimpleNamespace(inference_mode="trt_build", quantize_dtype=args.tensorrt)

    logger.info("Building TensorRT engine (ONNX export + trtexec, may take 10-30 min) ...")
    wan_trt_quantize_and_load_engine(
        policy=policy,
        cfg=cfg,
        onnx_path=onnx_path,
        engine_path=engine_path,
        model_type=_MODEL_TYPE,
        forward_loop=forward_loop,
    )

    logger.info("TRT engine saved to: %s", engine_path)


if __name__ == "__main__":
    main()
