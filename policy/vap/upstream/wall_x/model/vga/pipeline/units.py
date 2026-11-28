"""
PipelineUnit - Preprocessing units
Compiler pass style, supports condition check and fallback output
"""

import logging
import torch
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING, Union

from PIL import Image

if TYPE_CHECKING:
    from wall_x.model.vga.modeling.base import BasePipeline

# Single-view: list of frames; multi-view: list of per-view frame lists (see isinstance(..., list) below).
InputVideoFrame = Union[Image.Image, torch.Tensor]
InputVideoArg = Union[List[InputVideoFrame], List[List[InputVideoFrame]]]

logger = logging.getLogger(__name__)


class PipelineUnit(ABC):
    """Base class for pipeline processing units"""

    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple = None,
        input_params_posi: dict = None,
        input_params_nega: dict = None,
        onload_model_names: tuple = None,
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def check_condition(self, pipe: "BasePipeline", **kwargs) -> bool:
        """Check whether this unit should be executed"""
        return True

    def get_fallback_output(self, pipe: "BasePipeline", **kwargs) -> dict:
        """Default output when condition is not met, subclass can override"""
        return {}

    @abstractmethod
    def run(self, pipe: "BasePipeline", **kwargs) -> dict:
        """Actual processing logic, only called when condition is met"""
        raise NotImplementedError

    def process(self, pipe: "BasePipeline", **kwargs):
        """Unified execution entry point"""
        if not self.check_condition(pipe, **kwargs):
            # print(f"[======BYPASS=======] {self.name}")
            return self.get_fallback_output(pipe, **kwargs)

        # print(f"[=======EXECUTE=======] {self.name}")
        return self.run(pipe, **kwargs)


class PipelineUnitRunner:
    """Pipeline unit runner"""

    def __init__(self):
        pass

    def __call__(
        self,
        unit: PipelineUnit,
        pipe: "BasePipeline",
        inputs_shared: dict,
        inputs_posi: dict,
        inputs_nega: dict,
    ) -> Tuple[dict, dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(
                pipe,
                inputs_shared=inputs_shared,
                inputs_posi=inputs_posi,
                inputs_nega=inputs_nega,
            )
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {
                name: inputs_posi.get(name_)
                for name, name_ in unit.input_params_posi.items()
            }
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {
                    name: inputs_nega.get(name_)
                    for name, name_ in unit.input_params_nega.items()
                }
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {
                name: inputs_shared.get(name) for name in unit.input_params
            }
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


# ============ PipelineUnits on the training critical path ============


class WanVideoUnit_ShapeChecker(PipelineUnit):
    """Check and adjust video dimensions"""

    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def run(self, pipe, height, width, num_frames, **kwargs) -> dict:
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    """Initialize noise"""

    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "batch_size",
                "num_views",
                "seed",
                "rand_device",
                "vace_reference_image",
            )
        )

    def run(
        self,
        pipe,
        height,
        width,
        num_frames,
        batch_size,
        num_views,
        seed,
        rand_device,
        vace_reference_image,
        **kwargs,
    ) -> dict:
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = (
                len(vace_reference_image)
                if isinstance(vace_reference_image, list)
                else 1
            )
            length += f
        shape = (
            batch_size * num_views,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    """Encode text prompt"""

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            input_params=("input_video", "input_image", "num_views"),
            onload_model_names=("text_encoder",),
        )

    def run(
        self,
        pipe,
        prompt,
        positive,
        num_views,
        input_video: Optional[InputVideoArg] = None,
        input_image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        **kwargs,
    ) -> dict:
        if not getattr(pipe, "skip_text_encoder", False):
            pipe.load_models_to_device(self.onload_model_names)
        # When using Qwen, provide the first frame image
        first_frame_image = None
        if pipe.use_qwen25 or pipe.use_qwen35:
            if isinstance(input_video, list) and len(input_video) > 0:
                # Handle both single-view (list of images) and multi-view (list of lists of images)
                if isinstance(input_video[0], list):
                    # Collect 0th frame from all views
                    first_frame_image = [v[0] for v in input_video]
                else:
                    first_frame_image = input_video[0]
            elif isinstance(input_image, list) and len(input_image) > 0:
                first_frame_image = input_image
            elif input_image is not None:
                first_frame_image = input_image
        # prompt may be a single string or list of strings (batch_size > 1)
        prompt_emb = pipe.prompter.encode_prompt(
            prompt,
            positive=positive,
            device=pipe.device,
            first_frame_image=first_frame_image,
            print_qwen_output=getattr(pipe, "print_qwen_output", False),
        )
        # prompt_emb: [B, seq_len, dim] (B=1 for single string, B>1 for list)
        # Expand each sample's embedding to num_views copies:
        # [B, seq, dim] -> [B, num_views, seq, dim] -> [B*num_views, seq, dim]
        context = prompt_emb.repeat_interleave(num_views, dim=0)
        return {"context": context}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    """Encode input video to latent"""

    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
            ),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, input_video, **kwargs) -> bool:
        return input_video is not None

    def get_fallback_output(self, pipe, noise, **kwargs) -> dict:
        return {"latents": noise}

    def run(
        self,
        pipe,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(["vae"])
        if isinstance(input_video[0], torch.Tensor):
            from wall_x.model.vga.data_utils import tensor_video_to_vae_input

            input_video = tensor_video_to_vae_input(
                input_video, device=pipe.device, dtype=pipe.torch_dtype
            )
        else:
            input_video = pipe.preprocess_video(input_video)
        # [B, num_views, C, T, H, W]
        input_video = torch.stack(input_video, dim=1)
        input_video = input_video.reshape(
            -1, *input_video.shape[2:]
        )  # [B*num_views, C, T, H, W]
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)

        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(
                vace_reference_image, device=pipe.device
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
            return {"latents": latents}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    """Encode image using CLIP"""

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "height",
                "width",
            ),
            onload_model_names=("image_encoder",),
        )

    def check_condition(self, pipe, input_image, **kwargs) -> bool:
        if input_image is None:
            return False
        if pipe.image_encoder is None:
            return False
        if not pipe.dit.require_clip_embedding:
            return False
        return True

    def run(
        self,
        pipe,
        input_image,
        end_image,
        height,
        width,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)

        all_clip_context = []
        for i in range(len(input_image)):
            if isinstance(input_image[i], torch.Tensor):
                from wall_x.model.vga.data_utils import tensor_frame_to_image_input

                # input_image[i] may be [H,W,C], [B,H,W,C], or [B,N,H,W,C] (multi-frame)
                img_i = (
                    input_image[i][:, -1]
                    if input_image[i].dim() == 5
                    else input_image[i]
                )
                image = tensor_frame_to_image_input(
                    img_i,
                    device=pipe.device,
                    dtype=pipe.torch_dtype,
                )
                # image: [B, C, H, W] — batch encode all samples at once
                clip_context = pipe.image_encoder.encode_image([image])
            else:
                image = pipe.preprocess_image(
                    input_image[i].resize((width, height))
                ).to(pipe.device)
                clip_context = pipe.image_encoder.encode_image([image])
            all_clip_context.append(clip_context)

        # all_clip_context: list of [B, seq, dim] per view
        # Stack views and interleave: [num_views, B, seq, dim] -> [B, num_views, seq, dim] -> [B*num_views, seq, dim]
        all_clip_context = torch.stack(
            all_clip_context, dim=0
        )  # [num_views, B, seq, dim]
        all_clip_context = all_clip_context.transpose(0, 1).reshape(
            -1, *all_clip_context.shape[2:]
        )  # [B*num_views, seq, dim]
        all_clip_context = all_clip_context.to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        return {"clip_feature": all_clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    """Encode image using VAE"""

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, input_image, **kwargs) -> bool:
        if input_image is None:
            return False
        if not pipe.dit.require_vae_embedding:
            return False
        return True

    def run(
        self,
        pipe,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)

        # Determine how many history pixel frames are being conditioned on
        n_hist = (
            input_image[0].shape[1]
            if isinstance(input_image[0], torch.Tensor) and input_image[0].dim() == 5
            else 1
        )
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, n_hist:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        is_tensor_path = isinstance(input_image[0], torch.Tensor)

        if is_tensor_path:
            # 1) Build per-view VAE inputs and collect into a list
            vae_inputs = []
            for i in range(len(input_image)):
                if input_image[i].dim() == 5:
                    # Multi-frame: input_image[i] is [B, N_hist, H, W, C]
                    from wall_x.model.vga.data_utils import tensor_video_to_vae_input

                    history_vae = tensor_video_to_vae_input(
                        [input_image[i]],
                        device=pipe.device,
                        dtype=pipe.torch_dtype,
                    )[
                        0
                    ]  # [B, C, N_hist, H, W]
                    bsz, ch, n_hist_i, h_img, w_img = history_vae.shape
                    zeros = torch.zeros(
                        bsz,
                        ch,
                        num_frames - n_hist_i,
                        h_img,
                        w_img,
                        device=history_vae.device,
                        dtype=history_vae.dtype,
                    )
                    vae_inputs.append(
                        torch.cat([history_vae, zeros], dim=2)
                    )  # [B, C, T, H, W]
                else:
                    # Single-frame: input_image[i] is [H,W,C] or [B,H,W,C]
                    from wall_x.model.vga.data_utils import tensor_frame_to_image_input

                    image = tensor_frame_to_image_input(
                        input_image[i],
                        device=pipe.device,
                        dtype=pipe.torch_dtype,
                    )
                    # image: [B, C, H, W]
                    bsz, ch, h_img, w_img = image.shape
                    zeros = torch.zeros(
                        bsz,
                        ch,
                        num_frames - 1,
                        h_img,
                        w_img,
                        device=image.device,
                        dtype=image.dtype,
                    )
                    # [B, C, T, H, W]
                    vae_inputs.append(torch.cat([image.unsqueeze(2), zeros], dim=2))

            # 2) Stack views and reshape: [num_views, B, C, T, H, W] -> [num_views*B, C, T, H, W]
            vae_batch = torch.stack(vae_inputs, dim=0)  # [V, B, C, T, H, W]
            V, B = vae_batch.shape[:2]
            vae_batch = vae_batch.reshape(V * B, *vae_batch.shape[2:])
            vae_batch = vae_batch.to(dtype=pipe.torch_dtype, device=pipe.device)

            # 3) Single VAE encode for all views
            y_all = pipe.vae.encode(
                vae_batch,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            # y_all: [V*B, C_latent, T', H', W']
            y_all = y_all.to(dtype=pipe.torch_dtype, device=pipe.device)

            # 4) Prepend mask and reshape back to [B*num_views, C_latent+4, T', H', W']
            msk_b = msk.to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)
            msk_b = msk_b.expand(y_all.shape[0], -1, -1, -1, -1)
            # [V*B, C+4, T', H', W']
            y_all = torch.concat([msk_b, y_all], dim=1)

            # Interleave: current order is [v0_b0, v0_b1, ..., v1_b0, v1_b1, ...]
            # Target order is [b0_v0, b0_v1, ..., b1_v0, b1_v1, ...]
            y_all = y_all.reshape(V, B, *y_all.shape[1:])  # [V, B, ...]
            all_y = y_all.transpose(0, 1).reshape(-1, *y_all.shape[2:])  # [B*V, ...]
        else:
            all_y = []
            for i in range(len(input_image)):
                image = pipe.preprocess_image(
                    input_image[i].resize((width, height))
                ).to(pipe.device)
                vae_input = torch.concat(
                    [
                        image.transpose(0, 1),
                        torch.zeros(3, num_frames - 1, height, width).to(image.device),
                    ],
                    dim=1,
                )
                y = pipe.vae.encode(
                    [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )[0]
                msk = msk.to(dtype=pipe.torch_dtype, device=pipe.device)
                y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
                y = torch.concat([msk, y])
                all_y.append(y.unsqueeze(0))  # [1, C+4, T', H', W']
            all_y = torch.stack(all_y, dim=0)  # [num_views, 1, ...]
            all_y = all_y.transpose(0, 1).reshape(
                -1, *all_y.shape[2:]
            )  # [num_views, ...]

        return {"y": all_y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """Encode input image to latents using VAE. For Wan-AI/Wan2.2-TI2V-5B.

    Supports both single image and multi-view (list of images) input.
    For multi-view, each view's first frame in latents is replaced with
    the VAE encoding of the corresponding input image.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, input_image, **kwargs) -> bool:
        if input_image is None:
            return False
        if not pipe.dit.fuse_vae_embedding_in_latents:
            return False
        return True

    def run(
        self,
        pipe,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)

        is_tensor_path = isinstance(input_image[0], torch.Tensor)

        if is_tensor_path:
            from wall_x.model.vga.data_utils import tensor_frame_to_image_input

            # 1) Per-view:
            #    - Single-frame: input_image[i] is [H,W,C] or [B,H,W,C]
            #    - Multi-frame:  input_image[i] is [B, N_hist, H, W, C]
            # In the multi-frame case, we take the last history frame as the reference image.
            # Then convert to [B,C,H,W] and finally [B,C,1,H,W].
            vae_inputs = []
            for i in range(len(input_image)):
                frame_i = (
                    input_image[i][:, -1]
                    if input_image[i].dim() == 5
                    else input_image[i]
                )
                image = tensor_frame_to_image_input(
                    frame_i,
                    device=pipe.device,
                    dtype=pipe.torch_dtype,
                )  # [B, C, H, W]
                vae_inputs.append(image.unsqueeze(2))  # [B, C, 1, H, W]

            # 2) Stack and reshape: [V, B, C, 1, H, W] -> [V*B, C, 1, H, W]
            vae_batch = torch.stack(vae_inputs, dim=0)
            V, B = vae_batch.shape[:2]
            vae_batch = vae_batch.reshape(V * B, *vae_batch.shape[2:])

            # 3) Single VAE encode for all views
            z_all = pipe.vae.encode(
                vae_batch,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            # z_all: [V*B, c_latent, 1, h', w']

            # 4) Interleave: [v0_b0, v0_b1, ..., v1_b0, ...] -> [b0_v0, b0_v1, ..., b1_v0, ...]
            z_all = z_all.reshape(V, B, *z_all.shape[1:])
            z_all = z_all.transpose(0, 1).reshape(
                -1, *z_all.shape[2:]
            )  # [B*V, c, 1, h', w']

            # 5) Replace first frame in latents
            latents[:, :, 0:1] = z_all
            first_frame_latents = z_all
        else:
            # PIL path — unchanged
            first_frame_latents_list = []
            for i, img in enumerate(input_image):
                image = pipe.preprocess_image(img.resize((width, height))).transpose(
                    0, 1
                )
                z = pipe.vae.encode(
                    [image],
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                latents[i : i + 1, :, 0:1] = z
                first_frame_latents_list.append(z)
            first_frame_latents = torch.cat(first_frame_latents_list, dim=0)

        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": first_frame_latents,
        }


class WanVideoUnit_CfgMerger(PipelineUnit):
    """Merge CFG positive/negative prompts - take_over mode, does not use check_condition/run"""

    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def run(self, pipe, **kwargs) -> dict:
        # take_over mode does not use run
        raise NotImplementedError("CfgMerger uses take_over mode")

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        """take_over mode directly overrides process"""
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat(
                    (tensor_shared, tensor_shared), dim=0
                )
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_ActionContext(PipelineUnit):
    """Encode ground-truth actions into context tokens for DiT cross-attention (world model mode)."""

    def __init__(self):
        super().__init__(
            input_params=("actions", "agent_pos", "dof_mask"),
        )

    def check_condition(self, pipe, actions, **kwargs) -> bool:
        return pipe.action_as_condition and actions is not None

    def run(self, pipe, actions, agent_pos, dof_mask, **kwargs) -> dict:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        actions = actions.to(device=pipe.device, dtype=pipe.torch_dtype)
        agent_pos_cond = (
            agent_pos.to(device=pipe.device, dtype=pipe.torch_dtype)
            if agent_pos is not None
            else None
        )
        action_context = pipe.action_conditioner(actions, agent_pos_cond, dof_mask)
        return {"action_context": action_context}


class WanVideoUnit_SightconeViewMask(PipelineUnit):
    """Build a cross-view attention mask from calibrated camera sightcones."""

    def __init__(self):
        super().__init__(
            input_params=(
                "camera",
                "camera_names",
                "used_view_keys",
                "unnormalized_agent_pos",
                "read_whs",
                "width",
                "height",
                "silence_view_mask_result",
            ),
        )

    def check_condition(self, pipe, camera, **kwargs) -> bool:
        silent = bool(kwargs.get("silence_view_mask_result", False))
        return (pipe.use_sightcone_matching or silent) and camera is not None

    def run(
        self,
        pipe,
        camera,
        camera_names,
        used_view_keys,
        unnormalized_agent_pos,
        read_whs,
        width,
        height,
        **kwargs,
    ) -> dict:
        from wall_x.model.vga.utils_camera import (
            BimanualCameraSightCones,
            MAPPING_ACTUATIONS,
        )

        result_cones = []
        for k, (cam_info, names) in enumerate(zip(camera, camera_names)):
            sight_cones = {}
            view_map = {f"view{i + 1}": names[i] for i in range(len(names))}
            for view_name in used_view_keys:
                sight_cone = pipe._get_sight_cone(
                    *cam_info,
                    view_name=view_name,
                    view_map=view_map,
                    read_whs=read_whs,
                    target_wh=(width, height),
                )
                if sight_cone is None:
                    raise RuntimeError(
                        "Sightcone view mask is enabled but sight cone loading failed: "
                        f"robot_id={cam_info[0]}, dataset={cam_info[2]}, view={view_name}, "
                        f"camera_path={cam_info[1]}"
                    )
                sight_cones[view_name] = sight_cone
            result_cones.append(
                MAPPING_ACTUATIONS[cam_info[2]](
                    sight_cones, unnormalized_agent_pos[k], view_map
                )
            )
        cones = torch.cat(
            [
                torch.cat([cones[i].flatten(1, 2) for i in used_view_keys], dim=1)
                for cones in result_cones
            ],
            dim=0,
        )
        view_mask = BimanualCameraSightCones.intersection(cones, radius_multiplier=2.0)
        if kwargs.get("silence_view_mask_result", False):
            return {"view_mask_silenced": view_mask}
        return {"view_mask": view_mask}
