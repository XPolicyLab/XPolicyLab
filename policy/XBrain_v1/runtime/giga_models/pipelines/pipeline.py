import os
from importlib import import_module

import torch

from .. import utils


class BasePipeline:
    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LazyPipeline(BasePipeline):
    def __init__(self, pipeline, pipeline_info):
        self.pipeline = pipeline
        self.pipeline_info = pipeline_info
        self.device = None
        self.is_init = False

    def init_pipeline(self):
        if not self.is_init:
            self.pipeline = self.pipeline(**self.pipeline_info)
            if self.device is not None:
                self.pipeline.to(self.device)
            self.is_init = True

    def to(self, device):
        self.device = device
        if self.is_init:
            self.pipeline.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.init_pipeline()
        return self.pipeline(*args, **kwargs)


def get_text_pipelines():
    model_dir = utils.get_model_dir()
    pipelines = {
        'video_caption/cogvlm2_llama3_caption': {
            '_class_name': 'text.video_caption.pipeline_cogvlm2.CogVLM2Pipeline',
            'model_path': os.path.join(model_dir, 'huggingface/models--THUDM--cogvlm2-llama3-caption'),
            'torch_dtype': torch.bfloat16,
        },
        'video_caption/panda_video_llama': {
            '_class_name': 'text.video_caption.pipeline_panda_video_llama.PandaVideoLLaMAPipeline',
            'model_path': os.path.join(model_dir, 'others/panda-70M/checkpoint_best.pth'),
            'q_former_model_path': os.path.join(model_dir, 'torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth'),
            'llama_model_path': os.path.join(model_dir, 'others/vicuna/vicuna-v0/vicuna-7b-full-v0'),
        },
        'video_caption/Qwen2.5_VL_7B_Instruct': {
            '_class_name': 'text.video_caption.pipeline_qwen.Qwen2_5_VLPipeline',
            'model_path': os.path.join(model_dir, 'huggingface/models--Qwen--Qwen2.5-VL-7B-Instruct'),
        },
        'video_caption/Qwen3_4B': {
            '_class_name': 'text.video_caption.pipeline_qwen.QwenPipeline',
            'model_path': os.path.join(model_dir, 'huggingface/models--Qwen--Qwen3-4B'),
        },
        'video_caption/Qwen3_8B': {
            '_class_name': 'text.video_caption.pipeline_qwen.QwenPipeline',
            'model_path': os.path.join(model_dir, 'huggingface/models--Qwen--Qwen3-8B'),
        },
    }
    return pipelines


def get_vision_pipelines():
    model_dir = utils.get_model_dir()
    pipelines = {
        'depth_estimation/depth_anything/v2_small_hf': {
            '_class_name': 'vision.depth_estimation.pipeline_depth_anything.DepthAnythingPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'depth-anything/Depth-Anything-V2-Small-hf',
        },
        'depth_estimation/depth_anything/v2_base_hf': {
            '_class_name': 'vision.depth_estimation.pipeline_depth_anything.DepthAnythingPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'depth-anything/Depth-Anything-V2-Base-hf',
        },
        'depth_estimation/depth_anything/v2_large_hf': {
            '_class_name': 'vision.depth_estimation.pipeline_depth_anything.DepthAnythingPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'depth-anything/Depth-Anything-V2-Large-hf',
        },
        'depth_estimation/dpt/hybrid_midas': {
            '_class_name': 'vision.depth_estimation.pipeline_dpt.DPTForDepthEstimationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'Intel/dpt-hybrid-midas',
        },
        'depth_estimation/dpt/large': {
            '_class_name': 'vision.depth_estimation.pipeline_dpt.DPTForDepthEstimationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'Intel/dpt-large',
        },
        'depth_estimation/video_depth_anything/small': {
            '_class_name': 'vision.depth_estimation.pipeline_video_depth_anything.VideoDepthAnythingPipeline',
            'model_path': os.path.join(model_dir, 'video_depth_anything/video_depth_anything_vits.pth'),
        },
        'depth_estimation/video_depth_anything/large': {
            '_class_name': 'vision.depth_estimation.pipeline_video_depth_anything.VideoDepthAnythingPipeline',
            'model_path': os.path.join(model_dir, 'video_depth_anything/video_depth_anything_vitl.pth'),
        },
        'detection/grounding_dino/swint_ogc': {
            '_class_name': 'vision.detection.pipeline_grounding_dino.GroundingDINOPipeline',
            'config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
        },
        'edge_detection/canny': {
            '_class_name': 'vision.edge_detection.pipeline_canny.CannyPipeline',
        },
        'edge_detection/hed/apache2': {
            '_class_name': 'vision.edge_detection.pipeline_hed.HEDPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'edge_detection/lineart/sk_model': {
            '_class_name': 'vision.edge_detection.pipeline_lineart.LineartPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'edge_detection/mlsd/large_512_fp32': {
            '_class_name': 'vision.edge_detection.pipeline_mlsd.MLSDPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'edge_detection/pidinet/table5': {
            '_class_name': 'vision.edge_detection.pipeline_pidinet.PidiNetPipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'face_restoration/codeformer/v0.1.0': {
            '_class_name': 'vision.face_restoration.pipeline_codeformer.CodeFormerPipeline',
            'model_path': os.path.join(model_dir, 'codeformer/codeformer-v0.1.0.pth'),
        },
        'face_swap/insightface/inswapper_128': {
            '_class_name': 'vision.face_swap.pipeline_insightface.InsightFaceSwapPipeline',
            'model_path': os.path.join(model_dir, 'insightface/inswapper_128.onnx'),
        },
        'frame_interpolation/film/film_net_fp16': {
            '_class_name': 'vision.frame_interpolation.pipeline_film.FilmPipeline',
            'model_path': os.path.join(model_dir, 'frame_interpolation/film_net_fp16.pt'),
        },
        'image_restoration/prompt_ir': {
            '_class_name': 'vision.image_restoration.pipeline_prompt_ir.PromptIRPipeline',
            'model_path': os.path.join(model_dir, 'prompt_ir/model.ckpt'),
        },
        'inpaint/inpaint_anything/big_lama_sttn': {
            '_class_name': 'vision.inpaint.pipeline_inpaint_anything.InpaintAnythingPipeline',
            'lama_model_path': os.path.join(model_dir, 'inpaint_anything/big_lama.ckpt'),
            'lama_config_path': os.path.join(model_dir, 'inpaint_anything/big_lama.yaml'),
            'sttn_model_path': os.path.join(model_dir, 'inpaint_anything/sttn.pth'),
            'inpainter_target': 'lama',
        },
        'keypoints/dwpose/body_hand_face': {
            '_class_name': 'vision.keypoints.pipeline_dwpose.DWposePipeline',
            'pose_config': os.path.join(model_dir, 'dwposes/dwpose-l_384x288.py'),
            'pose_ckpt': os.path.join(model_dir, 'dwposes/dw-ll_ucoco_384.pth'),
        },
        'keypoints/openpose/body_hand_face': {
            '_class_name': 'vision.keypoints.pipeline_openpose.OpenPosePipeline',
            'model_path': 'lllyasviel/Annotators',
        },
        'keypoints/pose_aligner/2d_motion_retarget': {
            '_class_name': 'vision.keypoints.pipeline_pose_aligner.PoseAlignerPipeline',
            'model_dir': os.path.join(model_dir, 'retarget'),
        },
        'keypoints/rtmpose/performance': {
            '_class_name': 'vision.keypoints.pipeline_rtm_pose.RTMPosePipeline',
            'det_path': os.path.join(model_dir, 'rtmpose/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
            'pose_path': os.path.join(model_dir, 'rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'),
            'mode': 'performance',
        },
        'lane_detection/laneaf/dla34_640x288_batch2_v023': {
            '_class_name': 'vision.lane_detection.pipeline_laneaf.LaneAFPipeline',
            'model_path': os.path.join(model_dir, 'lane/dla34-640x288_batch2-v023.pth'),
        },
        'ocr/craft/craft_mlt_25k': {
            '_class_name': 'vision.ocr.pipeline_craft.OCRCRAFTPipeline',
            'model_path': os.path.join(model_dir, 'ocr_craft/craft_mlt_25k.pth'),
        },
        'optical_flow/unimatch/gmflow_scale2_regrefine6_mixdata': {
            '_class_name': 'vision.optical_flow.pipeline_unimatch.UniMatchPipeline',
            'model_path': os.path.join(model_dir, 'unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'),
        },
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_t': {
            '_class_name': 'vision.segmentation.pipeline_grounded_sam2.GroundedSAM2Pipeline',
            'gd_config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'gd_model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
            'sam_config_path': 'configs/sam2.1/sam2.1_hiera_t.yaml',
            'sam_model_path': os.path.join(model_dir, 'segment_anything_2/sam2.1_hiera_tiny.pt'),
        },
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_s': {
            '_class_name': 'vision.segmentation.pipeline_grounded_sam2.GroundedSAM2Pipeline',
            'gd_config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'gd_model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
            'sam_config_path': 'configs/sam2.1/sam2.1_hiera_s.yaml',
            'sam_model_path': os.path.join(model_dir, 'segment_anything_2/sam2.1_hiera_small.pt'),
        },
        'segmentation/grounded_sam2/gd_swint_ogc_sam21_hiera_l': {
            '_class_name': 'vision.segmentation.pipeline_grounded_sam2.GroundedSAM2Pipeline',
            'gd_config_path': os.path.join(model_dir, 'grounding_dino/GroundingDINO_SwinT_OGC.py'),
            'gd_model_path': os.path.join(model_dir, 'grounding_dino/groundingdino_swint_ogc.pth'),
            'sam_config_path': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'sam_model_path': os.path.join(model_dir, 'segment_anything_2/sam2.1_hiera_large.pt'),
        },
        'segmentation/segment_anything/vit_b_01ec64': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'segment_anything/sam_vit_b_01ec64.pth'),
            'model_type': 'vit_b',
        },
        'segmentation/segment_anything/vit_l_0b3195': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'segment_anything/sam_vit_l_0b3195.pth'),
            'model_type': 'vit_l',
        },
        'segmentation/segment_anything/vit_h_4b8939': {
            '_class_name': 'vision.segmentation.pipeline_segment_anything.SegmentAnythingPipeline',
            'model_path': os.path.join(model_dir, 'segment_anything/sam_vit_h_4b8939.pth'),
            'model_type': 'vit_h',
        },
        'segmentation/upernet/convnext_tiny': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'openmmlab/upernet-convnext-tiny',
        },
        'segmentation/upernet/convnext_small': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'openmmlab/upernet-convnext-small',
        },
        'segmentation/upernet/convnext_large': {
            '_class_name': 'vision.segmentation.pipeline_upernet.UperNetForSemanticSegmentationPipeline',
            '_hf_download_first': ['model_path'],
            'model_path': 'openmmlab/upernet-convnext-large',
        },
        'shot_boundary_detection/transnetv2': {
            '_class_name': 'vision.shot_boundary_detection.pipeline_transnetv2.TransNetV2Pipeline',
            'model_path': os.path.join(model_dir, 'transnetv2/transnetv2-pytorch-weights.pth'),
        },
        'super_resolution/real_esrgan/x2plus': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [os.path.join(model_dir, 'real_esrgan/RealESRGAN_x2plus.pth')],
            'model_names': ['x2plus'],
        },
        'super_resolution/real_esrgan/x4plus': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [os.path.join(model_dir, 'real_esrgan/RealESRGAN_x4plus.pth')],
            'model_names': ['x4plus'],
        },
        'super_resolution/real_esrgan/x4plus_anime_6B': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [os.path.join(model_dir, 'real_esrgan/RealESRGAN_x4plus_anime_6B.pth')],
            'model_names': ['x4plus_anime_6B'],
        },
        'super_resolution/real_esrgan/x2plus_x4plus': {
            '_class_name': 'vision.super_resolution.pipeline_real_esrgan.RealESRGANPipeline',
            'model_paths': [
                os.path.join(model_dir, 'real_esrgan/RealESRGAN_x2plus.pth'),
                os.path.join(model_dir, 'real_esrgan/RealESRGAN_x4plus.pth'),
            ],
            'model_names': ['x2plus', 'x4plus'],
        },
        'super_resolution/rgt/rgt_s': {
            '_class_name': 'vision.super_resolution.pipeline_rgt.RGTPipeline',
            'model_paths': [
                os.path.join(model_dir, 'rgt/RGT_S_x2.pth'),
                os.path.join(model_dir, 'rgt/RGT_S_x3.pth'),
                os.path.join(model_dir, 'rgt/RGT_S_x4.pth'),
            ],
        },
        'others/aesthetic/sa_0_4_vit_l_14_linear': {
            '_class_name': 'vision.others.pipeline_aesthetic.AestheticPipeline',
            'model_path': os.path.join(model_dir, 'aesthetic_clip/sa_0_4_vit_l_14_linear.pth'),
            'pretrained_path': os.path.join(model_dir, 'aesthetic_clip/ViT-L-14.pt'),
        },
        'others/blur': {
            '_class_name': 'vision.others.pipeline_blur.BlurPipeline',
            'strength': 'very_high',
        },
        'others/laplace': {
            '_class_name': 'vision.others.pipeline_laplace.LaplacePipeline',
        },
        'others/normal_bae/scannet': {
            '_class_name': 'vision.others.pipeline_normal_bae.NormalBaePipeline',
            'model_name': 'lllyasviel/Annotators',
        },
        'others/shuffle/content': {
            '_class_name': 'vision.others.pipeline_shuffle.ContentShufflePipeline',
        },
    }
    return pipelines


def get_pipelines():
    pipelines_list = [get_text_pipelines(), get_vision_pipelines()]
    pipelines = {}
    for pipelines_i in pipelines_list:
        for key in pipelines_i:
            assert key not in pipelines
        pipelines.update(pipelines_i)
    return pipelines


def load_pipeline(pipeline_name, lazy=False, **kwargs):
    pipelines = get_pipelines()
    pipeline_info = pipelines[pipeline_name]
    pipeline_info.update(kwargs)
    parts = pipeline_info.pop('_class_name').split('.')
    module_name = '.'.join(parts[:-1])
    module = import_module('giga_models.pipelines.' + module_name)
    pipeline = getattr(module, parts[-1])
    # Download models from Hugging Face first, if necessary.
    hf_download_keys = pipeline_info.pop('_hf_download_first', [])
    for key in hf_download_keys:
        pipeline_info[key] = utils.download_from_huggingface(pipeline_info[key])
    if lazy:
        return LazyPipeline(pipeline, pipeline_info)
    else:
        return pipeline(**pipeline_info)


def list_pipelines():
    pipelines = get_pipelines()
    return list(pipelines.keys())
