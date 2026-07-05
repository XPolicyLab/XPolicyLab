from typing import Optional, Dict, Any,Type,Tuple
from pathlib import Path
from dataclasses import dataclass
import logging  
# from itertools import islice

import numpy as np
 
from PIL import Image
from a1.data.vla import rlds

# import dlimp as dl

# import torch
# from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset

from a1.data.dataset import Dataset  

from a1.data.vla.utils import NormalizationType

# from a1.vla.util import FIXED_ACTION_DIM #32

from a1.data.vla.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from a1.data.vla.rlds import make_interleaved_dataset,make_single_dataset
## NOTE: above consolidated import already brings these in

from a1.data.vla.rlds.utils.data_utils import tree_map


def _ensure_finite(name: str, array, context: Dict[str, Any] = None) -> None:
    ctx = context or {}

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"NaN/Inf detected in {name}: shape={array.shape}, dtype={array.dtype}, context={ctx}")

# @dataclass
# class RLDSBatchTransform:
#     # action_tokenizer: ActionTokenizer
#     # base_tokenizer: PreTrainedTokenizerBase
#     # image_transform: ImageTransform
#     # prompt_builder_fn: Type[PromptBuilder]
#     predict_stop_token: bool = True
#     use_wrist_image: bool = False
#     use_proprio: bool = False

#     def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
#         """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
#         # return rlds_batch
    
        # dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        # # img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        # lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # actions = rlds_batch["action"]

        # # # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        # # prompt_builder = self.prompt_builder_fn("openvla")

        # # # Get future action chunk
        # future_actions = rlds_batch["action"][1:]
        # future_actions_string = ''.join(self.action_tokenizer(future_actions))

        # # # Get action chunk string
        # current_action_string = self.action_tokenizer(current_action)
        # action_chunk_string = current_action_string + future_actions_string
        # action_chunk_len = len(action_chunk_string)

        # conversation = [
        #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
        #     {"from": "gpt", "value": action_chunk_string},
        # ]
        # for turn in conversation:
        #     prompt_builder.add_turn(turn["from"], turn["value"])

        # # Tokenize (w/ `base_tokenizer`)
        # input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        # labels = list(input_ids)

        # # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        # #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        # input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        # pixel_values = self.image_transform(img)

        # # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        # if not self.predict_stop_token:
        #     labels[-1] = IGNORE_INDEX

        # return_dict = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name, actions=actions)

        # # Add additional inputs
        # if self.use_wrist_image:
        #     all_wrist_pixels = []
        #     for k in rlds_batch["observation"].keys():
        #         if "wrist" in k:
        #             img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
        #             pixel_values_wrist = self.image_transform(img_wrist)
        #             all_wrist_pixels.append(pixel_values_wrist)
        #     return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        # if self.use_proprio and "proprio" in rlds_batch["observation"]:
        #     proprio = rlds_batch["observation"]["proprio"]
        #     return_dict["proprio"] = proprio

        # return return_dict

def convert_gripper_qpos_to_1d(gripper_qpos_2d):
    """将2维夹爪关节位置转换为夹爪开合距离"""
    # 计算两个夹爪指间的距离
    return abs(gripper_qpos_2d[:,0] - gripper_qpos_2d[:,1])

@dataclass
class RLDSBatchTransform:
    # predict_stop_token: bool = True
    use_wrist_image: bool = True
    use_proprio: bool = True
    fixed_action_dim: int = 7
    pad_action_and_proprio: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        # return rlds_batch
        observation = rlds_batch["observation"]
        action = rlds_batch["action"]
        proprio = rlds_batch["observation"]["proprio"]
        # print(f"*** action.shape: {action.shape}, proprio.shape: {proprio.shape}")
        # if action and proprio dim samller than ACTION_DIM and PROPRIO_DIM, pad them to the same shape
        # 记录原始动作维度，计算需要padding的长度
        original_action_dim = action.shape[-1]
        pad_len_action = self.fixed_action_dim - original_action_dim
        if self.pad_action_and_proprio and pad_len_action > 0:
            # 仅对最后一维（特征维）进行padding到配置维度，时间维保持不变
            action = np.pad(action, ((0, 0), (0, pad_len_action)), mode='constant')
        if self.pad_action_and_proprio and proprio.shape[-1] < self.fixed_action_dim:
            pad_len_proprio = self.fixed_action_dim - proprio.shape[-1]
            proprio = np.pad(proprio, ((0, 0), (0, pad_len_proprio)), mode='constant')

        absolute_action_mask = rlds_batch["absolute_action_mask"]

        dataset_name = rlds_batch["dataset_name"]
        current_action, future_actions = action[0],action[1:]

        image_primary = observation["image_primary"]
        image_wrist = observation["image_wrist"]

        instruction = rlds_batch["task"]["language_instruction"].decode().lower()

        observation.pop("image_primary", None)  # Remove primary image from observation dict
        # if not self.use_wrist_image:
        observation.pop("image_wrist", None)
        # if not self.use_proprio:
        #     observation.pop("proprio", None)

        # if not self.predict_stop_token:
        #     labels[-1] = IGNORE_INDEX
        if image_primary.ndim ==4:
            image_primary = image_primary.squeeze(0)  # Remove batch dimension if present
            image_primary=image_primary.copy()
        if image_wrist.ndim ==4:
            image_wrist = image_wrist.squeeze(0)  # Remove batch dimension if present
            image_wrist=image_wrist.copy()
        # image_primary.setflags(write=True) # make it writable before converting it to a tensor
        assert image_primary.ndim == 3, f"Image should have 3 dimensions, got {image_primary.ndim}"
        assert image_wrist.ndim == 3, f"Wrist image should have 3 dimensions, got {image_wrist.ndim}"
        # Finite checks for image/action/proprio
        _ensure_finite("image_primary", image_primary, {"dataset": dataset_name})
        _ensure_finite("image_wrist", image_wrist, {"dataset": dataset_name})
        _ensure_finite("action", action, {"dataset": dataset_name})
        if self.use_proprio:
            _ensure_finite("proprio", proprio, {"dataset": dataset_name})
        
        # 生成与动作同形状的padding掩码，pad位置标记为True
        action_pad_mask = np.zeros_like(action, dtype=bool)
        if pad_len_action > 0:
            action_pad_mask[:, -pad_len_action:] = True

        return_dict = {  
            # "image": image_primary,  
            # "images":[image_primary,image_wrist],
            # "question": f"What action should the robot take to {instruction}?", # is the question necessary?
            "question": instruction, 
            # "message_list": conversation,  
            "answer": "Action",
            "style": "action",
            "action": action.copy(), 
            "action_pad_mask": action_pad_mask,
            "proprio": proprio.copy() if self.use_proprio else None,

            "timestep": observation['timestep'],


            "metadata": {  
                "dataset_name": dataset_name,  
                "instruction": instruction,  
                "action": action.copy(), 
                # "observation": observation,
                "absolute_action_mask": absolute_action_mask,
            }
        } 
        if self.use_wrist_image:
            return_dict["images"] = [image_primary, image_wrist]
        else:
            return_dict["image"] = image_primary
        # print("*** RLDSBatchTransform,action",action.shape)


        return return_dict


@dataclass
class DiTActionRLDSBatchTransform(RLDSBatchTransform):
    """Batch transform for DiT Action RLDS dataset."""
    tokenizer: Any = None  # 文本tokenizer
    processor: Any = None  # 图像processor
    max_text_length: int = 1024  # 文本最大长度

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """调用父类方法获取基础数据，然后添加 DiffusionTransformerAction 需要的处理"""
        # 先调用父类的处理方法获取基础数据
        base_result = super().__call__(rlds_batch)
        
        # 提取需要进一步处理的数据
        images = base_result["images"] if "images" in base_result else [base_result["image"],]
        # [image_primary, image_wrist]
        # question = base_result["question"]
        question = base_result["metadata"]["instruction"]  # 使用instruction作为问题

        # 处理图像 - 转换为PIL格式并使用processor处理
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # 确保是uint8格式
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        # 使用processor处理图像
        if self.processor is not None:
            if len(pil_images) == 1:
                processed_images = self.processor(images=pil_images[0], return_tensors="pt")
            else:
                processed_images = self.processor(images=pil_images, return_tensors="pt")
        # print("***** processed_images keys:", processed_images.keys())
        base_result["pixel_values"] = processed_images["pixel_values"]

        # 使用tokenizer处理文本
        if self.tokenizer is not None:
            # tokenized_text = self.tokenizer(question,padding="longest",return_tensors="pt") #max_length=self.max_text_length,
            tokenized_text = self.tokenizer(question, padding="max_length", max_length=self.max_text_length,return_tensors="pt") 
            # 调试：打印tokenized_text的键
            # print("***** tokenized_text keys:", tokenized_text.keys())
            # print("***** tokenized_text attention mask dtype:", tokenized_text["attention_mask"].dtype) # torch.Size([1, 1024]) torch.int64
            # 添加tokenized文本信息
            base_result["input_ids"] = tokenized_text["input_ids"].squeeze()
            #'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0]])})
            base_result["text_attention_mask"] = tokenized_text.get("attention_mask", None)
        
        return base_result

# 创建统一的Batch Transform for ModelSelector
class ModelSelectorBatchTransform(RLDSBatchTransform):
    """统一的batch transform，同时支持AffordVLA和DiTAction的数据格式"""
    
    def __init__(self, tokenizer=None, processor=None, max_text_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_text_length = max_text_length
    
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # 先调用父类方法获取基础数据
        result = super().__call__(rlds_batch)
        
        # 处理图像 - 为DiTAction准备pixel_values
        # if self.processor is not None and "images" in result:
            #deep copy images, avoid modifying the original images
        images = result["images"].copy() if "images" in result else [result["image"].copy(),]

        pil_images = []

        for img in images:
        
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
        
        if len(pil_images) == 1:
            processed_images = self.processor(images=pil_images[0], return_tensors="pt")
        else:
            processed_images = self.processor(images=pil_images, return_tensors="pt")
        
        result["pixel_values"] = processed_images["pixel_values"]
        
        # 处理文本 - 为DiTAction准备tokenized文本
        if self.tokenizer is not None:
            instruction = result["metadata"]["instruction"]
            tokenized_text = self.tokenizer(
                instruction, 
                padding="max_length", 
                max_length=self.max_text_length,
                return_tensors="pt"
            )
            # 保持二维形状以便后续 collate 按 batch 维拼接成 (B, L)
            result["input_ids_dit"] = tokenized_text["input_ids"] # squeeze()
            result["text_attention_mask"] = tokenized_text.get("attention_mask", None)
        
        return result


class RLDSDataset(Dataset,IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        traj_transform_threads: int = None,
        traj_read_threads: int = None,
        num_actions_chunk : int = 8,
        normalization_type: NormalizationType = NormalizationType.BOUNDS,
        # data_additional_opensource_data: str=None,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        single_sample_mode: bool = False,  # 新增参数
        sample_ratio: float = 1.0,
        sample_index: list = [0,10,20,30,40,50,60,70,80,90,100,101,102,103,104,105,106,
                              107,108,109,110,111,112,113,114,115,116,117,188,119,120,121,122],  # 新增参数，指定要使用的样本索引
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        assert data_root_dir is not None, "data_root_dir must be specified for RLDSDataset"
        assert data_mix is not None, "data_mix must be specified for RLDSDataset"

        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform
        self.traj_transform_threads = traj_transform_threads
        self.traj_read_threads = traj_read_threads
        self.normalization_type = normalization_type
        self.num_actions_chunk = num_actions_chunk
        # 确保在后续逻辑中可以访问该字段
        # self.data_additional_opensource_data = data_additional_opensource_data

        self.single_sample_mode = single_sample_mode
        self.sample_index = sample_index
        self._cached_samples = {}  # 缓存单个样本

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            print("*"*50)
            print("!"*50)
            print(f"********* RLDSDataset: {self.data_mix} is not in OXE_NAMED_MIXTURES")
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # if self.data_additional_opensource_data:
        #     if self.data_additional_opensource_data in OXE_NAMED_MIXTURES:
        #         mixture_spec.extend(OXE_NAMED_MIXTURES[self.data_additional_opensource_data])
        #     else:
        #         mixture_spec.append((self.data_additional_opensource_data, 1.0))
        #     logging.info(f"Using additional open-source RLDS dataset: {self.data_additional_opensource_data}")
        
        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=self.normalization_type,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=self.num_actions_chunk-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=32,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            # traj_transform_threads=(len(mixture_spec)*2 if self.traj_transform_threads is None else int(self.traj_transform_threads)),
            # traj_read_threads=(len(mixture_spec)*2 if self.traj_read_threads is None else int(self.traj_read_threads)),
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            # traj_transform_threads=32,
            # traj_read_threads=32,
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

        if sample_ratio < 1.0:
            self.dataset = self.dataset.take(int(self.dataset_length * sample_ratio))
            self.dataset_length = int(self.dataset_length * sample_ratio)

        # 如果是单样本模式，提前获取并缓存样本
        if self.single_sample_mode:
            self._cache_selected_samples()

        print("******", "after RLDSDataset initialization!")

    # def _cache_single_sample(self):
    #     """缓存指定索引的单个样本"""
    #     iterator = self.dataset.as_numpy_iterator()
    #     try:
    #         # 跳到指定的样本
    #         rlds_batch = next(islice(iterator, self.sample_index, None))
    #         self._cached_sample = self.batch_transform(rlds_batch)
    #         print(f"已缓存第 {self.sample_index} 个样本用于单样本测试")
    #     except StopIteration:
    #         raise IndexError(f"样本索引 {self.sample_index} 超出范围 (数据集长度={self.dataset_length})")


    def _cache_selected_samples(self):
        """缓存指定索引列表中的样本"""
        # 对索引进行排序，确保按顺序读取
        sorted_indices = sorted(self.sample_index)
        iterator = self.dataset.as_numpy_iterator()
        
        current_idx = 0
        cached_count = 0
        
        try:
            for rlds_batch in iterator:
                if current_idx in sorted_indices:
                    transformed_sample = self.batch_transform(rlds_batch)
                    self._cached_samples[current_idx] = transformed_sample
                    cached_count += 1
                    # print(f"已缓存第 {current_idx} 个样本")
                    
                    # 如果已经缓存了所有需要的样本，可以提前退出
                    if cached_count == len(sorted_indices):
                        break
                
                current_idx += 1
                
        except StopIteration:
            pass
        
        # 检查是否所有请求的样本都已缓存
        missing_indices = set(self.sample_index) - set(self._cached_samples.keys())
        if missing_indices:
            raise IndexError(f"样本索引 {missing_indices} 超出范围 (数据集长度={self.dataset_length})")
        
        print(f"总共缓存了 {len(self._cached_samples)} 个样本，索引为: {list(self._cached_samples.keys())}")


    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    # how to use this inferface
    def __iter__(self) -> Dict[str, Any]:
        if self.single_sample_mode:
            # 循环返回缓存的样本列表
            while True:
                # for idx in self.sample_index:
                #     yield self._cached_samples[idx]
                sample_idx = np.random.choice(self.sample_index)
                yield self._cached_samples[sample_idx]
        else:
            for rlds_batch in self.dataset.as_numpy_iterator():
                # ['observation', 'task', 'action', 'dataset_name', 'absolute_action_mask'])
                # print("***** rlds_batch",rlds_batch.keys())
                # ['image_primary', 'image_wrist', 'proprio', 'timestep', 'pad_mask_dict', 'pad_mask'])
                # print("***** rlds_batch['observation']",rlds_batch["observation"].keys())
                # print("***** rlds_batch['observation']['timestep']",rlds_batch["observation"]["timestep"])

                batch_trans = self.batch_transform(rlds_batch)
                # 
                # if self.data_mix.startswith("libero") and self.batch_transform.use_proprio:
                #     proprio = batch_trans["proprio"].copy() # writable copy
                #     # print(f"convert proprio shape from {proprio.shape} to 1D")

                #     proprio_lastone = convert_gripper_qpos_to_1d(proprio[:,-2:])
                #     batch_trans["proprio"] = proprio[:,:-1] # 去掉最后一个关节位置
                #     batch_trans["proprio"][:,-1] = proprio_lastone  # 保留最后一个夹爪关节位置作为夹爪开合距离
                yield batch_trans

    def __len__(self) -> int:
        return self.dataset_length
    

    def get(self, idx: int, rng) -> Dict[str, Any]:
        """
        map-style 接口，从头开始流式读取，跳到第 idx 个样本，再做 transform。
        注意:对大数据集来说,get(idx) 是 O(idx) 的。
        """
        raise NotImplementedError("Do not use get(), It is time-consuming, use __iter__ instead!")
        
        # if self.single_sample_mode:
        #     # 使用模运算循环访问缓存的样本
        #     # sample_idx = self.sample_index[idx % len(self.sample_index)]
        #     sample_idx = rng.choice(self.sample_index)
        #     return self._cached_samples[sample_idx]
        # else:
        #     iterator = self.dataset.as_numpy_iterator()
        #     try:
        #         rlds_batch = next(islice(iterator, idx, None))
        #     except StopIteration:
        #         raise IndexError(f"Index {idx} out of range (length={len(self)})")
        #     # return self.batch_transform(rlds_batch, rng)
        #     return self.batch_transform(rlds_batch)
    

    # === Explicitly Unused ===
    # def __getitem__(self, idx: int) -> None:
    #     raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

    # to support Molmo's dataset interface
    @classmethod  
    def download(cls, n_procs=1):  
        """Download method required by Molmo's dataset interface."""  
        logging.info("DummyRLDS: No download required for synthetic dataset")  
        pass  

class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out

class EpisodicPerStepRLDSDataset(RLDSDataset):
    """Return frame by frame, but episode comes from single dataset (not interleaved), and with episode_index and timestep_index."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."
        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self):
        # 逐“轨迹”读取，再拆成逐“帧”返回
        for episode_index, rlds_batch in enumerate(self.dataset.as_numpy_iterator()):
            T = rlds_batch["action"].shape[0]
            for i in range(T):
                # 取第 i 个时间步
                step = tree_map(lambda x: x[i], rlds_batch)

                # 先做原有 batch_transform
                out = self.batch_transform(step)

                # 附加 episode_index
                out["episode_index"] = int(episode_index)

                yield out

class EpisodicPerStepMultiRLDSDataset(RLDSDataset):
    """Iterate per-step while preserving episode boundaries and providing episode_index and timestep_index.

    Supports multiple datasets by sequentially iterating their episodes (no interleaving),
    producing a stable global episode_index across datasets.
    """

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]

        # Build per-dataset episodic datasets
        datasets, all_dataset_statistics = [], {}
        total_num_transitions = 0
        self.total_num_episodes = 0
        for dataset_kwargs in per_dataset_kwargs:
            dataset, _num_traj, dataset_statistics = make_single_dataset(
                dataset_kwargs,
                train=rlds_config["train"],
                traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
                frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
            )
            datasets.append(dataset)

            # Aggregate stats and length
            name = dataset_kwargs["name"]
            all_dataset_statistics[name] = dataset_statistics
            if "num_transitions" in dataset_statistics:
                total_num_transitions += int(dataset_statistics["num_transitions"])  # per-dataset transitions
            self.total_num_episodes += _num_traj

        # For compatibility with base class fields
        return datasets, total_num_transitions, all_dataset_statistics


        # Combine episodic datasets by random sampling across datasets
        # Equivalent to sampling episodes from all datasets with equal probability, then shuffle
        # combined = dl.DLataset.sample_from_datasets(datasets)
        # combined = combined.shuffle(rlds_config["shuffle_buffer_size"]//10)  # shuffle episodes globally
        # return combined, total_num_transitions, all_dataset_statistics



    def __iter__(self):
        # Iterate episodes sequentially across all datasets, yielding per-step samples
        episode_counter = 0
        for dataset in self.dataset:  # self.dataset is a list returned by make_dataset
            for rlds_batch in dataset.as_numpy_iterator():
                T = rlds_batch["action"].shape[0]

                # Randomize step order within episode
                step_indices = np.random.permutation(T).tolist()
                for i in step_indices:
                # for i in range(T):
                    step = tree_map(lambda x: x[i], rlds_batch)
                    out = self.batch_transform(step)

                    # Attach episode_index
                    out["episode_index"] = np.array(episode_counter)

                    if self.data_mix.startswith("libero") and self.batch_transform.use_proprio:
                        proprio = out["proprio"].copy() # writable copy
                        # print(f"convert proprio shape from {proprio.shape} to 1D")

                        proprio_lastone = convert_gripper_qpos_to_1d(proprio[:,-2:])
                        out["proprio"] = proprio[:,:-1] # 去掉最后一个关节位置
                        out["proprio"][:,-1] = proprio_lastone  # 保留最后一个夹爪关节位置作为夹爪开合距离

                    yield out

                episode_counter += 1


from torch.utils.data import DataLoader

def test_dataset():
    dataset = RLDSDataset(
    # dataset = EpisodicPerStepMultiRLDSDataset(
        data_root_dir=Path("/vast/users/xiaodan/zhangjian/datasets/modified_libero_rlds"),
        data_mix="libero_4_task_suites_no_noops_extra_10_task_8_6", # libero_spatial_no_noops, libero_4_task_suites_no_noops, libero_10_no_noops_task_8
        batch_transform=RLDSBatchTransform(),
        resize_resolution=(224, 224),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=False
    )

    # dataset = RLDSDataset(
    #     data_root_dir=Path("data/OXE"),
    #     data_mix="berkeley_cable_routing", # oxe_magic_soup_plus_minus_A1
    #     batch_transform=RLDSBatchTransform(),
    #     resize_resolution=(224, 224),
    #     shuffle_buffer_size=100_000,
    #     train=True,
    #     image_aug=False
    # )

    # import pickle
    # filehandler = open("test.obj","wb")
    # train_dataset=pickle.dumps(dataset)
    for sample in dataset:
        print(sample)
        break
    #    ("libero_spatial_no_noops", 1.0),
    #     ("libero_object_no_noops", 1.0),
    #     ("libero_goal_no_noops", 1.0),
    #     ("libero_10_no_noops", 1.0),
    # libero_4_task_suites_no_noops

    print("*** dataset length:",len(dataset))
    # print("*** dataset total_num_episodes:",dataset.total_num_episodes)
    for i,sample in enumerate(dataset):
        # print(sample)
        # print('** sample length',len(sample))
        # print('** sample[0].keys()',sample[0].keys())
        # for j in range(len(sample)):
            # print('** sample[%d].keys()'%(j),sample[j].keys())
        # print('** sample.keys()',sample.keys())
        print(i,len(dataset))
        # check if nan/inf in images,action and proprio
        if np.isnan(sample['images'][0]).any() or np.isinf(sample['images'][0]).any():
            print("** sample[images][0] has nan/inf")
        if np.isnan(sample['images'][1]).any() or np.isinf(sample['images'][1]).any():
            print("** sample[images][1] has nan/inf")
        if np.isnan(sample['action']).any() or np.isinf(sample['action']).any():
            print("** sample[action] has nan/inf")
        if np.isnan(sample['proprio']).any() or np.isinf(sample['proprio']).any():
            print("** sample[proprio] has nan/inf")

        # print('** sample["episode_index"]',sample["episode_index"])
        # print('** sample["timestep"]',sample["timestep"])
        # print('** sample[observation].keys()',sample["metadata"]['observation'].keys())  
        # print(sample['images'][0])
        # print("** sample[observation][image_primary].shape", sample['images'][0].shape,sample['images'][0].dtype)
        # print("** sample[observation][image_wrist].shape", sample['images'][1].shape,sample['images'][1].dtype)
        # # print("** sample[task]",sample['task'].keys(),type(sample['task']))
        # print("** action.shape", sample['action'].shape,sample['action'].dtype)
        # print("** proprio.shape", sample['proprio'].shape,sample['proprio'].dtype)
        # if i > 500:break
        # break  # Just print the first sample for testing

def test_dataloading():
    dataset = RLDSDataset(
        data_root_dir=Path("/path/to/data"),
        data_mix="dummy_rlds",
        batch_transform=RLDSBatchTransform(),
        resize_resolution=(224, 224),
        shuffle_buffer_size=1000,
        train=True,
        image_aug=False
    )

    val_dataloader = DataLoader(
            dataset,
            batch_size=2,
            sampler=None,
            # collate_fn=collator,
            num_workers=8,  # Important: Set to 0 if using RLDS, which uses its own parallelism
        )
    for batch in val_dataloader:
        print(batch)
        break

def test_diffusion_dataset_with_inheritance():
    # 假设您已经有了tokenizer和processor
    from transformers import AutoTokenizer, AutoProcessor,SiglipTextModel,SiglipVisionModel
    
    print('*** Using SiglipTextModel and SiglipVisionModel for testing ***')
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zhangjian/google/siglip-so400m-patch14-384")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/zhangjian/Qwen3/Qwen3-1.7B")
    processor = AutoProcessor.from_pretrained("/mnt/data/zhangjian/google/siglip-so400m-patch14-384")
    print("Tokenizer and Processor loaded successfully.")

    # print(" Building with tokenizer and processor...")
    # text_model = SiglipTextModel.from_pretrained("/mnt/data/zhangjian/google/siglip-so400m-patch14-384")
    # vision_model = SiglipVisionModel.from_pretrained("/mnt/data/zhangjian/google/siglip-so400m-patch14-384")
    # print("Text and Vision models loaded successfully.")
    
    # 创建继承版本的transform
    diffusion_transform = DiTActionRLDSBatchTransform(
        # 继承父类的参数
        predict_stop_token=True,
        use_wrist_image=True,
        use_proprio=True,
        # 新增的参数
        tokenizer=tokenizer,
        processor=processor,
    )
    
    dataset = RLDSDataset(
        data_root_dir=Path("/vast/users/xiaodan/zhangjian/datasets/modified_libero_rlds"),
        data_mix="libero_4_task_suites_no_noops", # libero_spatial_no_noops, 
        batch_transform=diffusion_transform,  # 使用继承的transform
        resize_resolution=(224, 224),
        shuffle_buffer_size=100_000,
        train=True,
        image_aug=False
    )
    
    for sample in dataset:
        # print("Sample keys:", sample.keys())
        # if "pixel_values" in sample:
        #     print("Pixel values shape:", sample["pixel_values"].shape)
        # if "input_ids" in sample:
        #     print("Input IDs shape:", sample["input_ids"].shape)
        # print("Action shape:", sample["action"].shape)
        # if "proprio" in sample:
        #     print("Proprio shape:", sample["proprio"].shape)
        if "text_attention_mask" in sample:
            print("Text attention mask shape:", sample["text_attention_mask"].shape,sample["text_attention_mask"].dtype)
            print(sample["text_attention_mask"])
        # 只打印第一个样本，避免输出过多
        break



if __name__ == "__main__":
    test_dataset()
    # test_diffusion_dataset_with_inheritance()
    # dataset = DummyRLDS(split="train", num_samples=10)
    # for i in range(len(dataset)):
    #     sample = dataset.get(i, np.random.default_rng())
    #     print(sample)