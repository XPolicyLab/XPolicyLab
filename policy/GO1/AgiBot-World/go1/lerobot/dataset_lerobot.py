import os
from typing import Any, Dict, List

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset
from PIL import Image

from go1.internvl.train.constants import IMG_END_TOKEN
from go1.internvl.train.dataset import build_transform, dynamic_preprocess, preprocess_internvl2_5
from go1.lerobot.dataset_transforms import Normalize, TransformedDataset, make_conversation


def tensor_to_pil(tensor):
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.dtype != torch.uint8:
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        else:
            tensor = tensor.clamp(0, 255).to(torch.uint8)

    # 转换为numpy并创建PIL图像
    numpy_img = tensor.cpu().numpy()
    return Image.fromarray(numpy_img)


class WrappedLeRobotDataset:
    def __init__(
        self,
        root,
        # internvl language related args
        text_tokenizer,
        # internvl vision related args
        num_image_token,
        transforms=None,
        action_chunk_size=30,
        space_args=None,
        debug=False,
        is_train=True,
        image_size=448,
        pad2square=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        normalize_type="imagenet",
        conversation_type=0,
        vis_frame=False,
        vis_dir=None,
        **kwargs,
    ):
        self.num_image_token = num_image_token
        self.text_tokenizer = text_tokenizer
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.conversation_type = conversation_type
        self.vis_frame = vis_frame
        self.vis_dir = vis_dir
        self.space_args = space_args
        self.action_key = space_args.space_repack["action"]
        self.state_key = space_args.space_repack["state"]
        self.root = root

        if len(root) == 1:
            # Single LeRobotDataset
            root = root[0]

            self.dataset_metas = [LeRobotDatasetMetadata(root=root, repo_id=os.path.basename(root))]

            # Debug mode: only use the first episode
            episode_ids = None
            if debug:
                episode_ids = [0]

            self.dataset = LeRobotDataset(
                repo_id=os.path.basename(root),
                root=root,
                episodes=episode_ids,
                delta_timestamps=(
                    {self.action_key: [t / self.dataset_metas[0].fps for t in range(action_chunk_size)]}
                    if action_chunk_size > 1
                    else None
                ),
                video_backend="pyav",
            )
            self.is_multi_dataset = False
            self.stats = self.dataset_metas[0].stats
        else:
            # Multiple LeRobotDatasets
            repo_ids = []
            for root in self.root:
                repo_ids.append(os.path.basename(root))

            # Debug mode: only use the first repo_id
            if debug and repo_ids:
                episodes = {repo_id: [0] for repo_id in repo_ids}

            self.dataset_metas = []
            for root_path in self.root:
                ds_meta = LeRobotDatasetMetadata(root=root_path, repo_id=os.path.basename(root_path))
                self.dataset_metas.append(ds_meta)

            self.dataset = MultiLeRobotDataset(
                root=os.path.dirname(self.root[0]),
                repo_ids=repo_ids,
                episodes=episodes if debug else None,
                delta_timestamps={
                    key: [t / dataset_meta.fps for t in range(action_chunk_size)]
                    for dataset_meta in self.dataset_metas
                    for key in [self.action_key]
                },
                video_backend="pyav",
            )
            self.is_multi_dataset = True
            self.stats = self.dataset.stats

        if transforms:
            trans_funcs = []
            for t in transforms:
                if t["type"] == "Normalize":
                    trans_funcs.append(Normalize(norm_stats=self.stats, key=[self.action_key, self.state_key]))
                else:
                    raise ValueError(f"Unknown transform: {t}")

            # Only apply transforms for MultiLeRobotDatasetrm /mnt/shimodi/code/lerobot_alpha/arm /mnt/shimodi/code/lerobot_alpha/alpha0/test_write.txtlpha0/test_write.txt
            if self.is_multi_dataset:
                for n, d in enumerate(self.dataset._datasets):
                    self.dataset._datasets[n] = TransformedDataset(
                        d, trans_funcs, num_frames=self.dataset._datasets[n].num_frames
                    )
            else:
                self.dataset = TransformedDataset(self.dataset, trans_funcs, num_frames=self.dataset.num_frames)

        self.meta = self.dataset_metas[0]
        self.debug = debug

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    @staticmethod
    def multi_image_get_item(
        raw_target: Dict[str, Any],
        img_transform,
        text_tokenizer,
        num_image_token,
        cam_keys: List[str] = [
            "cam_head_color",
            "cam_hand_right_color",
            "cam_hand_left_color",
        ],
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        image_size=448,
    ):
        images, num_tiles = [], []
        num_image = 0
        for cam_key in cam_keys:
            if cam_key in raw_target:
                num_image += 1
                if dynamic_image_size:
                    image = dynamic_preprocess(
                        raw_target[cam_key],
                        min_num=min_dynamic_patch,
                        max_num=max_dynamic_patch,
                        image_size=image_size,
                        use_thumbnail=use_thumbnail,
                    )
                    images += image
                    num_tiles.append(len(image))
                else:
                    images.append(raw_target[cam_key])
                    num_tiles.append(1)

        pixel_values = [img_transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [num_image_token * num_tile for num_tile in num_tiles]
        ntp_target = raw_target.get("ntp_target", "")
        conversation = [
            {"from": "human", "value": f"{'<image>'*num_image}{raw_target['final_prompt']}"},
            {"from": "gpt", "value": ntp_target},
        ]
        ret = preprocess_internvl2_5(
            "internvl2_5",
            [conversation],
            text_tokenizer,
            num_image_tokens,
            num_image=num_image,
            group_by_length=True,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret["attention_mask"] == 0, 1)
        image_end_token_id = text_tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret["input_ids"][0] == image_end_token_id).sum() == num_image, "image tokens are truncated"

        # Create the final return dictionary
        final_ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        return final_ret

    def __getitem__(self, index):
        raw_data = self.dataset[index]

        action = raw_data[self.action_key]
        if action.shape[1] == 1:
            action = action.squeeze(1)
        state = raw_data[self.state_key].reshape(1, raw_data[self.state_key].shape[-1])
        raw_target = {}
        if "cam_head_color" in self.space_args.space_repack:
            raw_target["cam_head_color"] = tensor_to_pil(
                raw_data[self.space_args.space_repack["cam_head_color"]].permute(1, 2, 0)
            )
        if "cam_hand_right_color" in self.space_args.space_repack:
            raw_target["cam_hand_right_color"] = tensor_to_pil(
                raw_data[self.space_args.space_repack["cam_hand_right_color"]].permute(1, 2, 0)
            )
        if "cam_hand_left_color" in self.space_args.space_repack:
            raw_target["cam_hand_left_color"] = tensor_to_pil(
                raw_data[self.space_args.space_repack["cam_hand_left_color"]].permute(1, 2, 0)
            )
        if "final_prompt" in self.space_args.space_repack:
            raw_target["final_prompt"] = raw_data[self.space_args.space_repack["final_prompt"]]
        else:
            raw_target["final_prompt"] = self.space_args.default_prompt
        raw_target["final_prompt"] = make_conversation(prompt=raw_target["final_prompt"])

        results = self.multi_image_get_item(
            raw_target=raw_target,
            img_transform=self.get_transform(),
            text_tokenizer=self.text_tokenizer,
            num_image_token=self.num_image_token,
            dynamic_image_size=self.dynamic_image_size,
            use_thumbnail=self.use_thumbnail,
            min_dynamic_patch=self.min_dynamic_patch,
            max_dynamic_patch=self.max_dynamic_patch,
            image_size=self.image_size,
        )
        results.update(
            {
                "action_gts": action,
                "state": state,
                "ctrl_freqs": torch.tensor([self.space_args.ctrl_freq], dtype=torch.float32),
            }
        )
        if self.debug:
            debug_data = {}
            if "cam_head_color" in raw_target:
                debug_data["cam_head_color"] = raw_target["cam_head_color"]
            if "cam_hand_right_color" in raw_target:
                debug_data["cam_hand_right_color"] = raw_target["cam_hand_right_color"]
            if "cam_hand_left_color" in raw_target:
                debug_data["cam_hand_left_color"] = raw_target["cam_hand_left_color"]
            if "observation.state" in raw_data:
                debug_data["raw_state"] = raw_data["observation.state"]
            if "final_prompt" in raw_target:
                debug_data["final_prompt"] = raw_target["final_prompt"]

            results.update(debug_data)

        return results

    def __len__(self):
        return len(self.dataset)
