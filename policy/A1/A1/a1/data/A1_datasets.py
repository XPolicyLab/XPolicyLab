import json, logging, os, re
from os.path import join, exists, isabs
import numpy as np
from a1.data.dataset import DatasetBase
from a1.tokenizer import get_special_token_ids,DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN, DEFAULT_PROPRIO_START_TOKEN, DEFAULT_PROPRIO_END_TOKEN
import random

if "DATA_DIR" in os.environ:
    A1_DATA_HOME = os.environ["DATA_DIR"]
else:
    raise ValueError("DATA_DIR is not set")

'''
Todo:
[x] agd20k
[x] blip_laion_cc
[x] sharerobot/affordance
[x] sharerobot/planning
[x] A0-Dataset-converted/converted_droid_molmo_sam2_molmo_planning
[x] A0-Dataset-converted/converted_droid_cotrack_molmo_planning
[x] RoboVQA
[X] A0-Dataset-converted/converted_maniskill_molmo
[X] A0-Dataset-converted/converted_hoi4d_planning
[x] pixmo-count
[ ] pixmo-point
[ ] hoi4d_trajectory
[ ] droid_trajectory
[ ] sharerobot/trajectory
[ ] 
...

'''


# since dataset is not orgnized in a standard way, we need to specify the path for each dataset
DATASET_PATHS = {
    "sr_planning": {
        "json": join(A1_DATA_HOME, "robobrain/stage2_3_0/planning_jsons"),
        "img": join(A1_DATA_HOME, "robobrain/stage_3_0/images")
    },
    "sr_affordance": {
        "json": join(A1_DATA_HOME, "robobrain/stage_4_0/a_json/"),
        "img": join(A1_DATA_HOME, "robobrain/stage_4_0/a_images/")
    },
    "sr_trajectory": {
        "json": join(A1_DATA_HOME, "robobrain/stage_4_0/t_json/"),
        "img": join(A1_DATA_HOME, "robobrain/stage_4_0/t_images/")
    },
    "agd20k": {
        "json": join(A1_DATA_HOME, "robobrain/stage_4_0/a_json/"),
        "img": join(A1_DATA_HOME, "robobrain/stage_4_0/a_images/")
    },
    "blip_laion_cc": {
        "json": join(A1_DATA_HOME, "robobrain/stage_1_0/"),
        "img": join(A1_DATA_HOME, "robobrain/stage_1_0/images")
    },
    "droid_cotrack_planning": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "")
    },
    "droid_molmo_sam2_planning": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "A0-Dataset/droid_molmo_sam2/")
    },
    "maniskill_planning": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "A0-Dataset/")
    },
    "comprehensive_trajectory": { # Existing, ensure img path is correct for its contents
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "") # This base path needs to contain subfolders like 'maniskill', 'droid_molmo_sam2' if images in JSON are relative like 'maniskill/...'
    },
    "RoboVQA": {
        "json": join(A1_DATA_HOME, "RoboVQA/"),
        "img_test": join(A1_DATA_HOME, "RoboVQA/test/"),
        "img_train": join(A1_DATA_HOME, "RoboVQA/train/")
    },
    "hoi4d_planning": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "")
    },
    # New Trajectory Dataset Paths
    "droid_molmo_sam2_trajectory": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "A0-Dataset/") # Assuming images are in 'droid_molmo_sam2' subfolder here, or adjust if path in JSON is absolute or different relative
    },
    "maniskill_trajectory": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "A0-Dataset/")
    },
    "droid_cotrack_trajectory": {
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"),
        "img": join(A1_DATA_HOME, "A0-Dataset")
    },
    "hoi4d_trajectory": { 
        "json": join(A1_DATA_HOME, "A0-dataset-converted-v1/"), # Placeholder, verify
        "img": join(A1_DATA_HOME, "A0-Dataset") # Placeholder, verify
    },
    "clever_math": {
        "json": join(A1_DATA_HOME, "clever-math/data/"),
        "img": join(A1_DATA_HOME, "clever-math/data/CLEVR_v1.0/images/"),
        # We use Reason-RFT's test images for clevrmath.
        "test_img": join(A1_DATA_HOME, "Reason-RFT/test_images/Visual-Counting-ID/")
    },
    "super_clevr": {
        "json": join(A1_DATA_HOME, "super-clever"),
        "img": join(A1_DATA_HOME, "super-clever/images")
    },
    "trance": {
        "json": join(A1_DATA_HOME, "TRANCE"),
        "img": join(A1_DATA_HOME, "TRANCE/image")
    },
    "oxe":{
        'data_dir': join(A1_DATA_HOME, "OXE")
    },
    "agibot":{
        "data_dir": join(A1_DATA_HOME, "AgibotWorld-Alpha"),
        "xlsx": "a1/data/vla/agibot/agibot_alpha_frame_ranges.xlsx",
        "norm_stat_file": "a1/data/vla/agibot/norm_stats.json"
    }
}

log = logging.getLogger(__name__)


def clean_and_prefix(text: str, n_placeholders: int) -> str:
    """Clean up the <image> placeholder in prompt and add appropriate prefix."""
    cleaned = re.sub(r'<\s*image\s*>', '', text).strip()
    if n_placeholders <= 0 or not cleaned:
        return cleaned
    prefix = " ".join(["<image>"] * n_placeholders)
    return f"{prefix} {cleaned}"


def is_existing_file(path):
    return bool(path) and os.path.exists(path)


def random_select_images(
    images: list[str], 
    n_ph: int, 
    rng = np.random.RandomState(42), 
    must_include_idx: int | None = None
) -> list[str]:
    """
    1. If images >= 2, select head and tail, and randomly select from the middle.
    2. Can specify an index to be included (must_include_idx).
    3. Returned images keep the original order.
    """
    if len(images) <= n_ph:
        return images

    indices = set()
    # Always include must_include_idx if specified
    if must_include_idx is not None:
        indices.add(must_include_idx)

    # Always include head and tail if enough images
    if len(images) >= 2:
        indices.add(0)
        indices.add(len(images) - 1)

    # Fill up to n_ph with random indices from the rest
    available = [i for i in range(len(images)) if i not in indices]
    n_to_select = n_ph - len(indices)
    if n_to_select > 0 and available:
        selected = rng.choice(available, size=min(n_to_select, len(available)), replace=False)
        indices.update(selected)

    # Sort indices to keep original order
    final_indices = sorted(indices)
    return [images[i] for i in final_indices]


class SRPlanning(DatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = "sr_planning"
        super().__init__(split, sample=sample)

    def load(self):
        examples = []
        json_dir = DATASET_PATHS["sr_planning"]["json"]
        img_dir = DATASET_PATHS["sr_planning"]["img"]
        
        # multiple tasks in different json files
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            full_path = join(json_dir, fname)
            log.info(f"[SR_Planning] loading {full_path}")
            with open(full_path, 'r') as f:
                data = json.load(f)

            for ex in data:
                images = [
                    im if exists(im)
                    else join(img_dir, im)
                    for im in ex["image"]
                    if im
                ]

                conversations = ex["conversations"]
                messages = []
                i = 0
                while i < len(conversations):
                    if conversations[i]["from"] != "human":
                        i += 1
                        continue

                    raw_prompt = conversations[i]["value"]
                    response = (conversations[i+1]["value"] 
                                if i+1 < len(conversations) and conversations[i+1]["from"]!="human" 
                                else "")
                    i += 2 if response else 1

                    messages.append({
                        "prompt": raw_prompt, # keep raw prompt for now
                        "text": response or "N/A",
                        "style": "action_planning"
                    })

                examples.append({
                    "images": images,
                    "message_list": messages,
                    "metadata": {
                        "id": ex["id"],
                        "task": ex["task"],
                        "selected_step": ex["selected_step"],
                        "answers": [m["text"] for m in messages if m["text"].strip()] # for inference evaluation
                    },
                })

        # train/val split
        # rng = np.random.RandomState(42)
        # rng.shuffle(examples)
        split_idx = len(examples) # whole for train now
        return examples[:split_idx] if self.split=="train" else examples[split_idx:]

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = self.data[item]
        images = example["images"]

        selected_images = random_select_images(
            images, 
            rng.randint(2, 4), 
            rng,
            int(example["metadata"]["selected_step"])  - 1,  # 1-indexed to 0-indexed
        )
    
        for m in example["message_list"]:
            m["prompt"] = clean_and_prefix(m["prompt"], len(selected_images))
        example["images"] = selected_images
        
        return example



class RoboVQA(DatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = "robovqa"
        super().__init__(split, sample=sample)

    def load(self):
        if self.split == "train":
            json_path = join(DATASET_PATHS["RoboVQA"]["json"], "train3m_llava.json")
            img_dir = DATASET_PATHS["RoboVQA"]["img_train"]
        else:
            json_path = join(DATASET_PATHS["RoboVQA"]["json"], "test_llava.json")
            img_dir = DATASET_PATHS["RoboVQA"]["img_test"]

        log.info(f"[RoboVQA] loading {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        examples = []
        for item in dataset:
            # images: list of image paths
            imgs = [
                im if exists(im) else join(img_dir, im)
                for im in item["image"]
                if im
            ]

            # select the first image if multiple
            
            conversations = item["conversations"]
            # extract prompt (from human; usually first in list)
            # robotvqa has only one conversation that is a question-answer pair
            prompt = conversations[0]['value'] if conversations and conversations[0].get("from") == "human" else ""
            response = conversations[1]['value'] if len(conversations) > 1 else ""

            example = {
                "images": imgs,
                "message_list": [{
                    "prompt": prompt,
                    "text": response,
                    "style": "action_planning"
                }],
                "metadata": {
                    "id": item["id"],
                    "answers": [response] if response else []
                }
            }
            examples.append(example)
        return examples

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = self.data[item]
        selected_images = random_select_images(example["images"], rng.randint(2, 4), rng)
        msg = example["message_list"][0]
        msg["prompt"] = clean_and_prefix(msg["prompt"], len(selected_images))
        example["images"] = selected_images
        return example

class OXEFAST(DatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None, data_name: str = "oxe_magic_soup_plus_minus", train_config = None):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = "OXEFAST"
        self.data_name = data_name
        self.train_config = train_config
        self.data_config = train_config.data
        self.tokenizer = train_config.model.get_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size()
        self.fast_skip_tokens = 2048
        self.action_tokenizer = train_config.model.get_action_tokenizer()
        self.sample_ratio = 1.0 
        # if 'libero' in self.data_name:
        #     self.sample_ratio = 1.0 
        # else:
        #     self.sample_ratio = 0.1
        # self.sample_ratio = 0.1
        super().__init__(split, sample=sample)

    def load(self):
        from a1.data.vla.rlds_datasets import RLDSDataset, RLDSBatchTransform
        batch_transform = RLDSBatchTransform(
            use_wrist_image=self.data_config.use_wrist_image,
            use_proprio=self.data_config.use_proprio,
        )
        dataset = RLDSDataset(
            data_root_dir=DATASET_PATHS['oxe']['data_dir'], 
            data_mix = self.data_name,
            batch_transform=batch_transform,
            resize_resolution=self.train_config.model.vision_backbone.image_default_input_size,
            shuffle_buffer_size=100_000,
            train="train",
            image_aug=True,
            sample_ratio=self.sample_ratio,
            # single_sample_mode=True,# just for Model convergence  of test some samples
            )
        self.dataset = dataset
        self.dataset_iterator = iter(self.dataset)
        if self.split == 'train':   
            return [1]*len(self.dataset)
        else:
            return [1]*(2**9)

    def _build_action_tokens(self, actions):
        """
        Construct the action token sequence of ACTION_DIMS*NUM_ACTIONS_CHUNK, 
        predict the next NUM_ACTIONS_CHUNK steps of actions, and insert them at the end of the sequence.
        """  
        tokens = self.action_tokenizer(actions)   
        if isinstance(tokens, list):
            tokens = np.array(tokens)     
        tokens = self.vocab_size - 1 - self.fast_skip_tokens - tokens
        return tokens

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        # Ensure iterator exists and auto-reset on exhaustion to avoid StopIteration bubbling up
        if not hasattr(self, "dataset_iterator") or self.dataset_iterator is None:
            self.dataset_iterator = iter(self.dataset)
        while True:
            try:
                example = next(self.dataset_iterator)
                break
            except StopIteration:
                # Reinitialize iterator when exhausted (supports effectively infinite sampling)
                self.dataset_iterator = iter(self.dataset)
                continue
        proprio = example['proprio']
        if proprio is not None:
            # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
            discretized_state = np.digitize(proprio, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

            # Convention: prefix includes prompt and string-representation of state, followed by ';'
            state_str = " ".join(map(str, discretized_state))
            # example['question'] = f"Task: {example['question']}, State: {state_str};\n"
            example['question'] += DEFAULT_PROPRIO_START_TOKEN + state_str + DEFAULT_PROPRIO_END_TOKEN
        action = example['action']
        action_tokens = self._build_action_tokens(action)   
        if len(action_tokens.shape) == 2:
            action_tokens = action_tokens.squeeze(0)
        action_tokens = action_tokens.tolist()
        action_tokens = self.tokenizer.decode(action_tokens)
        example['answer'] += DEFAULT_ACT_START_TOKEN + action_tokens + DEFAULT_ACT_END_TOKEN
        
        return example

class AgibotFAST(DatasetBase):
    def __init__(self,  split: str = "train", sample: int | None = None, data_name: str = "agibot-alpha", train_config = None):
        self.split = split
        self.identifier = "agibot"
        self.data_name = data_name
        assert self.data_name in ["agibot-alpha"]
        self.train_config = train_config
        self.data_config = train_config.data
        self.tokenizer = train_config.model.get_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size()
        self.fast_skip_tokens = 2048
        self.action_tokenizer = train_config.model.get_action_tokenizer()
        self.sample_ratio = 1.0 
        super().__init__(split, sample=sample)

    def load(self):
        from a1.data.vla.agibot_datasets import RobotDatasetReader
        from a1.vla.constants import ACTION_DIMS, NUM_ACTIONS_CHUNK
        dataset = RobotDatasetReader(DATASET_PATHS["agibot"]["data_dir"], DATASET_PATHS["agibot"]["xlsx"], DATASET_PATHS["agibot"]["norm_stat_file"], NUM_ACTIONS_CHUNK)
        self.dataset = dataset
        self.dataset_iterator = iter(self.dataset)
        if self.split == 'train':   
            return [1]*len(self.dataset)
        else:
            return [1]*(2**9)
    
    def _build_action_tokens(self, actions):
        """
        Construct the action token sequence of ACTION_DIMS*NUM_ACTIONS_CHUNK, 
        predict the next NUM_ACTIONS_CHUNK steps of actions, and insert them at the end of the sequence.
        """  
        tokens = self.action_tokenizer(actions)   
        if isinstance(tokens, list):
            tokens = np.array(tokens)     
        tokens = self.vocab_size - 1 - self.fast_skip_tokens - tokens
        return tokens

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = next(self.dataset_iterator)
        action = example['action']
        proprio = example['state']

        return_dict = {  
            # "image": image_primary,  
            # "images":[image_primary,image_wrist],
            # "question": f"What action should the robot take to {instruction}?", # is the question necessary?
            "question": example['task_name'], 
            # "message_list": conversation,  
            "answer": "Action",
            "style": "action",
            "action": action.copy(), 
            "proprio": proprio if self.data_config.use_proprio else None,

            "timestep": 0.1,


            "metadata": {  
                "dataset_name": self.data_name,  
                "instruction": example['task_name'],  
                "action": action.copy(), 
                # "observation": observation,
                # "absolute_action_mask": absolute_action_mask,
            }
        } 
        return_dict["images"] = example['image']
        example = return_dict
        proprio = example['proprio']
        if proprio is not None:
            # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
            discretized_state = np.digitize(proprio, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

            # Convention: prefix includes prompt and string-representation of state, followed by ';'
            state_str = " ".join(map(str, discretized_state))
            # example['question'] = f"Task: {example['question']}, State: {state_str};\n"
            example['question'] += DEFAULT_PROPRIO_START_TOKEN + state_str + DEFAULT_PROPRIO_END_TOKEN
        action = example['action']
        action_tokens = self._build_action_tokens(action)   
        if len(action_tokens.shape) == 2:
            action_tokens = action_tokens.squeeze(0)
        action_tokens = action_tokens.tolist()
        action_tokens = self.tokenizer.decode(action_tokens)
        example['answer'] += DEFAULT_ACT_START_TOKEN + action_tokens + DEFAULT_ACT_END_TOKEN
        return example

class BlipLaionCC(DatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        self.split = split
        self.identifier = "blip_laion_cc"
        super().__init__(split, sample=sample)

    def load(self):
        json_dir = DATASET_PATHS["blip_laion_cc"]["json"]
        json_path = join(json_dir, "blip_laion_cc_sbu_558k.json")
        img_dir = DATASET_PATHS["blip_laion_cc"]["img"]

        log.info(f"[BlipLaionCC] loading {json_path}")

        with open(json_path, 'r') as f:
            dataset = json.load(f)
        data = []
        for item in dataset:
            if not exists(item['image']):
                item['image'] = join(img_dir, item['image'])
            
            conversations = item['conversations']
            # extract prompt (from human; usually first in list)
            prompt = conversations[0]['value'] if conversations and conversations[0].get("from") == "human" else ""
            response = conversations[1]['value'] if len(conversations) > 1 else ""

            entry = {
                "image": item["image"],
                "prompt": prompt,
                "text": response,
                "metadata": {
                    "id": item["id"]
                }
            }
            data.append(entry)

        return data

    def get(self, item, rng):
        entry = self.data[item]
        imgs = [entry["image"]]
        sel = random_select_images(imgs, 3, rng)  # will repeat head/tail if only one
        raw = re.sub(r'^(?:<\s*image\s*>\\s*)+', '', entry["prompt"]).strip()
        prm = clean_and_prefix(raw, len(sel))
        return {
            "image": sel[0],
            "prompt": prm,
            "text": entry["text"],
            "metadata": entry["metadata"],
            "style": self.identifier
        }


class BboxDatasetBase(DatasetBase):
    '''
    base class for datasets with similar bbox-related structures
    '''
    def __init__(self, split: str, identifier: str, json_filter: str, split_ratio: float = 0.95, sample: int | None = None):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = identifier
        self.json_filter = json_filter
        self.split_ratio = split_ratio
        super().__init__(split, sample=sample)

    def load(self):
        examples = []
        json_dir = DATASET_PATHS[self.identifier]["json"]
        img_dir = DATASET_PATHS[self.identifier]["img"]
        
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(self.json_filter):
                continue
            full_path = join(json_dir, fname)
            log.info(f"[{self.identifier.upper()}] loading {full_path}")
            with open(full_path, 'r') as f:
                data = json.load(f)

            for ex in data:
                img_path = ex["image_path"]
                if not exists(img_path):
                    img_path = join(img_dir, img_path)

                question = clean_and_prefix(ex["instruction"], 1)
                meta_data = ex["meta_data"]

                W = meta_data["original_width"]
                H = meta_data["original_height"]

                # Normalize bbox values to 0–100
                x = round(ex["affordance"]["x"] / W * 100, 1)
                y = round(ex["affordance"]["y"] / H * 100, 1)
                wid = round(ex["affordance"]["width"] / W * 100, 1)
                hei = round(ex["affordance"]["height"] / H * 100, 1)

                answer = f'<bbox x="{x}" y="{y}" width="{wid}" height="{hei}" alt="affordance">affordance</bbox>'

                examples.append({
                    "image": img_path,
                    "prompt": question,
                    "text": answer,
                    "style": "affordance_detection",
                    "metadata": {
                        "id": ex["id"],
                        "original_dataset": meta_data["original_dataset"],
                        "original_height": H,
                        "original_width": W,
                        "img_path": img_path,
                        "source_file": fname,
                        "answers": [answer]
                    }
                })

        # train/test split
        split_idx = int(len(examples) * self.split_ratio)
        train_ex = examples[:split_idx]
        test_ex = examples[split_idx:]
        
        return train_ex if self.split == "train" else test_ex

    def get(self, item, rng):
        return self.data[item]


class SRAffordance(BboxDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split, 
            identifier="sr_affordance", 
            json_filter="affordance.json",
            split_ratio=1, # for train
            sample=sample
        )


class AGD20K(BboxDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split, 
            identifier="agd20k", 
            json_filter="agd20k.json",
            split_ratio=0, # whole for test
            sample=sample
        )



# For one split.
class PlanningDatasetBase(DatasetBase):
    '''
    # base class for planning-related datasets
    '''
    def __init__(self, split: str, identifier: str, json_filename: str, sample: int | None = None):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = identifier
        self.json_filename = json_filename
        super().__init__(split, sample=sample)

    def load(self):
        examples = []
        json_dir = DATASET_PATHS[self.identifier]["json"]
        img_dir = DATASET_PATHS[self.identifier]["img"]
        json_path = join(json_dir, self.json_filename)

        log.info(f"[{self.identifier.upper()}] loading {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        for ex in data:
            imgs = [
                im if exists(im)
                else join(img_dir, im)
                for im in ex["image"]
                if im 
            ]

            conversations = ex["conversations"]
            messages = []
            i = 0
            while i < len(conversations):
                if conversations[i]["from"] != "human":
                    i += 1
                    continue

                raw_prompt = conversations[i]["value"]
                response = (conversations[i+1]["value"] 
                            if i+1 < len(conversations) and conversations[i+1]["from"]!="human" 
                            else "")
                i += 2 if response else 1

                messages.append({
                    "prompt": raw_prompt,
                    "text": response or "N/A",
                    "style": "action_planning"
                })

            examples.append({
                "images": imgs,
                "message_list": messages,
                "metadata": {
                    "id": ex["id"],
                    "task": ex["task"],
                    "selected_step": ex["selected_step"],
                    "answers": [m["text"] for m in messages if m["text"].strip()] # for inference evaluation
                },
            })

        # All data goes to train
        split_idx = int(len(examples)) 
        return examples[:split_idx] if self.split=="train" else examples[split_idx:]

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = self.data[item]
        images = example["images"]
        selected_step = int(example["metadata"]["selected_step"]) - 1  # 1-indexed to 0-indexed
        selected_images = random_select_images(
            images, 
            rng.randint(2, 4), 
            rng,
            selected_step
        )
        for m in example["message_list"]:
            m["prompt"] = clean_and_prefix(m["prompt"], len(selected_images))
        example["images"] = selected_images
        return example


class DroidCotrackPlanning(PlanningDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="droid_cotrack_planning",
            json_filename="a0_droid_cotrack_planning.json",
            sample=sample
        )


class DroidMolmoSam2Planning(PlanningDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="droid_molmo_sam2_planning",
            json_filename="a0_droid_molmo_sam2_planning.json",
            sample=sample
        )


class ManiskillPlanning(PlanningDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="maniskill_planning",
            json_filename="a0_maniskill_planning.json",
            sample=sample
        )

class Hoi4DPlanning(PlanningDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__( 
            split=split,
            identifier="hoi4d_planning",
            json_filename="hoi4d_frame_planning.json",
            sample=sample
        )
        for i, item in enumerate(self.data):
            for j, path in enumerate(item["images"]):
                pattern = r'(hoi4d_frame/[^/]+)_([^/]+)_([^/]+)_([^/]+)_([^/]+)_([^/]+)_([^/]+)/'
                replacement = r'\1/\2/\3/\4/\5/\6/\7/'
                self.data[i]["images"][j] = re.sub(pattern, replacement, self.data[i]["images"][j])


class TrajectoryDatasetBase(DatasetBase):
    def __init__(self, split: str, identifier: str, json_filename: str, sample: int | None = None, split_ratio: float = 1.0):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = identifier
        self.json_filename = json_filename
        self.split_ratio = split_ratio
        super().__init__(split, sample=sample)

    def load(self):
        examples = []
        
        if self.identifier not in DATASET_PATHS:
            log.error(f"Dataset identifier '{self.identifier}' not found in DATASET_PATHS. Available: {list(DATASET_PATHS.keys())}")
            assert False, f"Dataset identifier '{self.identifier}' not found in DATASET_PATHS. Available: {list(DATASET_PATHS.keys())}"

        json_dir = DATASET_PATHS[self.identifier]["json"]
        img_base_dir = DATASET_PATHS[self.identifier]["img"]
        json_path = join(json_dir, self.json_filename)

        if not exists(json_path):
            assert False, f"JSON file not found: {json_path} for dataset {self.identifier}"

        log.info(f"[{self.identifier.upper()}] loading {json_path}")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            # log.error(f"Error decoding JSON from {json_path}: {e}")
            assert False, f"Error decoding JSON from {json_path}: {e}"
        
        if not isinstance(data, list):
            log.error(f"Expected a list of examples from {json_path}, but got {type(data)}")
            assert False, f"Expected a list of examples from {json_path}, but got {type(data)}"

        for ex_idx, ex in enumerate(data):
            img_path_suffix = ex.get("image") or ex.get("image_path")
            if not img_path_suffix:
                assert False, f'No image path for example ID {ex}.'

            if isabs(img_path_suffix) and exists(img_path_suffix):
                full_img_path = img_path_suffix
            else:
                # If img_path_suffix is like "droid_molmo_sam2/images/...", 
                # ensure img_base_dir is the parent of "droid_molmo_sam2" folder.
                # Example: if img_base_dir = "/path/to/A0-Dataset/" and suffix = "droid_molmo_sam2/images/..."
                # then full_img_path = "/path/to/A0-Dataset/droid_molmo_sam2/images/..."
                full_img_path = join(img_base_dir, img_path_suffix)
            
            instruction = ex['instruction']
            points_data = ex.get('points') or ex.get('trajectory')
            if points_data is None:
                assert False, f"No points data found for ID {ex.get('id', f'index_{ex_idx}')}."

            point_strs = '<points'
            for p_idx, p_coords in enumerate(points_data):
                if isinstance(p_coords, list) and len(p_coords) == 2:
                    try:
                        x = round(float(p_coords[0]) * 100, 1)
                        y = round(float(p_coords[1]) * 100, 1)
                        point_strs += f' x{p_idx+1}="{x}" y{p_idx+1}="{y}"'
                    except (ValueError, TypeError) as e:
                        assert False, f"Invalid point data for ID {ex.get('id', f'index_{ex_idx}')} point {p_idx}: {p_coords}. Error: {e}"
                else:
                    assert False, f"Malformed point entry for ID {ex.get('id', f'index_{ex_idx}')} point {p_idx}: {p_coords}."
            point_strs += ' alt="trajectory">trajectory</points>'

            messages = [{
                "prompt": instruction,
                "text": point_strs,
                "style": "trajectory_prediction"
            }]

            metadata = {
                "id": ex['id'],
                "meta_data": ex['meta_data'],
                "answers": [point_strs],  # For evaluation, keep the full point string
                "img_path": full_img_path,
                "original_width": ex['meta_data']['original_width'],
                "original_height": ex['meta_data']['original_height']
            }
            if "original_dataset" not in metadata["meta_data"]:
                if "original_dataset" in ex:
                    metadata["meta_data"]["original_dataset"] = ex["original_dataset"]
                elif "task_name" in metadata.get("meta_data", {}):
                     metadata["meta_data"]["original_dataset"] = metadata["meta_data"]["task_name"]

            examples.append({
                "image": full_img_path,
                "message_list": messages,
                "metadata": metadata,
            })
        
        n_total = len(examples)
        # random.Random(42).shuffle(examples) # Optional: shuffle for consistent train/test splits if not handled by dataloader

        if self.split == "train":
            split_idx_end = int(n_total * self.split_ratio)
            return examples[:split_idx_end]
        else: # "test" or "validation"
            split_idx_start = int(n_total * self.split_ratio)
            return examples[split_idx_start:]

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        
        if not self.data or item >= len(self.data):
            raise IndexError(f"Index {item} out of bounds for {self.identifier} (split {self.split}) size {len(self.data)}")

        example_data = self.data[item]
        example = {
            "image": example_data["image"],
            "message_list": [msg.copy() for msg in example_data["message_list"]],
            "metadata": example_data["metadata"].copy() 
        }
        
        num_images_for_prompt = 0
        if example.get("image") and is_existing_file(example["image"]): # Check if image exists
            example["images"] = [example["image"]]
            num_images_for_prompt = 1
        else:
            if example.get("image"): # Path was provided but file doesn't exist
                log.debug(f"Image file not found: {example['image']} for item {item} in {self.identifier}")
                assert False, f"Image file not found: {example['image']} for item {item} in {self.identifier}"

        if example["message_list"]:
            raw_prompt = example["message_list"][0]["prompt"]
            example["message_list"][0]["prompt"] = clean_and_prefix(raw_prompt, num_images_for_prompt)
        
        example["style"] = example["message_list"][0].get("style", "trajectory_prediction") if example["message_list"] else "trajectory_prediction"
        
        return example

class SRTrajectory(TrajectoryDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="sr_trajectory", 
            json_filename="trajectory.json", # Assumes this file exists
            sample=sample,
            split_ratio=0.95
        )

class DroidMolmoSam2Trajectory(TrajectoryDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="droid_molmo_sam2_trajectory", 
            json_filename="a0_droid_molmo_sam2_10samples_optimized.json", # Assumes this file exists
            sample=sample,
            split_ratio=0.95
        )

class ManiskillTrajectory(TrajectoryDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="maniskill_trajectory", 
            json_filename="a0_maniskill_10samples_optimized.json", # Assumes this file exists
            sample=sample,
            split_ratio=0.95
        )

class DroidCotrackTrajectory(TrajectoryDatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        super().__init__(
            split=split,
            identifier="droid_cotrack_trajectory", 
            json_filename="a0_droid_cotrack_10samples_optimized.json",
            sample=sample,
            split_ratio=0.95
        )

class CleverMath(DatasetBase):
    def __init__(self, split: str = "train", sample: int | None = None):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.identifier = "clever_math"
        super().__init__(split, sample=sample)

    def load(self):
        if self.split == "train":
            json_path = join(DATASET_PATHS["clever_math"]["json"], "CLEVR_train_llava_questions.json")
            img_dir = join(DATASET_PATHS["clever_math"]["img"], "train")
        else:
            # In test split, no answer is provided, we use the validation set.
            json_path = join(DATASET_PATHS["clever_math"]["json"], "Visual-Counting-id-test-1k.json")
            img_dir = join(DATASET_PATHS["clever_math"]["test_img"])

        log.info(f"[CleverMath] loading {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        examples = []
        for item in dataset:
            if isinstance(item['image'], list):
                imgs = [
                    im if exists(im) else join(img_dir, im)
                    for im in item['image']
                ]
            else:
                imgs = [item['image'] if exists(item['image']) else join(img_dir, item['image'])]
            if self.split == "train":
                conversations = item["conversations"]
                prompt = conversations[0]['value'] if conversations and conversations[0].get("from") == "human" else ""
                response = conversations[1]['value'] if len(conversations) > 1 else ""
            else:
                prompt = item["problem"]
                response = item["answer"]

            example = {
                "images": imgs,
                "prompt": prompt,
                "text": response,
                "style": self.identifier,
                "metadata": {
                    "id": item["id"],
                    "answers": [response] if response else []
                }
            }
            examples.append(example)
        return examples

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = self.data[item]
        selected_images = random_select_images(example["images"], rng.randint(2, 4), rng)
        # Unchange the position of <image> in the prompt.
        # example["prompt"] = clean_and_prefix(example["prompt"], len(selected_images))
        example["images"] = selected_images
        return example

class SuperCLEVR(DatasetBase):
    def __init__(self, split: str = "test", sample: int | None = None):
        assert split in ["test", "val", "validation"], "SuperCLEVR is intended for test or validation."
        self.split = split
        self.identifier = "super_clevr"
        super().__init__(split, sample=sample)

        self.data = self.load()
    def load(self):
        json_path = join(DATASET_PATHS["super_clevr"]["json"], "Visual-Counting-ood-test-1k.json")
        img_dir = DATASET_PATHS["super_clevr"]["img"]


        log.info(f"[SuperCLEVR] loading {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        examples = []
        for item in dataset:
            if isinstance(item['image'], list):
                imgs = [
                    im if exists(im) else join(img_dir, im)
                    for im in item['image']
                ]
            else:
                imgs = [item['image'] if exists(item['image']) else join(img_dir, item['image'])]

            prompt = item["problem"]
            response = item["answer"]

            # Since the `response` is long-form text, we need extract its gt for test.
            gt = self.extract_gt(response)
            example = {
                "images": imgs,
                "prompt": prompt,
                "text": response,
                "style": self.identifier,
                "metadata": {
                    "id": item["id"],
                    "answers": [response] if response else []
                }
            }
            examples.append(example)
        return examples
    
    def extract_gt(self, response):
        match = re.search(r"<CONCLUSION>\s*(.*?)\s*</CONCLUSION>", response, re.DOTALL | re.IGNORECASE)
        gt = match.group(1).strip() if match else ""
        return gt

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = self.data[item]
        selected_images = random_select_images(example["images"], rng.randint(2, 4), rng)
        example["prompt"] = clean_and_prefix(example["prompt"], len(selected_images))
        example["images"] = selected_images
        return example

class TRANCE(DatasetBase):
    def __init__(self, split: str = "test", sample: int | None = None):
        assert split in ["train", "test_id", "test_ood_left", "test_ood_right"]
        self.split = split
        self.identifier = "trance"
        super().__init__(split, sample=sample)

        self.data = self.load()
    def load(self):
        if self.split == "train":
            json_path = join(DATASET_PATHS["trance"]["json"], "trance-train-529k-LLaVA.json")
        if self.split == "test_id":
            json_path = join(DATASET_PATHS["trance"]["json"], "trance-id-test-1k.json")
        elif self.split == "test_ood_left":
            json_path = join(DATASET_PATHS["trance"]["json"], "trance-ood-left-test-1k.json")
        elif self.split == "test_ood_right":
            json_path = join(DATASET_PATHS["trance"]["json"], "trance-ood-right-test-1k.json")
        img_dir = DATASET_PATHS["trance"]["img"]


        log.info(f"[TRANCE] loading {json_path}")
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        examples = []
        for item in dataset:
            if isinstance(item['images'], list):
                imgs = [
                    im if exists(im) else join(img_dir, im)
                    for im in item['images']
                ]
            else:
                imgs = [item['images'] if exists(item['images']) else join(img_dir, item['images'])]

            conversations = item["conversations"]
            prompt = conversations[0]['value'] if conversations and conversations[0].get("from") == "human" else ""
            response = conversations[1]['value'] if len(conversations) > 1 else ""

            example = {
                "images": imgs,
                "prompt": prompt,
                "text": response,
                "style": self.identifier,
                "metadata": {
                    "id": item["id"],
                    "answers": [response] if response else []
                }
            }
            examples.append(example)
        return examples

    def get(self, item, rng):
        if rng is None:
            rng = np.random.RandomState(42)
        example = self.data[item]
        selected_images = random_select_images(example["images"], rng.randint(2, 4), rng)
        # example["prompt"] = clean_and_prefix(example["prompt"], len(selected_images))
        example["images"] = selected_images
        return example

if __name__ == "__main__":
    from pprint import pprint

    print("loading datasets...")

    # List of dataset classes and their respective splits
    datasets = [
        # (AGD20K, "test"),
        # (BlipLaionCC, "train"),
        # (SRAffordance, "train"),
        # (SRPlanning, "train"),
        # (DroidCotrackPlanning, "train"),
        # (DroidMolmoSam2Planning, "train"),
        # (ManiskillPlanning, "train"),
        # (Hoi4DPlanning, "train"),
        # (RoboVQA, "test"),
        # (DroidMolmoSam2Trajectory, "train"),
        # (ManiskillTrajectory, "train"),
        # (DroidCotrackTrajectory, "train"),
        (CleverMath, "train"),
        (CleverMath, "test"),
        # (SuperCLEVR, "train"),
        (SuperCLEVR, "test"),
        (TRANCE, "train"),
        (TRANCE, "test_id"),
        (TRANCE, "test_ood_left"),
        (TRANCE, "test_ood_right"),
    ]

    # Iterate through each dataset, load and print a sample
    for dataset_class, split in datasets:
        print(f"\n--- Loading {dataset_class.__name__} (split: {split}) ---")
        # Use a small sample size for quick testing if the dataset is large
        dataset = dataset_class(split=split, sample=5 if split=="train" else None) 
        if not dataset.data:
            print(f"No data loaded for {dataset_class.__name__} (split: {split}). Check JSON paths and content.")
            continue
        
        print(f"Loaded {len(dataset)} examples.")
        if len(dataset) > 0:
            sample_idx = 0 # random.randint(0, len(dataset) - 1)
            try:
                sample_item = dataset.get(sample_idx, None)
                
                # Determine the image path(s) carefully
                image_to_check = None
                if sample_item:
                    if "images" in sample_item and sample_item["images"]:
                        image_to_check = sample_item["images"][0]
                    elif "image" in sample_item and sample_item["image"]: # Should be under "images" after get()
                        image_to_check = sample_item["image"] 
                
                print(f"Sample {sample_idx}:")
                pprint(sample_item)
                if image_to_check:
                    print("Sample image path:", image_to_check)
                    print("Is existing file:", is_existing_file(image_to_check))
                else:
                    print("No image path in sample item or sample item is None.")
            except IndexError as e:
                print(f"Error getting sample from {dataset_class.__name__}: {e}")
        else:
            print(f"Dataset {dataset_class.__name__} (split: {split}) is empty after loading.")


    print("\nDone.")


