# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VQA v2 loading script. Build from - https://huggingface.co/datasets/HuggingFaceM4/VQAv2/raw/main/VQAv2.py"""

import json
import os
from collections import defaultdict
from pathlib import Path

import datasets
import requests
from tqdm import tqdm

_URLS = {
    "questions": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
    },
    "annotations": {
        "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    }
}

_URLS_IMAGES = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "val": "http://images.cocodataset.org/zips/val2014.zip",
    "test": "http://images.cocodataset.org/zips/test2015.zip",
}

_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "v2_OpenEnded_mscoco_val2014_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
    },
    "annotations": {
        "train": "v2_mscoco_train2014_annotations.json",
        "val": "v2_mscoco_val2014_annotations.json",
    },
    "images": {
        "train": "train2014",
        "val": "val2014",
        "test": "test2015",
    },
}


def download_file(url, filename):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    # Get the total file size
    total_size = int(response.headers.get('content-length', 0))
    
    # Open the local file to write the downloaded content
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


class VQAv2BuilderMultiQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, vqa_source=None):
        if vqa_source is None:
            vqa_source = os.getcwd()
        self.vqa_source = vqa_source
        os.makedirs(self.vqa_source, exist_ok=True)

        super().__init__()

    def _info(self):
        features = datasets.Features(
            {
                "messages": [
                    {
                        "question": datasets.Value("string"),
                        "answers": [
                            datasets.Value("string")
                        ],
                        "style": datasets.Value("string"),
                        "question_id": datasets.Value("int64"),
                    }
                ],
                "image_id": datasets.Value("int64"),
                "image": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features
        )
    
    def _split_generators(self, dl_manager):
        downloaded_pointer = dl_manager.download(_URLS)
        downloaded_pointer["images"] = dl_manager.download(_URLS_IMAGES)

        # Download the images manually
        # if not os.path.exists(f"{self.vqa_source}/train2014.zip"):
        #     download_file(_URLS_IMAGES["train"], f"{self.vqa_source}/train2014.zip")
        # if not os.path.exists(f"{self.vqa_source}/val2014.zip"):
        #     download_file(_URLS_IMAGES["val"], f"{self.vqa_source}/val2014.zip")
        # if not os.path.exists(f"{self.vqa_source}/test2015.zip"):
        #     download_file(_URLS_IMAGES["test"], f"{self.vqa_source}/test2015.zip")
        #
        downloaded_pointer["images"] = {
            "train": f"{self.vqa_source}/train2014.zip",
            "val": f"{self.vqa_source}/val2014.zip",
            "test": f"{self.vqa_source}/test2015.zip",
        }

        data_dir = dl_manager.extract(downloaded_pointer)
        
        gen_kwargs = {}
        for split_name in ["val", "test", "train"]:
            split_gen_kwargs = {}
            for dir_name in list(set(_URLS.keys()) | set(["images"])):
                if split_name in data_dir[dir_name]:
                    split_gen_kwargs[f"{dir_name}_path"] = Path(data_dir[dir_name][split_name]) / _SUB_FOLDER_OR_FILE_NAME[dir_name][split_name]
                else:
                    split_gen_kwargs[f"{dir_name}_path"] = None
            gen_kwargs[split_name] = split_gen_kwargs
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=gen_kwargs["train"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=gen_kwargs["val"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=gen_kwargs["test"],
            ),
        ]

    def _generate_examples(self, questions_path, annotations_path, images_path):
        questions = json.load(open(questions_path, "r"))
        if annotations_path is not None:    
            dataset = json.load(open(annotations_path, "r"))
            qa = {ann["question_id"]: ann for ann in dataset["annotations"]}
        else:
            qa = None

        grouped_by_image = defaultdict(list)
        for q in questions["questions"]:
            grouped_by_image[q["image_id"]].append(q)

        for image_id, questions in grouped_by_image.items():
            messages = []
            for question in questions:
                anno = qa[question["question_id"]] if qa is not None else None
                if anno is not None:
                    messages.append(dict(
                        question=question["question"],
                        answers=[x["answer"] for x in anno["answers"]],
                        question_id=question["question_id"],
                        style="vqa2",
                    ))
                else:
                    messages.append(dict(
                        question=question["question"],
                        answers=None,
                        question_id=question["question_id"],
                        style="vqa2",
                    ))
            
            yield image_id, dict(
                messages=messages,
                image_id=image_id,
                image=str(images_path / f"COCO_{images_path.name}_{image_id:0>12}.jpg"),
            )
