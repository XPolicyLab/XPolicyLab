import json
import os
from pathlib import Path
import datasets

import requests
from tqdm import tqdm
import json
import os
from pathlib import Path

import datasets
import requests
from tqdm import tqdm

_URLS = {
    "annotations": "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
}

_URLS_IMAGES = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
    "test": "http://images.cocodataset.org/zips/test2017.zip",
}

_SUB_FOLDER_OR_FILE_NAME = {
    "annotations": {
        "train": "aokvqa_v1p0_train.json",
        "val": "aokvqa_v1p0_val.json",
        "test": "aokvqa_v1p0_test.json",
    },
    "images": {
        "train": "train2017",
        "val": "val2017",
        "test": "test2017",
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


class AOkVqaBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, aokvqa_source=None):
        if aokvqa_source is None:
            aokvqa_source = os.getcwd()
        self.aokvqa_source = aokvqa_source
        os.makedirs(self.aokvqa_source, exist_ok=True)

        super().__init__()

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "question_id": datasets.Value("string"),
                "difficult_direct_answer": datasets.Value("bool"),
                "direct_answers": [
                    datasets.Value("string")
                ],
                "choices": [
                    datasets.Value("string")
                ],
                "correct_choice_idx": datasets.Value("int64"),
                "image": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features
        )
    
    def _split_generators(self, dl_manager):
        downloaded_pointer = dl_manager.download(_URLS)

        # Download the images manually
        if not os.path.exists(f"{self.aokvqa_source}/train2017.zip"):
            download_file(_URLS_IMAGES["train"], f"{self.aokvqa_source}/train2017.zip")
        if not os.path.exists(f"{self.aokvqa_source}/val2017.zip"):
            download_file(_URLS_IMAGES["val"], f"{self.aokvqa_source}/val2017.zip")
        if not os.path.exists(f"{self.aokvqa_source}/test2017.zip"):
            download_file(_URLS_IMAGES["test"], f"{self.aokvqa_source}/test2017.zip")

        downloaded_pointer["images"] = {
            "train": f"{self.aokvqa_source}/train2017.zip",
            "val": f"{self.aokvqa_source}/val2017.zip",
            "test": f"{self.aokvqa_source}/test2017.zip",
        }

        print(f"Downloaded files: {downloaded_pointer}. Extracting...")
        data_dir = dl_manager.extract(downloaded_pointer)
        
        gen_kwargs = {}
        for split_name in ["val", "test", "train"]:
            split_gen_kwargs = {}

            split_gen_kwargs["images_path"] = Path(data_dir["images"][split_name]) / _SUB_FOLDER_OR_FILE_NAME["images"][split_name]
            split_gen_kwargs["annotations_path"] = Path(data_dir["annotations"]) / _SUB_FOLDER_OR_FILE_NAME["annotations"][split_name]

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

    def _generate_examples(self, annotations_path, images_path):
        data = json.load(open(annotations_path, "r"))

        for ex in data:
            image_id = ex["image_id"]
            image_file_name = str(images_path / f"{image_id:0>12}.jpg")

            # TODO: Figure out the data path by running in runtime.
            # image_file_name = get_coco_image_file(image_id, split)
            
            yield ex['question_id'], dict(
                question=ex['question'],
                question_id=ex['question_id'],
                difficult_direct_answer=ex['difficult_direct_answer'],
                choices=ex['choices'],
                direct_answers=ex["direct_answers"] if "direct_answers" in ex else None,
                correct_choice_idx=ex['correct_choice_idx'] if "correct_choice_idx" in ex else None,
                image=image_file_name,
            )
