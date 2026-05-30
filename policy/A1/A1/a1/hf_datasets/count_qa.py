import io
import json
from os.path import join
from typing import List

import datasets
import numpy as np
import pandas as pd
from PIL import Image

QAS_URL = "https://raw.githubusercontent.com/google-research/big_vision/46b2456f54b9d4f829d1925b78943372b376153d/big_vision/datasets/countbenchqa/data/countbench_paired_questions.json"
PARQUET_URL = "https://huggingface.co/datasets/nielsr/countbench/resolve/main/data/train-00000-of-00001-cf54c241ba947306.parquet"


class CountQaBuilder(datasets.GeneratorBasedBuilder):
    """CountQa dataset from PaliGemi, it is built by merging the CountBench image/count pairs with
    the natural language questions from the PaliGemma paper. Script adapted from:
    https://github.com/google-research/big_vision/blob/46b2456f54b9d4f829d1925b78943372b376153d/big_vision/datasets/countbenchqa/countbenchqa.py#L21
    """
    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="count_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                "question": datasets.Value("string"),
                "count": datasets.Value("int32"),
                "example_id": datasets.Value("int32"),
                "image_url": datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        qas_src, parquet = dl_manager.download(
            [QAS_URL, PARQUET_URL]
        )
        return [
            datasets.SplitGenerator(name="test", gen_kwargs=dict(qas=qas_src, parquet=parquet))
        ]

    def _generate_examples(self, qas, parquet):
        df = pd.read_parquet(parquet)
        with open(join(qas)) as f:
            questions = json.load(f)
        df['question'] = [x["question"] for x in questions]

        out = []
        for idx, row in df.iterrows():
            # Some entries have no image.
            if row['image'] is None:
                continue
            image_bytes = io.BytesIO(row['image']['bytes'])
            image = np.array(Image.open(image_bytes))
            if len(image.shape) != 3:
                continue  # Filter out one bad image.

            yield idx, {
                'image': image,
                'question': row['question'],
                'count': row['number'],
                'example_id': idx,
                'image_url': row['image_url'],
            }


if __name__ == "__main__":
    CountQaBuilder().download_and_prepare()

