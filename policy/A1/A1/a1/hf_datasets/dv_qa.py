import json
from collections import defaultdict
from os.path import join
from typing import List

import datasets


class DvQaBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="dv_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                'image_id': datasets.Value("string"),
                "questions": datasets.Sequence(datasets.Features({
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }))
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_ids = dict(
            image_dir="1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ",
            annotation_dir="1VKYd3kaiCFziSsSv4SgQJ2T5m7jxuh5u",
        )
        file_urls = {
            k: f"https://drive.usercontent.google.com/download?id={v}&confirm=t"
            for k, v in file_ids.items()
        }
        downloaded_files = dl_manager.download_and_extract(file_urls)
        return [
            datasets.SplitGenerator(name=k, gen_kwargs=dict(downloaded_files, split=k))
            for k in [datasets.Split.TRAIN, "val_hard", "val_easy"]
        ]

    def _generate_examples(self, image_dir, annotation_dir, split):
        with open(join(annotation_dir, f"{split}_qa.json")) as f:
            data = json.load(f)
        grouped_by_image = defaultdict(list)
        for question in data:
            grouped_by_image[question["image"]].append(question)
        for image, questions in grouped_by_image.items():
            for q in questions:
                q.pop("image")
                q.pop("answer_bbox")
            yield image, dict(
                image_id=image,
                image=join(image_dir, "images", image),
                questions=[dict(question=q["question"], answer=q["answer"]) for q in questions]
            )


if __name__ == "__main__":
    DvQaBuilder().download_and_prepare()
