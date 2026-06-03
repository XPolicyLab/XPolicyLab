import json
from collections import defaultdict
from os.path import join
from typing import List

import datasets


class PlotQaBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="plot_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(decode=False),
                "image_index": datasets.Value("int64"),
                "questions": datasets.Sequence(datasets.Features({
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "question_id": datasets.Value("int64"),
                }))
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_ids = dict(
            train_images="1AYuaPX-Lx7T0GZvnsPgN11Twq2FZbWXL",
            validation_images="1i74NRCEb-x44xqzAovuglex5d583qeiF",
            test_images="1D_WPUy91vOrFl6cJUkE55n3ZuB6Qrc4u",
            train_annotations="1UNvkdq1YJD_ne6D3zbWtoQij37AtfpNp",
            validation_annotations="1y9RwXSye2hnX0e2IlfSK34ESbeVblhH_",
            test_annotations="1OQBkoe_dpvFs-jnWAdRdxzh1-hgNd9bO",
        )
        file_urls = {
            k: f"https://drive.usercontent.google.com/download?id={v}&confirm=t"
            for k, v in file_ids.items()
        }
        downloaded_files = dl_manager.download(file_urls)
        extracted_files = dl_manager.extract(file_urls)
        return [
            datasets.SplitGenerator(name=k, gen_kwargs={"image_dir": extracted_files[f"{k}_images"], "annotations": extracted_files[f"{k}_annotations"]})
            for k in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
        ]

    def _generate_examples(self, image_dir, annotations):
        with open(annotations, "r") as f:
            data = json.load(f)
        grouped_by_image = defaultdict(list)
        for question in data["qa_pairs"]:
            grouped_by_image[question["image_index"]].append(question)
        out = []
        for image_index, questions_data in grouped_by_image.items():
            questions = []
            for q in questions_data:
                answer = q["answer"]
                questions.append(dict(
                    question=q["question_string"],
                    answer=str(answer),
                    question_id=q["question_id"],
                ))
            with open(join(image_dir, "png", str(image_index) + ".png")) as f:
                image_bytes = f.read()
            yield image_index, dict(
                image=image_bytes,
                image_index=image_index,
                questions=questions,
            )


if __name__ == "__main__":
    PlotQaBuilder().download_and_prepare()
