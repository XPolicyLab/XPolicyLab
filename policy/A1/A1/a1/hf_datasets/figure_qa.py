import json
from collections import defaultdict
from os.path import join
from typing import List

import datasets


class FigureQaBuilder(datasets.GeneratorBasedBuilder):
    URLS = {
        "validation1": "https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-validation1-v1.tar.gz",
        "validation2": "https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-validation2-v1.tar.gz",
        "train": "https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-train1-v1.tar.gz",
        "test1": "https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-test1-v1.tar.gz",
        "test2": "https://download.microsoft.com/download/c/3/1/c315c9d8-8239-487e-a895-2d3ff805b508/figureqa-test2-v1.tar.gz"
    }
    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="figure_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                "image_index": datasets.Value("int64"),
                "questions": datasets.Sequence(datasets.Features({
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("int32"),
                }))
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        downloaded_files = dl_manager.download(self.URLS)
        extracted_files = dl_manager.extract(downloaded_files)
        splits = []
        for k, v in extracted_files.items():
            if k == "train":
                folder_name = "train1"
            elif "test" in k:
                folder_name = "no_annot_" + k
            else:
                folder_name = k
            splits.append(datasets.SplitGenerator(
                name=k,
                gen_kwargs={"source_dir": join(v, folder_name)}
            ))
        return splits

    def _generate_examples(self, source_dir):
        with open(join(source_dir, "qa_pairs.json"), "r") as f:
            data = json.load(f)
        grouped_by_image = defaultdict(list)
        for question in data["qa_pairs"]:
            grouped_by_image[question["image_index"]].append(question)
        out = []
        for image_index, questions_data in grouped_by_image.items():
            questions = []
            for q in questions_data:
                questions.append(dict(
                    question=q["question_string"],
                    answer=q.get("answer"),
                    question_id=q["question_id"],
                ))
            yield image_index, dict(
                image=join(source_dir, "png", str(image_index) + ".png"),
                image_index=image_index,
                questions=questions,
            )


if __name__ == "__main__":
    FigureQaBuilder().download_and_prepare()
