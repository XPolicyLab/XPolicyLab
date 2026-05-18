import json
from os import listdir
from os.path import join
from typing import List

import datasets


class TabMwpBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="tab_mwp")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "choices": datasets.features.Sequence(datasets.Value("string")),
                "table_title": datasets.Value("string"),
                "table": datasets.Value("string"),
                "column_num": datasets.Value("int64"),
                "row_num": datasets.Value("int64"),
                "solution": datasets.Value("string"),
                "ques_type": datasets.Value("string"),
                "ans_type": datasets.Value("string"),
                "unit": datasets.Value("string"),
                "grade": datasets.Value("int32"),
                "example_id": datasets.Value("int64"),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        source = dl_manager.download_and_extract("https://codeload.github.com/lupantech/PromptPG/legacy.zip/main")
        return [
            datasets.SplitGenerator(name=k, gen_kwargs=dict(source=source, split=k))
            for k in ["train", "dev", "test"]
        ]

    def _generate_examples(self, source, split):
        files = listdir(source)
        assert len(files) == 1
        home = join(source, files[0], "data", "tabmwp")
        with open(join(home, f"problems_{split}.json")) as f:
            data = json.load(f)
        for example_id, example in data.items():
            assert example.pop("split") == split
            example["example_id"] = int(example_id)
            example["image"] = join(home, "tables", example_id + ".png")
            del example["table_for_pd"]
            yield example_id, example


if __name__ == "__main__":
    TabMwpBuilder().download_and_prepare()
