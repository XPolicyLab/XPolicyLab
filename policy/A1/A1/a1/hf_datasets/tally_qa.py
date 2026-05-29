import json
from collections import defaultdict
from os.path import join, exists

import datasets


class TallyQaBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version('1.0.0')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="tally_qa")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                "image_id": datasets.Value("int32"),
                "image/filename": datasets.Value("string"),
                "questions": datasets.Sequence(datasets.Features({
                    "answer": datasets.Value("int32"),
                    "issimple": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "data_source": datasets.Value("string"),
                    "question_id": datasets.Value("int64"),
                }))
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        data = dl_manager.download_and_extract(dict(
            train2014="http://images.cocodataset.org/zips/train2014.zip",
            val2014="http://images.cocodataset.org/zips/val2014.zip",
            VG_100K="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
            VG_100K_2="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
            src="https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true",
        ))
        return [
            datasets.SplitGenerator(name=name, gen_kwargs=dict(
                images=data, src=join(data["src"], f"{name}.json")))
            for name in ["train", "test"]
        ]

    def _generate_examples(self, src, images):
        with open(src) as f:
            data = json.load(f)
        grouped_by_image = defaultdict(list)
        for ex in data:
            grouped_by_image[ex["image"]].append(ex)

        for image, questions in grouped_by_image.items():
            image_id = questions[0]["image_id"]
            for q in questions:
                assert q.pop("image_id") == image_id
                assert q.pop("image") == image
                if "issimple" in q:
                    q["issimple"] = int(q["issimple"])
                else:
                    q["issimple"] = -1
            image_src, path = image.split("/")
            image_path = join(images[image_src], image_src, path)
            if not exists(image_path):
                import pdb; pdb.set_trace()
            yield image_id, {
                "image_id": image_id,
                "image": image_path,
                "questions": questions,
                "image/filename": image
            }
