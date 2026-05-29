"""Datasets the load directly from source files,
Currently not used in favour of using HF datasets"""
import json
import logging
from collections import defaultdict
from os.path import exists
from os.path import join

from a1.data.dataset import DATA_HOME, DatasetBase
from a1.hf_datasets.android_control import AndroidControlBuilder
from a1.util import load_json

DOWNLOADS = join(DATA_HOME, "downloads")
INFOQA_SOURCE = join(DATA_HOME, "info_qa")
ST_QA_SRC = join(DATA_HOME, "scene-text")
COCO_IMAGES = join(DATA_HOME, "coco_images")
VG_IMAGES = join(DATA_HOME, "vg")
SCIENCE_QA_SRC = join(DATA_HOME, "science_qa")
CHARTQA_SOURCE = join(DATA_HOME, "chartqa")
DOCQA_SOURCE = join(DATA_HOME, "docqa")
TEXT_VQA_SRC = join(DATA_HOME, "text_vqa")
VQA_SOURCE = join(DATA_HOME, "vqa2")
OKVQA_SOURCE = join(DATA_HOME, "okvqa")
COCO_IMAGE_URL = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/"
TAB_WMP_SRC = join(DATA_HOME, "tabwmp")
TALLY_QA_SRC = join(DATA_HOME, "tally_qa")
AI2D_SRC = join(DATA_HOME, "ai2d")
ANDROID_CONTROL_SRC = join(DATA_HOME, "android")
COUNT_BENCH_QA_SRC = join(DATA_HOME, "countbench_qa")


def get_coco_image_file(image_id, split):
    if split == "test":
        subset = "test2015"
    elif split == "val":
        subset = "val2014"
    else:
        subset = "train2014"
    return join(COCO_IMAGES, f'{subset}/COCO_{subset}_{str(image_id).zfill(12)}.jpg')


def is_in_coco_val2017(image_id, _val2017=set()):
    if not _val2017:
        with open(join(COCO_IMAGES, f"2017_val_images_ids.json")) as f:
            _val2017.update(json.load(f))
    return image_id in _val2017


def get_coco_image_url(image_id, split):
    if split == "train":
        subset = "train2017"
    elif split == "val":
        if is_in_coco_val2017(image_id):
            subset = "val2017"
        else:
            subset = "train2017"
    else:
        raise ValueError(f"No public urls for images in split {split}")
    return f"https://s3.us-east-1.amazonaws.com/images.cocodataset.org/{subset}/{image_id}.jpg"


def _load_vqa2(q_src, a_src, split, multi_question, raw_answer=False):
    logging.info(f"Loading questions from {q_src}")
    q_data = load_json(q_src)["questions"]

    if a_src is not None:
        logging.info(f"Loading vqa2 answers from {a_src}")
        a_data = load_json(a_src)
        anno_map = {}
        for anno in a_data["annotations"]:
            anno_map[anno["question_id"]] = anno
    else:
        anno_map = None

    if split == "val":
        # Used to figure out the image URLs
        val2017 = set(load_json(join(COCO_IMAGES, f"2017_val_images_ids.json")))

    grouped_by_image = defaultdict(list)
    for q in q_data:
        grouped_by_image[q["image_id"]].append(q)

    data = []
    for image_id, questions in grouped_by_image.items():
        json_questions = []
        for question in questions:
            anno = None
            if anno_map is not None:
                anno = anno_map[question["question_id"]]
            q_data = dict(
                question=question["question"],
                question_type=None if anno is None else anno["question_type"],
                question_id=question["question_id"],
            )
            if anno is not None:
                if raw_answer:
                    q_data["answers"] = [x["raw_answer"] for x in anno["answers"]]
                else:
                    q_data["answers"] = [x["answer"] for x in anno["answers"]]
            json_questions.append(q_data)
        image = get_coco_image_file(image_id, split)
        metadata = dict(
            image_id=image_id,
        )
        if split != "test":
            if split == "val" and image_id in val2017:
                metadata["image_url"] = f"{COCO_IMAGE_URL}/val2017/{image_id}.jpg"
            else:
                metadata["image_url"] = f"{COCO_IMAGE_URL}/train2017/{image_id}.jpg"
        if not multi_question:
            for q in json_questions:
                q_metadata = dict(**metadata, example_id=q["question_id"])
                data.append(dict(
                    question=q["question"],
                    answers=q["answers"],
                    image=image,
                    metadata=q_metadata
                ))
        else:
            data.append(dict(
                questions=[{k: q[k] for k in ["question", "answers"]}
                           for q in json_questions],
                image=image,
                metadata=metadata
            ))
    return data


class Vqa2014Manual(DatasetBase):
    def __init__(self, split, multi_question=False):
        assert split in ["train", "validation", "test"]
        self.multi_question = multi_question
        super().__init__("vqa2_multi" if multi_question else "vqa2", split)

    def load(self):
        split = self.split
        if self.split == "validation":
            split = "val"
        q_src = f"{VQA_SOURCE}/v2_OpenEnded_mscoco_{split}2014_questions.json"
        a_src = None
        if split != "test":
            a_src = f"{VQA_SOURCE}/v2_mscoco_{split}2014_annotations.json"
        return _load_vqa2(q_src, a_src, split, self.multi_question)

    def get(self, item, rng):
        ex = self.data[item]
        if not self.multi_question:
            return dict(
                style="vqa2",
                answers=ex["answers"],
                metadata=ex["metadata"],
                image=ex["image"],
                question=ex["question"],
            )
        else:
            messages = []
            for q in ex["questions"]:
                messages.append(dict(
                    question=q["question"],
                    answers=q["answers"],
                    style="vqa2",
                ))
            return dict(
                metadata=ex["metadata"],
                image=ex["image"],
                message_list=messages
            )


class AOkVqaManual(DatasetBase):
    def __init__(self, split, direct_answer=False):
        suffix = "da" if direct_answer else "mc"
        self.direct_answer = direct_answer
        super().__init__(f"a_okvqa_{suffix}", split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        src = join(A_OK_VQA_SOURCE, f"aokvqa_v1p0_{split}.json")
        logging.info(f"Loading a_ok_vqa from {src}")
        with open(src) as f:
            data = json.load(f)

        out = []
        for ex in data:
            image_id = ex['image_id']
            image_file_name = get_coco_image_file(image_id, split)
            if split == "train":
                if not exists(image_file_name):
                    image_file_name = get_coco_image_file(image_id, "val")
                    assert exists(image_file_name), f"Missing expected file {image_file_name}"

            if self.direct_answer:
                if ex["difficult_direct_answer"] and split in ["val", "test"]:
                    continue
                out.append(dict(
                    image=image_file_name,
                    question=ex["question"],
                    answers=ex["direct_answers"],
                    metadata=dict(
                        example_id=ex["question_id"]
                    )
                ))
            else:
                out.append(dict(
                    image=image_file_name,
                    question=ex["question"],
                    options=ex["choices"],
                    answer_idx=ex.get("correct_choice_idx"),
                    metadata=dict(
                        example_id=ex["question_id"],
                    )
                ))
        return out

    def get(self, item, rng):
        return dict(**self.data[item], style=self.identifier)


class AndroidControl(DatasetBase):  # TODO needs a preparation script

    def download(self, n_procs=1):
        AndroidControlBuilder().download_and_prepare(num_proc=n_procs)

    def __init__(self, split, sample=None, mode="all"):
        assert split in ["train", "validation", "test"]
        self.split = split
        self.mode = mode
        super().__init__(f"android_control", split, sample=sample)

    def load(self):
        src = join(ANDROID_CONTROL_SRC, f"{self.split}.jsonl")
        logging.info(f"Loading android control from {src}")
        with open(src) as f:
            return f.readlines()

    def get(self, item, rng):
        data = self.data[item]
        ex = json.loads(data)
        ll, hl_ll, hl, hl_cot = [
            dict(
                prompt="low_level: " + ex["metadata/ll_instruction"],
                text=ex["metadata/target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["metadata/hl_instruction"] + " low_level: " + ex["metadata/ll_instruction"],
                text=ex["metadata/target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["metadata/hl_instruction"],
                text=ex["metadata/target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level_cot: " + ex["metadata/hl_instruction"],
                text="Plan: " + ex["metadata/ll_instruction"] + " Action: " + ex["metadata/target_action"],
                style="android_control"
            )
        ]
        example = dict(
            image=join(ANDROID_CONTROL_SRC, "images", ex["image"]),
            metadata=dict(
                target_action=ex["metadata/target_action"],
                target_box=ex["metadata/target_box"],
                ll_instruction=ex["metadata/ll_instruction"],
                hl_instruction=ex["metadata/hl_instruction"],
            )
        )
        if self.mode == "ll":
            example.update(ll)
        elif self.mode == "hl":
            example.update(hl)
        elif self.mode == "hl_ll":
            example.update(hl_ll)
        elif self.mode == "hl_cot":
            example.update(hl_cot)
        elif self.mode == "all":
            example["message_list"] = [ll, hl_ll, hl, hl_cot]
        else:
            raise NotImplementedError(self.mode)
        return example


class DocQaManual(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                split = "val"
            if split == "test":
                src = join(DOCQA_SOURCE, f"{split}_v1.0.json")
            else:
                src = join(DOCQA_SOURCE, f"{split}_v1.0_withQT.json")
            if not exists(src):
                raise ValueError(
                    "DocQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 1)"
                    f" please download and unzip the data into `{DOCQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__("doc_qa", split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        if self.split == "test":
            src = join(DOCQA_SOURCE, f"{split}_v1.0.json")
        else:
            src = join(DOCQA_SOURCE, f"{split}_v1.0_withQT.json")
        logging.info(f"Loading docqa data from {src}")
        with open(src) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            assert ex.pop("data_split") == split
            image_path = join(DOCQA_SOURCE, ex["image"])
            if self.split == "test":
                for k in ["answers", "question_types"]:
                    assert k not in ex
                    ex[k] = []
            out.append(dict(
                image=join(DOCQA_SOURCE, ex["image"]),
                question=ex["question"],
                answers=ex.get("answers"),
                metadata=dict(
                    doc_id=ex["docId"],
                    question_types=ex.get("question_types"),
                    example_id=ex["questionId"],
                ),
            ))
        return out

    def get(self, item, rng):
        return dict(self.data[item], style="doc_qa")


class ChartQa(DatasetBase):
    def __init__(self, split, select_answer="best", parts="both", weighted=False):
        self.select_answer = select_answer
        self.weighted = weighted
        assert split in ["train", "validation", "test"]
        assert parts in ["human", "augmented", "both"]
        self.parts = parts
        if weighted:
            assert parts == "both"
            identified = "chart_qa_weighted"
        else:
            identified = "chart_qa" if self.parts == "both" else f"chart_qa_{self.identifier}"
        super().__init__(identified, split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "val"
        examples = []
        if self.parts == "both":
            parts = ["human", "augmented"]
        else:
            parts = [self.parts]
        for part in parts:
            src = f"{CHARTQA_SOURCE}/{split}/{split}_{part}.json"
            logging.info(f"Loading chartqa data from {src}")
            with open(src) as f:
                data = json.load(f)
                for ex_id, ex in enumerate(data):
                    ex = dict(
                        image=join(CHARTQA_SOURCE, split, "png", ex.pop("imgname")),
                        question=ex["query"],
                        answers=ex["label"],
                        metadata=dict(
                            is_human=part == "human",
                            example_id=ex_id
                        )
                    )
                    examples.append(ex)
        return examples

    def get(self, item, rng):
        ex = dict(self.data[item], style="chart_qa")
        if self.weighted:
            is_human = ex["metadata"]["is_human"]
            # Weight to balanced human/augmented sets
            if is_human:
                w = 2*20901/(20901+7398)
            else:
                w = 2*7398/(20901+7398)
            ex["weight"] = w
        return ex
