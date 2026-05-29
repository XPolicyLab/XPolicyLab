import ast
import csv
import unicodedata
from os.path import join, exists
from typing import List

import datasets
import numpy as np
from PIL import Image


def crop_with_context(img_shape, bbox, margin=0.2):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
    w, h = img_shape
    crop_h = y2 - y1
    crop_w = x2 - x1

    x1_a = int(max(0, x1 - margin * crop_w))
    x2_a = int(min(w, x2 + margin * crop_h))
    y1_a = int(max(0, y1 - margin * crop_w))
    y2_a = int(min(h, y2 + margin * crop_h))

    crop_hh = y2_a - y1_a
    crop_ww = x2_a - x1_a

    xx1 = max(0, (x1_a + x2_a) // 2 - max(crop_hh, crop_ww) // 2)
    xx2 = min(w, (x1_a + x2_a) // 2 + max(crop_hh, crop_ww) // 2)
    yy1 = max(0, (y1_a + y2_a) // 2 - max(crop_hh, crop_ww) // 2)
    yy2 = min(h, (y1_a + y2_a) // 2 + max(crop_hh, crop_ww) // 2)

    return xx1, yy1, xx2, yy2


class ClockBenchBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="clock_bench")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                'cropped_image': datasets.Image(),
                "hour": datasets.Value("int64"),
                "minute": datasets.Value("int64"),
                "image_id": datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        splits = ["coco", "openimg", "movies"]
        image_ids = dict(
            coco="1j1RkuzYTPppR8_VGMx3PQLFut6toRHaq",
            openimg="1GT7TCwOfvdH0f-Mgqcu9JTliPfvqNl0s",
            movies="1zkWBY9rL0Bzyc5ird8Z9mZFJcDte0FN-",
        )
        image_urls = {
            k: f"https://drive.usercontent.google.com/download?id={v}&confirm=t"
            for k, v in image_ids.items()
        }
        downloaded_images = dl_manager.download_and_extract(image_urls)
        annotation_urls = {
            k: f"https://raw.githubusercontent.com/charigyang/itsabouttime/refs/heads/main/data/{k}_final.csv"
            for k in splits
        }
        downloaded_annotations = dl_manager.download(annotation_urls)
        return [
            datasets.SplitGenerator(
                name=k,
                gen_kwargs=dict(
                    image_dir=downloaded_images[k],
                    anno_file=downloaded_annotations[k],
                    split=k
                )
            )
            for k in splits
        ]

    def _generate_examples(self, image_dir, anno_file, split):
        image_sub_dir = {
            "coco": "trainval_images",
            "openimg": "data",
            "movies": "ClockMovies/images",
        }
        img_dir = join(image_dir, image_sub_dir[split])

        with open(anno_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        for row in rows:
            img_path = join(img_dir, unicodedata.normalize('NFD', row["file_name"]))
            if not exists(img_path):
                continue
            image = Image.open(img_path).convert("RGB")
            bbox_det = ast.literal_eval(row['bbox_det'])
            x1, y1, x2, y2 = crop_with_context(image.size, bbox_det)
            cropped_image = np.asarray(image)[y1:y2, x1:x2, :]
            image = np.asarray(image)
            assert image.dtype == np.uint8
            hour = int(row['hour'])
            minute = int(row['minute'])
            image_id = row['img_id']
            yield img_path, dict(
                image=image,
                cropped_image=cropped_image,
                hour=hour,
                minute=minute,
                image_id=image_id,
            )


if __name__ == "__main__":
    ClockBenchBuilder().download_and_prepare()