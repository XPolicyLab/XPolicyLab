import json
from collections import defaultdict
from os import listdir
from os.path import join
from typing import List

import PIL
import datasets
import numpy as np
from PIL import Image, ImageFont
from PIL.ImageDraw import ImageDraw
from tqdm import tqdm

AI2D_ALL = "http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip"
AI2D_TEST_IDS = "https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv"


def flatten_lists(xss):
    return [x for xs in xss for x in xs]


def get_font_size(image_size: tuple[int, int]) -> int:
    ''' Get the font size based on the size of the image. '''
    min_font_size = 16
    max_font_size = 32
    # dynamically adjust font size based on size of image.
    longest_side = min(image_size)
    fontsize = min(max(min_font_size, int(longest_side / 50)), max_font_size)

    return fontsize


def draw_abc_labels(image_f, annotations, mode="in_box", use_transparent=False):
    image = PIL.Image.open(image_f)

    # fill = (0, 77, 255)
    fill = (25, 249, 255)

    if use_transparent:
        draw = ImageDraw(image, "RGBA")
        fill = fill + (127, )
    else:
        draw = ImageDraw(image, "RGB")

    # Load the font
    font_name = "Arial.ttf"

    for text_box in annotations["text"].values():
        (x1, y1), (x2, y2) = text_box["rectangle"]
        if y1 == y2:
            # Not clear what to do here, but this never happens for correct answers
            # so I assume its okay to just not add the box
            continue
        letter = text_box["replacementText"]
        draw.rectangle([x1, y1, x2, y2], fill=fill)

        if mode == "left": # draw text on the left of the box with white background
            # fontsize = 16
            fontsize = get_font_size(image.size)
            x_margin = 12
            y_margin = 10
            font = ImageFont.truetype(font_name, fontsize)

            # text top left corner
            xy = (max(x1-x_margin, 0), max(y1-y_margin, 0))

            # get text bbox and draw background
            text_bbox = draw.textbbox(xy, text=letter, font=font, )
            draw.rectangle(text_bbox, fill="white")

            # draw text
            draw.text(xy, text=letter, fill="black", font=font)

        elif mode == "in_box": # cover the box with text
            target_height = y2 - y1
            fontsize = 22
            font = ImageFont.truetype(font_name, fontsize)
            textbox = draw.textbbox((100, 100), letter, font=font)
            if target_height > 4:
                while textbox[3] - textbox[1] > target_height:
                    fontsize -= 1
                    font = ImageFont.truetype(font_name, fontsize)
                    textbox = draw.textbbox((100, 100), letter, font=font)
            cx = (textbox[2] - textbox[0]) / 2
            ry = target_height - (textbox[3] - textbox[1])
            draw.text(((x1+x2)/2 - cx, y1 - (textbox[1]-100) + ry/2), text=letter, fill="black", font=font)
        else:
            raise ValueError()
    return image


class Ai2dDatasetBuilder(datasets.GeneratorBasedBuilder):
    """
    AI2D dataset builder, this builder adds the labelled boxes as needed to the AI2D images
    using both transparent and opaque boxes
    """
    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dataset_name="ai2d")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'image': datasets.Image(),
                'question': datasets.Value("string"),
                'image_id': datasets.Value("int32"),
                'option_is_abc': datasets.Sequence(datasets.Value("bool")),
                'image_name': datasets.Value("string"),
                'question_id': datasets.Value("string"),
                'abc_label': datasets.Value("bool"),
                'answer_texts': datasets.Sequence(datasets.Value("string")),
                'correct_answer': datasets.Value("int32"),
                'has_transparent_box': datasets.Value("bool"),
            }),
        )

    def get_abc_image(self, image, annotations) -> np.ndarray:
        return np.array(draw_abc_labels(image, annotations, mode="in_box", use_transparent=False))

    def get_transparent_abc_image(self, image, annotations) -> np.ndarray:
        return np.array(draw_abc_labels(image, annotations, mode="left", use_transparent=True))

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        data_src = join(dl_manager.download_and_extract(AI2D_ALL), "ai2d")
        ai2d_test_ids = dl_manager.download(AI2D_TEST_IDS)

        question_dir = join(data_src, "questions")
        ans_dir = join(data_src, "annotations")
        with open(ai2d_test_ids) as f:
            test_ids = [int(x.strip()) for x in f.readlines()]
        assert len(test_ids) == len(set(test_ids))
        test_ids = set(test_ids)

        train_mix = defaultdict(list)
        train_transparent = defaultdict(list)
        test_mix = []
        for question_file in tqdm(listdir(question_dir), "building_ai2d", ncols=100):
            with open(join(question_dir, question_file)) as f:
                questions = json.load(f)
                image_id = int(questions["imageName"].split(".")[0])
                image = join(data_src, "images", questions["imageName"])
                abc_replacement_to_ans = {}
                if any(x["abcLabel"] for x in questions["questions"].values()):
                    with open(join(data_src, f"annotations/{questions['imageName']}.json")) as f:
                        annotations = json.load(f)
                    abc_image = self.get_abc_image(image, annotations)
                    transparent_abc_image = self.get_transparent_abc_image(image, annotations)
                    for text_box in annotations["text"].values():
                        assert text_box["replacementText"] not in abc_replacement_to_ans
                        assert text_box["replacementText"] == text_box["replacementText"].upper()
                        abc_replacement_to_ans[text_box["replacementText"]] = text_box["rectangle"]
                else:
                    abc_image = None
                    transparent_abc_image = None

                for question, question_data in questions["questions"].items():
                    q_image = image
                    q_transparent_image = image
                    options = question_data["answerTexts"]
                    question_id = question_data["questionId"]
                    answer = options[question_data["correctAnswer"]]
                    if question_data["abcLabel"]:
                        q_image = abc_image
                        q_transparent_image = transparent_abc_image
                        # Sanity check
                        if answer.upper() in abc_replacement_to_ans:
                            (x1, y1), (x2, y2) = abc_replacement_to_ans[answer.upper()]
                            assert y1 != y2

                    pruned_options = []
                    correct = question_data["correctAnswer"]
                    for ix, option in enumerate(options):
                        if option == "{}":
                            # Seems to indicates a null option
                            assert ix != correct
                            continue
                        if option in pruned_options:
                            # Some questions have duplicates answers, while the data only marks one
                            # as the correct value. I assume returning either is okay in practice
                            # so we will prune the second duplicate the answers
                            continue
                        pruned_options.append(option)

                    option_is_abc = []
                    for option in pruned_options:
                        option_is_abc.append(option.upper() in abc_replacement_to_ans)

                    example = dict(
                        question=question,
                        image=q_image,
                        image_id=image_id,
                        option_is_abc=option_is_abc,
                        image_name=questions["imageName"],
                        question_id=question_data["questionId"],
                        abc_label=question_data["abcLabel"],
                        answer_texts=pruned_options,
                        correct_answer=pruned_options.index(answer),
                        has_transparent_box=False,
                    )
                    transparent_example = dict(
                        question=question,
                        image=q_transparent_image,
                        image_id=image_id,
                        option_is_abc=option_is_abc,
                        image_name=questions["imageName"],
                        question_id=question_data["questionId"]+"_transparent",
                        abc_label=question_data["abcLabel"],
                        answer_texts=pruned_options,
                        correct_answer=pruned_options.index(answer),
                        has_transparent_box=question_data["abcLabel"],
                    )
                    if image_id in test_ids:
                        test_mix.append((question_data["questionId"], example))

                        # for mix, add question with transparent image
                        if question_data["abcLabel"]:
                            assert transparent_abc_image is not None
                            test_mix.append((transparent_example["question_id"], transparent_example))
                    else:
                        train_mix[image_id].append((question_data["questionId"], example))
                        train_transparent[image_id].append((transparent_example["question_id"], transparent_example))

                        # for mix, add question with transparent image
                        if question_data["abcLabel"]:
                            assert transparent_abc_image is not None
                            train_mix[image_id].append((transparent_example["question_id"], transparent_example))

        keys = sorted(train_mix)
        np.random.RandomState(5961).shuffle(keys)
        n_val = 384
        print(f"Holding out {n_val}/{len(keys)} images and {sum(len(train_mix[k]) for k in keys[:n_val])} questions for val")

        validation_mix = flatten_lists(train_mix[k] for k in keys[:n_val])
        splits = dict(
            # train=flatten_lists(train.values()),
            train=flatten_lists(train_mix[k] for k in keys[n_val:]), # placed here to avoid overwrite
            validation=validation_mix,
            test=test_mix,
        )

        print("Dataset sizes:")
        for key, data in splits.items():
            print(f"{key}: {len(data)}")

        return [
            datasets.SplitGenerator(name=name, gen_kwargs=dict(examples=examples))
            for name, examples in splits.items()
        ]

    def _generate_examples(self, examples):
        for ex in examples:
            yield ex


if __name__ == "__main__":
    Ai2dDatasetBuilder().download_and_prepare()

