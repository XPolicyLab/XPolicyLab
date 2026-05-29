"""Classes the compute metrics given ground truth/prediction pairs"""
import base64
import dataclasses
import io
import json
import logging
import os
import re
import ast
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from html import escape as html_escape
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchmetrics import MeanMetric
from a1.tokenizer import get_special_token_ids,DEFAULT_ACT_START_TOKEN, DEFAULT_ACT_END_TOKEN
from .vqa import vqa_score, robovqa_score, anls_metric, relaxed_correctness, \
    a_okvqa_score, select_mc_option, mmmu_score, real_world_qa_score, math_vista_score, \
    clevrmath_score, superclevr_score, trance_score
from ..html_utils import build_html_table, postprocess_prompt, BoxesToVisualize, \
    get_html_image_with_boxes
from ..torch_util import (
    get_global_rank,
    get_world_size,
)
from ..util import flatten_list, extract_points, extract_bboxes, extract_points_from_point_count

log = logging.getLogger(__name__)


def get_openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return key


def mean_metric(v):
    metric = MeanMetric(nan_strategy="error")
    metric.update(np.mean(v) if len(v)>0 else 0, len(v))
    return metric


@dataclasses.dataclass
class HtmlTable:
    """Returned as special metric for visualizing predictions"""
    rows: List[Dict[str, Any]]

    def get_html(self):
        return build_html_table(self.rows)


def annotation_to_box(points, point_dist=4):
    to_show = []
    for point in points:
        if len(point) == 2:
            x, y = point
            to_show.append([x-point_dist, y-point_dist, x+point_dist, y+point_dist])
        else:
            to_show.append(point)
    return to_show


def gather_examples_as_html(
    n_examples, voc, metadatas, predictions,
    scores=None, fix_width=True, pred_points=None, gt_points=None, pred_actions=None, gt_actions=None
) -> HtmlTable:
    """Builds a HTML table visualization of the predictions"""

    n = len(predictions["predictions"])
    if n_examples is not None:
        # Divide by world size since we will aggregate visualization across all processes
        n = min(n, n_examples)
        n = (n + get_world_size() - 1) // get_world_size()
    rows = []
    new_tokens = predictions["predictions"]
    prompt_tokens = predictions["prompts"]
    for ix in range(n):
        prompt_text = postprocess_prompt(voc.decode(prompt_tokens[ix][prompt_tokens[ix] >= 0]))
        metadata = metadatas[ix]
        pred_seq = new_tokens[ix]
        pred_txt = voc.decode(pred_seq[pred_seq >= 0])

        row = dict()
        image_src = None
        if "image_url" in metadata:
            image_src = metadata['image_url']
        elif "image" in metadata:
            with Image.open(metadata["image"]) as img:
                image_data = io.BytesIO()
                img.save(image_data, format='JPEG')
                image_data = image_data.getvalue()
            image_src = f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}'
        if image_src is not None:
            ex_pred_points, gt_pred_points = None, None
            if pred_points is not None:
                ex_pred_points = pred_points[ix]
            if gt_points is not None:
                gt_pred_points = gt_points[ix]
            if ex_pred_points is None and gt_pred_points is None:
                row["image"] = f"<img style=\"max-height:500px;max-width:500px;height:auto;width:auto;\" src={image_src}><img>"
            else:
                to_show = []
                if ex_pred_points is not None:
                    to_show.append(BoxesToVisualize(annotation_to_box(ex_pred_points), "blue", format="xyxy"))
                if gt_pred_points is not None:
                    to_show.append(BoxesToVisualize(annotation_to_box(gt_pred_points, 3), "green", format="xyxy"))
                row["image"] = get_html_image_with_boxes(image_src, to_show)
        row["prompt"] = html_escape(prompt_text)
        if pred_actions is not None:
            pred = str(pred_actions[ix])
            row["prediction"] = html_escape(pred)
        else:
            row["prediction"] = html_escape(pred_txt)

        if gt_actions is not None:
            gt = str(gt_actions[ix])
        elif "answers" in metadata:
            gt = metadata["answers"]
        elif "answer" in metadata:
            gt = metadata["answer"]
        elif "caption" in metadata:
            gt = metadata["caption"]
        else:
            gt = None
        if gt is not None:
            gt = "<br>".join(html_escape(x) for x in gt)
            row["gt"] = gt
        if scores is not None:
            if isinstance(scores[ix], dict):
                for k, v in scores[ix].items():
                    if isinstance(v, str):
                        row[k] = v
                    else:
                        row[k] = "" if v is None else f"{v:0.3f}"
            else:
                row["score"] = f"{scores[ix]:0.3f}"
        rows.append(row)
    return HtmlTable(rows)


def get_gcs_url(output_file):
    assert output_file.startswith("gs://")
    return f"https://storage.cloud.google.com/{output_file[5:]}?authuser=1"


class Evaluator:
    def __call__(self, metadatas, predictions, tokenizer, step=None):
        raise NotImplementedError()


class SavePredictions(Evaluator):

    @staticmethod
    def get_file_name(step, process_index):
        filename = ""
        if step is not None:
            filename += f"step{step}-"
        if get_world_size() > 1 and process_index is not None:
            filename += f"shard{process_index}"
        filename += "predictions"
        return filename

    def __init__(self, output_dir, json=True, save_tokens=True,
                 log_examples=10):
        self.save_tokens = save_tokens
        self.output_dir = output_dir
        self.log_examples = log_examples
        self.json = json

    def __call__(self, metadatas, predictions, tokenizer,
                 step=None, scores=None):
        if not self.output_dir.startswith("gs://"):
            if not os.path.exists(self.output_dir):
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        json_data = []
        html_data = []

        n_no_eos = 0
        for tok in new_tokens:
            if not np.any(tok == tokenizer.eos_token_id):
                n_no_eos += 1
        if n_no_eos > 0:
            logging.warning(f"{n_no_eos}/{len(new_tokens)} ({n_no_eos/len(new_tokens):00.4f}) "
                            f"examples have no EOS, your inference tokens might be too short")

        for ex_ix, pred_seq in enumerate(new_tokens):
            text = tokenizer.decode(pred_seq[pred_seq >= 0])
            json_row = dict(prediction=text)
            if self.save_tokens:
                json_row["n_tokens"] = pred_seq.tolist()
            prompt_text = postprocess_prompt(tokenizer.decode(prompt_tokens[ex_ix][prompt_tokens[ex_ix] >= 0]))
            if tokenizer.adds_space:
                sep = " "
            else:
                sep = ""
            json_row["prompt"] = prompt_text
            metadata = metadatas[ex_ix]
            if ex_ix < self.log_examples:
                log.info("*"*30)
                if "example_id" in metadata:
                    log.info(metadata['example_id'])
                log.info(' '.join((prompt_text + sep + text).split()))
                # log.warning(f"no print prompt_text")
            json_row.update({k: v for k, v in metadata.items() if isinstance(v, (str, float, int))})
            json_data.append(json_row)

        json_file = None
        html_file = None
        metrics = {}

        if self.json:
            log.info("Save prediction JSON")
            if get_world_size() > 1 and self.json:
                if get_global_rank() == 0:
                    all_predictions = [None]*get_world_size()
                    dist.gather_object(json_data, all_predictions)
                    json_data = flatten_list(all_predictions)
                else:
                    dist.gather_object(json_data, None)

            if get_global_rank() == 0:
                json_file = os.path.join(self.output_dir, self.get_file_name(step, None) + ".json")
                with open(json_file, "w") as f:
                    json.dump(json_data, f)
                log.info("done saving json")
        return metrics


def is_point_in_region(point: Tuple[float, float], mask: np.ndarray) -> bool:
    
    
    # Check if the point (x, y) is within the region defined by the boolean mask.

    # Parameters:
    # - point (tuple of floats): x/y-coordinate of the point
    # - mask (2D numpy array): Boolean mask of shape [H, W] representing the region

    # Returns:
    # - bool: True if the point is within the region, False otherwise
    
    height, width = mask.shape
    x, y = point

    # Round the coordinates to the nearest integer
    x_int = int(round(x))
    y_int = int(round(y))

    # Check if the rounded point is within the bounds of the image
    if x_int < 0 or x_int >= width or y_int < 0 or y_int >= height:
        return False

    # Check if the point is within the region
    return mask[y_int, x_int]


def is_valid_format(input_string):
    # Define the regular expression pattern
    pattern = re.compile(
        r'^(\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)\n?)+$|'
        r'^<point\s+x="\s*\d+(\.\d+)?"\s+y="\s*\d+(\.\d+)?"\s+alt="[\s\S]*?">[\s\S]*?</point>$|'
        r'^<points\s+(x\d+="\s*\d+(\.\d+)?"\s+y\d+="\s*\d+(\.\d+)?"\s+)+alt="[\s\S]*?">[\s\S]*?</points>$|'
        r'^<point\s+p=\s*\d{3}\s*,\s*\d{3}\s+alt="[\s\S]*?">[\s\S]*?</point>$|'
        r'^<points\s+(\d+=\s*\d{3}\s*,\s*\d{3}\s+)+alt="[\s\S]*?">[\s\S]*?</points>$'
    )

    # Match the entire input string against the pattern
    match = pattern.fullmatch(input_string.strip())

    # Return True if the match is successful, False otherwise
    return match is not None


def compute_precision(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(preds)


def compute_recall(row_ind: np.ndarray, col_ind: np.ndarray, preds: np.ndarray, masks: List[np.ndarray]):
    cnt = 0
    for i, j in zip(row_ind, col_ind):
        if is_point_in_region(preds[i], masks[j]):
            cnt += 1
    return cnt / len(masks)


def f1_score(precision: float, recall: float, epsilon: float = 1e-10):
    if precision == 0 or recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall + epsilon)


# Metric function aiming to replicate the step-wise accuracy as described in Appendix D.3 of
# the AndroidControl paper. Very specific to the action space and edge cases in that dataset.
def compute_stepwise_accuracy(ground_truth, predictions, target_bbs):
    def parse_action(action_str):
        action_str = action_str.strip().lower()

        def get_coords(action):
            coords = re.findall(r'\d+(?:\.\d+)?', action)
            return (float(coords[0]), float(coords[1])) if len(coords) >= 2 else ''

        if action_str.startswith("click"):
            return {"type": "click", "coords": get_coords(action_str)}
        elif action_str.startswith("long press"):
            return {"type": "long press", "coords": get_coords(action_str)}
        elif action_str.startswith("type"):
            return {"type": "type", "text": action_str[5:]}
        elif action_str.startswith("scroll"):
            return {"type": "scroll", "direction": action_str.split()[1] if len(action_str.split()) >= 2 else ''}
        elif action_str == "wait":
            return {"type": "wait"}
        elif action_str.startswith("open app"):
            return {"type": "open_app", "app_name": action_str[9:]}
        elif action_str == "navigate home":
            return {"type": "navigate home"}
        elif action_str == "navigate back":
            return {"type": "navigate back"}
        return {"type": None}

    def within_bounding_box(coords, box):
        x, y = coords[0], coords[1]
        bbox_values = re.findall(r'\d+\.\d+', box)
        x1, y1, x2, y2 = [float(val) for val in bbox_values]
        return x1 <= x <= x2 and y1 <= y <= y2

    all_predictions = []
    metrics = []
    for gt_action, pred_action, gt_box in zip(ground_truth, predictions, target_bbs):
        metric = 'incorrect'  # default to a prediction being incorrect until proven otherwise

        gt_parsed = parse_action(gt_action)
        pred_parsed = parse_action(pred_action)
        correct_predictions = 0
        if gt_parsed["type"] == pred_parsed["type"]:
            if gt_parsed["type"] in ["click", "long press"]:
                gt_coords = gt_parsed["coords"]
                if "coords" in pred_parsed and pred_parsed['coords'] != '' and gt_box != None and gt_parsed['coords'] != '':
                    pred_coords = pred_parsed["coords"]
                    if within_bounding_box(pred_coords, gt_box):
                        correct_predictions += 1
                        metric = 'correct'
            elif gt_parsed["type"] == "type" and gt_parsed["text"] == pred_parsed["text"]:
                correct_predictions += 1
                metric = 'correct'
            elif gt_parsed["type"] == "scroll" and gt_parsed["direction"] == pred_parsed["direction"]:
                correct_predictions += 1
                metric = 'correct'
            elif gt_parsed["type"] in ["navigate home", "navigate back", "wait"]:
                correct_predictions += 1  # These actions have no parameters to compare
                metric = 'correct'
            elif gt_parsed["type"] == "open_app" and pred_parsed["app_name"] == gt_parsed["app_name"]:
                correct_predictions += 1
                metric = 'correct'
            else:
                if gt_parsed == pred_parsed:
                    correct_predictions += 1
                    metric = 'correct'
        else:
            # Consider open_app and click on app name equivalent
            if pred_parsed["type"] == "click" and gt_parsed["type"] == "open_app":
                if gt_box not in [None, ''] and "coords" in pred_parsed and pred_parsed['coords'] != '':
                    if within_bounding_box(pred_parsed['coords'], gt_box):
                        correct_predictions += 1
                        metric = 'correct'
        all_predictions.append(correct_predictions)
        metrics.append(metric)
    # max with 1 since its technically possible for a node to get 0 valid examples
    return all_predictions, metrics


class AndroidControlEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]

        targets = []
        preds = []
        target_bbs = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_text = tokenizer.decode(pred_seq[pred_seq >= 0])
            if "Action:" in pred_text:
                pred_text = pred_text.split("Action:")[1].strip()

            # Get metadata needed for metrics and vis
            preds.append(pred_text)
            targets.append(metadata["target_action"])
            target_bbs.append(metadata["target_box"])

        accuracy, corrects = compute_stepwise_accuracy(targets, preds, target_bbs)
        """
        if self.n_to_log:
            for ex in range(self.n_to_log):
                wandb.log({
                    #f"{ex}/image": wandb.Image(images[ex], caption=instructions[ex]),
                    f"{ex}/instruction": instructions[ex],
                    f"{ex}/target_answer": targets[ex],
                    f"{ex}/model_answer": preds[ex],
                    f"{ex}/metric": corrects[ex]
                })
        """

        out = {'accuracy': mean_metric(accuracy)}
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, accuracy
            )
        return out



class MultiThresholdBoxEval(Evaluator):
    def __init__(self, n_to_log=None, iou_thresholds=None):
        """
        """
        self.n_to_log = n_to_log
        if iou_thresholds is None:
            self.iou_thresholds = [0.5 + 0.05 * i for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        ious = []
        scores = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_text = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            gt_box = self.extract_box(str(metadata["answers"]))
            pred_box = self.extract_box(pred_text)
            # log.info(f'gt_text: {str(metadata["answers"])}, pred_text: {pred_text}')
            # log.info(f'gt_box: {gt_box}, pred_box: {pred_box}')
            # assert False
            if gt_box is None or pred_box is None:
                iou = 0.0
            else:
                iou = self.compute_iou(pred_box, gt_box)

            ious.append(iou)
            scores.append(1 if iou >= 0.5 else 0)

        accuracy50 = mean_metric(scores)

        N = len(ious)
        per_thresh = []
        for t in self.iou_thresholds:
            hit = sum(1 for iou in ious if iou >= t)
            per_thresh.append(hit / N)
        mean_ap = mean_metric(per_thresh)
  
        out = {
            "accuracy@0.50": accuracy50,
            "mAP@[0.50:0.95]": mean_ap,
        }

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, scores
            )
        return out

    def extract_box(self, text):
        """
        Tries to extract (x, y, width, height) from a structured string like:
        - "{'type': 'Affordance', 'x': 11.1, 'y': 12.5, 'width': 45.2, 'height': 66.7}"
        - "(11.1, 12.5, 45.2, 66.7)"
        - "x=11.1, y=12.5, width=45.2, height=66.7"
        - "11.1, 12.5, 45.2, 66.7"
        Returns None if it cannot find 4 numbers.
        """
        x_match = re.search(r"x=([-\d.]+)", text)
        y_match = re.search(r"y=([-\d.]+)", text)
        w_match = re.search(r"width=([-\d.]+)", text)
        h_match = re.search(r"height=([-\d.]+)", text)

        if all([x_match, y_match, w_match, h_match]):
            x = float(x_match.group(1))
            y = float(y_match.group(1))
            w = float(w_match.group(1))
            h = float(h_match.group(1))
            return (x, y, w, h)

        # Fallback: find 4 floats anywhere in the text
        matches = re.findall(r"-?\d+\.?\d*", text)
        if len(matches) >= 4:
            x, y, w, h = map(float, matches[:4])
            return (x, y, w, h)

        # Fail case
        return None

    def compute_iou(self, box1, box2):
        """
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xa1, ya1 = x1, y1
        xa2, ya2 = x1 + w1, y1 + h1
        xb1, yb1 = x2, y2
        xb2, yb2 = x2 + w2, y2 + h2

        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)

        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        inter_area = inter_w * inter_h

        area_a = w1 * h1
        area_b = w2 * h2
        union = area_a + area_b - inter_area

        return inter_area / union if union > 0 else 0.0


class CoordinateAccuracyEval(Evaluator):
    def __init__(self, n_to_log=None, threshold=2.0):  # Default : 2
        self.n_to_log = n_to_log
        self.threshold = threshold

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]

        scores = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_text = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            gt_point = self.extract_point(str(metadata["answers"]))

            pred_point = self.extract_point(pred_text)
            if pred_point is None:
                scores.append(0)
                continue

            if self.is_within_threshold(pred_point, gt_point, self.threshold):
                scores.append(1)
            else:
                scores.append(0)

        out = {
            "coordinate_accuracy": mean_metric(scores),
        }

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, scores
            )
        return out

    def extract_point(self, text):
        match = re.search(r"\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?", text)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            return (x, y)
        return None

    def is_within_threshold(self, pred, gt, threshold):
        distance = sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
        return distance <= threshold


class PointingEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer
        scores = defaultdict(list)
        pred_points = []
        gt_points = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            answer_points = metadata["points"]
            masks = metadata["masks"]
            image_w, image_h = metadata["image_size"]
            abs_preds = extract_points(pred, image_w, image_h)

            if len(answer_points) == 0:
                precision = recall = f1 = float(abs_preds is None or len(abs_preds) == 0)
                abs_gts = None
            else:
                abs_gts = answer_points
                if not is_valid_format(pred):
                    precision = recall = f1 = 0.0
                else:
                    abs_preds = np.array(abs_preds)
                    dists = cdist(abs_preds, abs_gts)
                    row_ind, col_ind = linear_sum_assignment(dists)
                    precision = compute_precision(row_ind, col_ind, abs_preds, masks)
                    recall = compute_recall(row_ind, col_ind, abs_preds, masks)
                    f1 = f1_score(precision, recall)
            scores["precision"].append(precision)
            scores["recall"].append(recall)
            scores["f1"].append(f1)

            pred_points.append(abs_preds)
            gt_points.append(abs_gts)

        out = {}

        if "was_lowered" in metadatas[0]:
            # Get a score with and without lowering
            lowered = np.array([(x["was_lowered"] or x["was_lowered"] is None) for x in metadatas])
            nocase = np.array([(not x["was_lowered"] or x["was_lowered"] is None) for x in metadatas])
            for k, v in scores.items():
                v = np.array(v)
                out[k] = mean_metric(v[nocase])
                out[f"{k}_lower"] = mean_metric(v[lowered])
        else:
            for k, v in scores.items():
                out[k] = mean_metric(v)

        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] for k in scores} for i in range(len(new_tokens))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores,
                pred_points=pred_points, gt_points=gt_points
            )
        return out


class PointCountEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer
        all_scores = defaultdict(list)
        per_category_scores = defaultdict(list)
        gt_counts_per_device = []
        pred_points = []
        gt_points = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            original_pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            pred = original_pred.lower().rstrip(".").strip()
            gt = metadata["count"]

            pred_int = None
            parts = pred.split()
            if parts:
                try:
                    pred_int = int(parts[-1])
                except ValueError:
                    pass

                if pred_int is None:
                    if parts[-1] in WORD_TO_NUM:
                        pred_int = WORD_TO_NUM[parts[-1]]

            # Parse out the int for point and count data
            if pred_int is None:
                match = re.match(".*a total of ([0-9]+).*", pred)
                if match:
                    pred_int = int(match.group(1))

            if pred_int is None:
                match = re.match(".*\bnone\b.*", pred)
                if match:
                    pred_int = 0

            if pred_int is None:
                correct, close, valid = 0, 0, False
            else:
                correct = gt == pred_int
                close = abs(gt - pred_int) <= 1
                valid = True
            all_scores["close"].append(close)
            all_scores["valid"].append(valid)
            all_scores["correct"].append(correct)

            per_category_scores[int(gt)].append(correct)
            gt_counts_per_device.append(int(gt))

            abs_preds = None
            abs_gts = None
            if "image_size" in metadata:
                image_w, image_h = metadata["image_size"]
                try:
                    if len(re.findall(r"(\d+\.\d+),\s*(\d+\.\d+)", original_pred)) > 0:
                        abs_preds = np.array(extract_points_from_point_count(original_pred, image_w, image_h))
                except Exception as e:
                    print("Failed extracting pred points with error - ", e)
                    abs_preds = None
                try:
                    if "points" in metadata:
                        abs_gts = metadata["metadata/points"]
                except Exception as e:
                    print("Failed extracting gt points with error - ", e)
                    abs_gts = None

            pred_points.append(abs_preds)
            gt_points.append(abs_gts)

        num_examples_per_device = torch.tensor(len(new_tokens), dtype=torch.int32, device=torch.device("cuda"))
        num_examples = torch.zeros(get_world_size(), dtype=torch.int32, device=torch.device("cuda"))
        dist.all_gather_into_tensor(num_examples, num_examples_per_device)
        max_num_examples = num_examples.detach().cpu().max().item()
        gt_counts_per_device = torch.tensor(gt_counts_per_device, dtype=torch.int32, device=torch.device("cuda"))
        gt_counts_per_device = torch.cat(
            [gt_counts_per_device, torch.full((max_num_examples - len(new_tokens),), -1, dtype=torch.int32, device=torch.device("cuda"))],
            dim=0,
        )
        gt_counts = torch.zeros(get_world_size() * max_num_examples, dtype=torch.int32, device=torch.device("cuda"))
        dist.all_gather_into_tensor(gt_counts, gt_counts_per_device)
        gt_counts = gt_counts.detach().cpu().numpy()
        gt_counts = np.sort(np.unique(gt_counts[gt_counts >= 0]))

        out = {}
        for k, v in all_scores.items():
            out[k] = mean_metric(v)

        for k in gt_counts:
            out[f"correct_{k}"] = mean_metric(per_category_scores[k])

        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, all_scores["correct"],
                pred_points=pred_points, gt_points=gt_points
            )
        return out

class TrajectoryEval(Evaluator):

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        prompt_text = tokenizer.decode(prompt_tokens[0][prompt_tokens[0] >= 0])
        # log.info(f'preprocess prompt: {prompt_text}')
        prompt_text = postprocess_prompt(prompt_text)
        log.info(f'prompt_text:{prompt_text}')
        vocab = tokenizer
        scores = defaultdict(list)
        pred_points = []
        gt_points = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            answer_points = metadata["answers"][0]
            image_w, image_h = 100, 100
            abs_preds = extract_points(pred, image_w, image_h)
            answer_points = extract_points(answer_points, image_w, image_h)
            # abs_preds = abs_preds[:4]
            # answer_points = answer_points[:4]
            if len(abs_preds)!=len(answer_points):
                if len(abs_preds)>len(answer_points):
                    abs_preds = abs_preds[:len(answer_points)]
                else:
                    answer_points = answer_points[:len(abs_preds)]
            mae = np.abs(np.array(abs_preds) - np.array(answer_points)).mean()
            scores["mae"].append(mae)

            pred_points.append(abs_preds)
            gt_points.append(answer_points)

        out = {}

        for k, v in scores.items():
            out[k] = mean_metric(v)

        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] for k in scores} for i in range(len(new_tokens))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores,
                pred_points=pred_points, gt_points=gt_points
            )
        return out

class ActionEval(Evaluator):
    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log
        self.fast_skip_tokens = 2048

    def _map_action_tokens(self, tokens, tokenizer,action_tokenizer):
        """
        Construct the action token sequence of ACTION_DIMS*NUM_ACTIONS_CHUNK, 
        predict the next NUM_ACTIONS_CHUNK steps of actions, and insert them at the end of the sequence.
        """      
        tokens = tokenizer.vocab_size() - 1 - self.fast_skip_tokens - tokens
        return tokens

    def extract_action(self, pred, tokenizer, action_tokenizer):
        prefix = tokenizer.encode(DEFAULT_ACT_START_TOKEN)[0]
        suffix = tokenizer.encode(DEFAULT_ACT_END_TOKEN)[0]
        pred = pred.tolist()
        if get_global_rank() == 0:
            print('prefix',prefix)
            print('suffix',suffix)
            print('pred',pred)
        try:
            pos = pred.index(prefix)+1
            bar_pos = pred.index(suffix)
        except Exception as e:
            print('warning: no action start or end token found')
            return np.zeros((action_tokenizer.horizon, action_tokenizer.action_dim), dtype=np.float32)
        pred = pred[pos:bar_pos]
        pred = np.array(pred)
        if get_global_rank() == 0:
            print('processed pred',pred)
        action_tokens = self._map_action_tokens(pred, tokenizer, action_tokenizer)
        if get_global_rank() == 0:
            print('processed action_tokens',action_tokens)
        pred_action = action_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_tokenizer.horizon, action_dim=action_tokenizer.action_dim
        )[0]
        return pred_action

    def __call__(self, metadatas, predictions, tokenizer, step=None, action_tokenizer=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        prompt_text = tokenizer.decode(prompt_tokens[0][prompt_tokens[0] >= 0])
        # log.info(f'preprocess prompt: {prompt_text}')
        prompt_text = postprocess_prompt(prompt_text)
        log.info(f'prompt_text:{prompt_text}')
        vocab = tokenizer
        scores = defaultdict(list)
        pred_actions = []
        gt_actions = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            pred_action = self.extract_action(pred_seq, tokenizer, action_tokenizer)
            gt_action = metadata["action"]
            mae = np.abs(np.array(pred_action) - np.array(gt_action)).mean()
            if get_global_rank() == 0:
                print('pred_action',pred_action)
                print('gt_action',gt_action)
                print('mae',mae)
            scores["mae"].append(mae)

            pred_actions.append(pred_action)
            gt_actions.append(gt_action)

        out = {}

        for k, v in scores.items():
            out[k] = mean_metric(v)

        if self.n_to_log:
            per_example_scores = [{k: scores[k][i] for k in scores} for i in range(len(new_tokens))]
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, per_example_scores,
                pred_actions=pred_actions, gt_actions=gt_actions
            )
        return out

def _math_vista_score(args):
    return math_vista_score(*args)


class MathVistaEval(Evaluator):
    def __init__(self, n_to_log=None, n_threads=4):
        self.n_to_log = n_to_log
        self.n_threads = n_threads

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer

        _args = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            _args.append((pred, metadatas[ex_ix], get_openai_key()))

        scores = []
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            for score in pool.map(_math_vista_score, _args):
                scores.append(score)

        out = dict(score=mean_metric(scores))
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, scores)
        return out


class VqaEval(Evaluator):

    def __init__(self, score_fn=("vqa_score",), n_to_log=None):
        self.metric = score_fn
        assert len(set(self.metric)) == len(self.metric)
        self.n_to_log = n_to_log

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        vocab = tokenizer
        score_lists = defaultdict(list)

        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadatas[ex_ix]
            if "answer" in metadata:
                answers = metadata["answer"]
            elif "answers" in metadata:
                answers = metadata["answers"]
            else:
                answers = None
            if isinstance(answers, str):
                answers = [answers]

            pred = vocab.decode(pred_seq[pred_seq >= 0]).strip()
            pred_long = pred
            if "Answer:" in pred:
                pred = pred.split("Answer:")[1].strip()
            elif "\n" in pred:
                preds = [" ".join(x.strip().split()) for x in pred.split("\n")]
                counts = Counter(preds)
                max_count = max(counts.values())
                pred = [x for x in preds if counts[x] == max_count][0]
            else:
                pred = " ".join(pred.strip().split())

            for metric in self.metric:
                if metric == "vqa_score":
                    score = vqa_score(answers, pred)
                elif metric == "robovqa_score":
                    score = robovqa_score(answers, pred)
                elif metric == "superclevr_score":
                    score = superclevr_score(answers, pred_long)
                elif metric == "clevrmath_score":
                    score = clevrmath_score(answers, pred_long)
                elif metric in ["trance_test_id_score",
                                "trance_test_ood_left_score",
                                "trance_test_ood_right_score"]:
                    score = trance_score(answers, pred_long)
                elif metric == "ansl":
                    score = max(anls_metric(ref, pred) for ref in answers)
                elif metric == "relaxed_correctness":
                    score = max(relaxed_correctness(ans, pred) for ans in answers)
                elif metric == "a_okvqa_score":
                    score = a_okvqa_score(answers, pred)
                elif metric == "em":
                    score = pred.lower() in [x.lower() for x in answers]
                elif metric == "mc":
                    options = metadata["option_names"]
                    get_answer_idx = select_mc_option(pred, options)
                    score = get_answer_idx == metadata["answer_idx"]
                elif metric in ["mc_ai2d_transparent", "mc_ai2d_opaque"]: # mc split by transparency
                    has_transparent_box = metadata["has_transparent_box"]
                    abc_label = metadata["abc_label"]
                    # for abc_label, either evaluate on opaque or transparent boxes
                    if abc_label:
                        if metric == "mc_ai2d_transparent" and not has_transparent_box:
                            continue
                        elif metric == "mc_ai2d_opaque" and has_transparent_box:
                            continue
                    options = metadata["option_names"]
                    get_answer_idx = select_mc_option(pred, options)
                    score = get_answer_idx == metadata["answer_idx"]
                elif metric == "mmmu_score":
                    score = mmmu_score(answers, pred, metadata)
                elif metric == "real_world_qa_score":
                    score = real_world_qa_score(metadata["answer"], pred, metadata)
                elif metric == "seed_bench_score":
                    options = list("abcd".upper())
                    get_answer_idx = select_mc_option(pred, options)
                    score = get_answer_idx == metadata["metadata/answer_idx"][ex_ix]
                    data_type = metadata["metadata/data_type"][ex_ix].decode("utf-8")
                    score_lists[f"seed_bench_{data_type}_score"].append(score)
                elif metric == "math_vista_score":
                    score = math_vista_score(metadata["answer"], pred, metadata, get_openai_key())
                else:
                    raise NotImplementedError(metric)
                score_lists[metric].append(score)


        if "is_human" in metadatas[0]:
            is_human = np.array([x["is_human"] for x in metadatas])
            for k, v in list(score_lists.items()):
                score_lists[f"{k}_human"] = np.array(v)[is_human]
                score_lists[f"{k}_aug"] = np.array(v)[np.logical_not(is_human)]

        out = {}
        for k, v in score_lists.items():
            out[k] = mean_metric(v)

        if self.n_to_log:
            score_to_log = score_lists[self.metric[0]]
            if len(score_to_log) != len(metadatas):
                score_to_log = None
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, vocab, metadatas, predictions, score_to_log)
        return out


WORD_TO_NUM = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'zero': 0,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20
}


class CountEval:

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, metadata, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        all_scores = defaultdict(list)
        points = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            metadata = metadata[ex_ix]
            pred = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            pred = pred.split()[0].rstrip(".,")
            gt = metadata["count"]
            if "image_size" in metadata:
                w, h = metadata["image_size"]
                points.append(extract_points(pred, w, h))

            pred_int = None
            try:
                pred_int = int(pred)
            except ValueError:
                pass
            pred_int = 1

            if pred_int is None:
                pred = pred.lower()
                if pred in WORD_TO_NUM:
                    pred_int = WORD_TO_NUM[pred]

            # Parse out the int for point and count data
            if pred_int is None:
                match = re.match(".*a total of ([0-9]+).*", pred)
                if match:
                    pred_int = int(match.group(1))

            if pred_int is None:
                match = re.match(".*\bnone\b.*", pred)
                if match:
                    pred_int = 0

            if pred_int is None:
                correct, close, valid = 0, 0, False
            else:
                correct = gt == pred_int
                close = abs(gt - pred_int) <= 1
                valid = True
            all_scores["close"].append(close)
            all_scores["valid"].append(valid)
            all_scores["correct"].append(correct)

        out = {}
        for k, vals in all_scores.items():
            out[k] = mean_metric(vals)
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadata, predictions, all_scores["correct"],
                all_scores["valid"], pred_points=points
            )
        return out


class ClockEval:
    METRICS = [
        "overall_close",
        "overall_exact",
        "correctly_declines_to_answer",
        "all_correct",
        "all_close",
        "hour_correct",
        "minute_correct",
        "second_correct",
        "minute_close",
        "second_close",
    ]

    def __init__(self, n_to_log=None, is_test=False):
        self.n_to_log = n_to_log
        self.is_test = is_test

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        err_threshold = 1 if self.is_test else 3
        all_scores = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            metadata = metadatas[ex_ix]
            scores = {}
            hour = metadata["hour"]
            if self.is_test and hour == 12:
                hour = 0
            minute = metadata["minute"]
            if self.is_test and minute == 60:
                minute = 0
            second = metadata["second"]
            answerable = hour > -1 or minute > -1 or second > -1

            scores["gt"] = f"{hour}:{minute}:{second}"
            if not answerable:
                scores["correctly_declines_to_answer"] = ":" not in pred
                scores["overall_close"] = scores["correctly_declines_to_answer"]
                scores["overall_exact"] = scores["correctly_declines_to_answer"]
                all_scores.append(scores)
                continue

            # pred = inputs["metadata/text"][ex_ix].decode("utf-8")
            parts = pred.split(":")
            try:
                pred_hour = int(parts[0].split()[-1])
                if "PM" in pred and not self.is_test:
                    pred_hour += 12
                if self.is_test and pred_hour >= 12:
                    pred_hour -= 12
                hour_correct = (pred_hour == hour) or (hour == 0 and pred_hour == 12)
            except (ValueError, IndexError):
                hour_correct = False
            scores["hour_correct"] = hour_correct

            try:
                minute_pred = int(parts[1].split()[0])
                if self.is_test and minute_pred == 60:
                    minute_pred = 0
                minute_correct = minute_pred == minute
                minute_close = abs(minute_pred - minute) <= err_threshold
                if self.is_test:
                    minute_close = minute_close or abs(minute_pred - minute) == 59
            except (ValueError, IndexError):
                minute_correct = 0
                minute_close = 0
            scores["minute_correct"] = minute_correct
            scores["minute_close"] = minute_close

            if second != -1:
                try:
                    second_pred = int(parts[2].split()[0])
                    second_correct = second_pred == second
                    second_close = abs(second_pred - second) <= err_threshold
                except (ValueError, IndexError):
                    second_correct = 0
                    second_close = 0
                scores["second_correct"] = second_correct
                scores["second_close"] = second_close
                scores["all_correct"] = minute_correct and hour_correct and second_correct
                scores["all_close"] = minute_close and hour_correct and second_close
            else:
                scores["all_correct"] = minute_correct and hour_correct
                scores["all_close"] = minute_close and hour_correct
                if self.is_test:
                    scores["all_close"] = scores["all_close"] or (
                        abs(hour * 60 + minute - (pred_hour * 60 + minute_pred)) == 719
                    )

            # FIXME can we remove?
            scores["overall_close"] = scores["all_close"]
            scores["overall_exact"] = scores["all_correct"]
            all_scores.append(scores)

        to_show = []
        for score in all_scores:
            to_show.append({k: score[k] for k in ["overall_close", "overall_exact"]})

        out = {}
        for k in self.METRICS:
            out[k] = mean_metric([x[k] for x in all_scores if k in x])
        if self.n_to_log:
            out["predictions"] = gather_examples_as_html(
                self.n_to_log, tokenizer, metadatas, predictions, to_show)
        return out


def compute_area(bbox: list, invalid: float=None) -> float:
    x1, y1, x2, y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1) * (y2 - y1)

    return area


def compute_iou(bbox1: list, bbox2: list, verbose: bool=False):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2

    x1_in = max(x1, x1_)
    y1_in = max(y1, y1_)
    x2_in = min(x2, x2_)
    y2_in = min(y2, y2_)

    intersection = compute_area(bbox=[x1_in, y1_in, x2_in, y2_in], invalid=0.0)
    area1 = compute_area(bbox1, invalid=0)
    area2 = compute_area(bbox2, invalid=0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou


class RefExpEval:
    """Refexp eval, not used by default"""

    def __init__(self, n_to_log=None):
        self.n_to_log = n_to_log

    def __call__(self, inputs, predictions, tokenizer, step=None):
        new_tokens = predictions["predictions"]
        prompt_tokens = predictions["prompts"]
        scores = []
        points = []
        gt_points = []
        for ex_ix, pred_seq in enumerate(new_tokens):
            pred = tokenizer.decode(pred_seq[pred_seq >= 0]).strip()
            pred_box = None
            image_w, image_h = inputs["metadata/image_size"][ex_ix]

            pred_box = extract_bboxes(pred, image_w, image_h)
            x0, y0, w, h = inputs["metadata/bbox"][ex_ix]
            ref_box = [x0, y0, x0+w, y0+h]
            gt_points.append([ref_box])
            if pred_box:
                iou = compute_iou(pred_box[0], ref_box)
                points.append([pred_box[0]])
            else:
                points.append([])
                iou = 0
            scores.append(dict(acc=iou>0.5, iou=iou, valid=bool(pred_box)))

        out = {}
        for k in scores[0]:
            vals = [x[k] for x in scores if k in x]
            out[k] = mean_metric(vals)
        return out
