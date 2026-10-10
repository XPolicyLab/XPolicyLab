import argparse
import base64
import os
from glob import glob
from io import BytesIO
from typing import Any, Dict

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

from llava.utils import io
from llava.utils.logging import logger

app = Flask(__name__)


def _initialize(mmdb_dir: str) -> Dict[str, Any]:
    config = io.load(os.path.join(mmdb_dir, "config.json"))
    model_name_or_path = config["model_name_or_path"]

    # Load model and processor
    model = AutoModel.from_pretrained(model_name_or_path).cuda()
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    # Load image features and metainfos
    features, metainfos = [], []
    for fpath in glob(os.path.join(mmdb_dir, "*.pt")):
        features.append(io.load(fpath, map_location="cuda"))
        metainfos.extend(io.load(fpath.replace(".pt", ".jsonl")))
    features = torch.cat(features, dim=0)

    # Remove duplicate image paths
    indices = []
    paths = set()
    for index, metainfo in enumerate(metainfos):
        if metainfo["path"] not in paths:
            indices.append(index)
            paths.add(metainfo["path"])
    features = features[indices]
    metainfos = [metainfos[i] for i in indices]

    logger.info(f"Loaded {len(features)} image features from '{mmdb_dir}' with unique paths.")
    return {"model": model, "processor": processor, "features": features, "metainfos": metainfos}


def _encode(image) -> str:
    if isinstance(image, str):
        image = Image.open(image)
    buffer = BytesIO()
    try:
        image.save(buffer, format="JPEG")
    except Exception:
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    # Load and preprocess the image
    image = Image.open(request.files["image"]).convert("RGB")
    inputs = app.config["processor"](images=[image], return_tensors="pt").to("cuda")

    # Extract image features from the query image
    with torch.inference_mode(), torch.cuda.amp.autocast():
        query = app.config["model"].get_image_features(**inputs)

    # Compute cosine similarity and retrieve most similar images
    scores = cosine_similarity(query, app.config["features"])
    scores, indices = scores.topk(10)

    # Prepare the response with metadata of the top 5 images
    responses = []
    for index, score in zip(indices, scores):
        metainfo = app.config["metainfos"][index.item()]
        response = {
            "uid": metainfo["uid"],
            "path": metainfo["path"],
            "score": score.item(),
            "image": _encode(metainfo["path"]),
        }
        responses.append(response)
    return jsonify({"image": _encode(image), "responses": responses})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdb-dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app.config.update(_initialize(args.mmdb_dir))
    app.run(host="0.0.0.0", port=args.port)
