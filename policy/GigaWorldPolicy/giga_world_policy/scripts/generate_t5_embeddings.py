"""Generate T5 text embeddings for LeRobot datasets using the UMT5 encoder from Wan2.2.

Usage:
    python scripts/generate_t5_embeddings.py \
        --data_root /path/to/lerobot_dataset \
        --wan_model_path /path/to/Wan2.2-TI2V-5B-Diffusers
"""

import argparse
import glob
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel


def collect_task_metadata(data_root: str) -> dict[str, list[dict]]:
    """Collect all (lerobot_path, task_descriptions) pairs.

    Supports both direct LeRobot v2.1 roots (<data_root>/meta/tasks.jsonl)
    and nested task/lerobot layouts.
    """
    result = {}

    def add_tasks(tasks_file: str):
        lerobot_dir = os.path.dirname(os.path.dirname(tasks_file))
        if lerobot_dir in result:
            return
        tasks = []
        with open(tasks_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        result[lerobot_dir] = tasks

    direct = os.path.join(data_root, "meta", "tasks.jsonl")
    if os.path.isfile(direct):
        add_tasks(direct)

    for tasks_file in sorted(
        glob.glob(os.path.join(data_root, "**", "lerobot", "meta", "tasks.jsonl"), recursive=True)
    ):
        add_tasks(tasks_file)
    return result


@torch.no_grad()
def encode_texts(
    texts: list[str], tokenizer, model, device, max_length=512
) -> list[torch.Tensor]:
    """Encode a batch of texts into UMT5 embeddings."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    outputs = model(**inputs)
    # outputs.last_hidden_state: [batch, seq_len, 4096]
    embeddings = outputs.last_hidden_state.cpu().float()
    results = []
    for i, length in enumerate(inputs.attention_mask.sum(dim=1)):
        results.append(embeddings[i, :length])
    return results


def materialize_symlink_dir(dir_path: str):
    """If dir_path is a symlink, replace it with a real dir and re-symlink contents."""
    if os.path.islink(dir_path):
        real_target = os.path.realpath(dir_path)
        os.unlink(dir_path)
        os.makedirs(dir_path, exist_ok=True)
        for item in os.listdir(real_target):
            src = os.path.join(real_target, item)
            dst = os.path.join(dir_path, item)
            if not os.path.exists(dst):
                os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", required=True, help="Root of pretrain_gwp dataset"
    )
    parser.add_argument(
        "--wan_model_path",
        default=os.environ.get("GIGAWORLD_PRETRAINED_PATH", os.environ.get("WAN22_DIFFUSERS_PATH", "")),
        help="Path to Wan2.2 diffusers model (contains text_encoder/ and tokenizer/). "
        "Defaults to $GIGAWORLD_PRETRAINED_PATH / $WAN22_DIFFUSERS_PATH.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    te_path = os.path.join(args.wan_model_path, "text_encoder")
    tok_path = os.path.join(args.wan_model_path, "tokenizer")

    print(f"Loading UMT5 encoder from: {te_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    model = UMT5EncoderModel.from_pretrained(te_path, torch_dtype=torch.float16).to(
        args.device
    )
    model.eval()

    print(f"Scanning datasets under: {args.data_root}")
    tasks_by_dataset = collect_task_metadata(args.data_root)
    print(f"Found {len(tasks_by_dataset)} sub-datasets")

    # Collect unique texts
    unique_texts = set()
    for tasks in tasks_by_dataset.values():
        for t in tasks:
            unique_texts.add(t["task"])
    unique_texts = sorted(unique_texts)
    print(f"Total unique task descriptions: {len(unique_texts)}")

    # Batch encode all unique texts
    text_to_embed = {}
    for i in tqdm(range(0, len(unique_texts), args.batch_size), desc="Encoding"):
        batch = unique_texts[i : i + args.batch_size]
        embeds = encode_texts(batch, tokenizer, model, args.device)
        for text, embed in zip(batch, embeds):
            text_to_embed[text] = embed

    # Save per-dataset t5_text_embeds.pt
    for lerobot_dir, tasks in tqdm(tasks_by_dataset.items(), desc="Saving"):
        meta_dir = os.path.join(lerobot_dir, "meta")
        materialize_symlink_dir(meta_dir)

        embed_dict = {}
        for t in tasks:
            embed_dict[t["task_index"]] = text_to_embed[t["task"]]

        out_path = os.path.join(meta_dir, "t5_text_embeds.pt")
        torch.save(embed_dict, out_path)

    print("Done! T5 embeddings saved to all sub-datasets.")


if __name__ == "__main__":
    main()
