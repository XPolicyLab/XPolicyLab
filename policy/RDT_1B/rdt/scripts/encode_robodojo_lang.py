import argparse
import fnmatch
import os

import h5py
import torch
import yaml
from tqdm import tqdm

from models.multimodal_encoder.t5_encoder import T5Embedder


DEFAULT_MODEL_PATH = "/mnt/xspark-data/xspark_shared/model_weights/t5-v1_1-xxl"
DEFAULT_DATA_ROOT = "/mnt/nfs/niantian/RoboDojo_env/aloha_data/RoboDojo"


def _match(name, pattern):
    return pattern == "*" or fnmatch.fnmatchcase(name, pattern)


def _discover_task_dirs(data_root, task_pattern, env_pattern):
    task_dirs = []
    for task_name in sorted(os.listdir(data_root)):
        if not _match(task_name, task_pattern):
            continue
        task_root = os.path.join(data_root, task_name)
        if not os.path.isdir(task_root):
            continue
        for env_name in sorted(os.listdir(task_root)):
            if not _match(env_name, env_pattern):
                continue
            env_root = os.path.join(task_root, env_name)
            if os.path.isdir(env_root):
                task_dirs.append(env_root)
    return task_dirs


def _first_hdf5(task_dir):
    for filename in sorted(os.listdir(task_dir)):
        if filename.endswith(".hdf5"):
            return os.path.join(task_dir, filename)
    raise FileNotFoundError(f"No .hdf5 file found in {task_dir}")


def _read_instruction(task_dir):
    with h5py.File(_first_hdf5(task_dir), "r") as h5_file:
        instruction = h5_file.attrs.get("language_instruction", "")
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")
    if not instruction:
        raise ValueError(f"No language_instruction attr found in {task_dir}")
    return str(instruction)


def _encode_instructions(text_encoder, tokenizer, instructions, device):
    tokenized = tokenizer(
        instructions,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    input_ids = tokenized["input_ids"].to(device)
    attn_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = text_encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )["last_hidden_state"].detach().cpu()

    attn_mask = attn_mask.cpu().bool()
    return [text_embeds[i][attn_mask[i]] for i in range(len(instructions))]


def main():
    parser = argparse.ArgumentParser(description="Pre-encode one language embedding per RoboDojo task/env group.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task_pattern", default="*")
    parser.add_argument("--env_pattern", default="arx_x5")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config_path", default="configs/base.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--offload_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    task_dirs = _discover_task_dirs(args.data_root, args.task_pattern, args.env_pattern)
    if not task_dirs:
        raise FileNotFoundError(f"No task dirs matched under {args.data_root}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    text_embedder = T5Embedder(
        from_pretrained=args.model_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    pending = []
    for task_dir in task_dirs:
        save_path = os.path.join(task_dir, "lang_embed.pt")
        if args.overwrite or not os.path.exists(save_path):
            pending.append((task_dir, save_path, _read_instruction(task_dir)))

    for task_dir, save_path, instruction in tqdm(pending, desc="Encoding RoboDojo instructions"):
        embedding = _encode_instructions(text_encoder, tokenizer, [instruction], device)[0]
        torch.save(embedding, save_path)
        tqdm.write(f"saved {save_path}: {instruction}")

    empty_embed_path = os.path.join("data", "empty_lang_embed.pt")
    if args.overwrite or not os.path.exists(empty_embed_path):
        torch.save(_encode_instructions(text_encoder, tokenizer, [""], device)[0], empty_embed_path)
        print(f"saved {empty_embed_path}")

    print(f"Finished. Encoded {len(pending)} task/env groups.")


if __name__ == "__main__":
    main()
