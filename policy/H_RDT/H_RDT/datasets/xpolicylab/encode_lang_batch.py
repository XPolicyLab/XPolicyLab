import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from models.encoder.t5_encoder import T5Embedder


def main():
    current_dir = Path(__file__).resolve().parent
    csv_path = current_dir / "task_instructions.csv"
    output_dir = current_dir / "lang_embeddings"
    output_dir.mkdir(exist_ok=True)

    model_path = os.environ.get("T5_MODEL_PATH", str(project_root / "t5-v1_1-xxl"))
    config_path = os.environ.get("HRDT_CONFIG_PATH", str(project_root / "configs" / "hrdt_finetune.yaml"))
    gpu_id = os.environ.get("HRDT_LANG_GPU", "0")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"[xpolicylab] reading task instructions from: {csv_path}")
    print(f"[xpolicylab] output directory: {output_dir}")
    print(f"[xpolicylab] T5 model path: {model_path}")
    print(f"[xpolicylab] config path: {config_path}")
    print(f"[xpolicylab] device: {device}")

    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    df = pd.read_csv(csv_path)
    text_embedder = T5Embedder(
        from_pretrained=model_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=None,
    )
    tokenizer = text_embedder.tokenizer
    text_encoder = text_embedder.model

    success_count = 0
    for idx, row in df.iterrows():
        task_name = row["task_name"]
        instruction = row["instruction"]
        save_path = output_dir / f"{task_name}.pt"

        print(f"\n[xpolicylab] Processing [{idx + 1}/{len(df)}]: {task_name}")
        print(f"[xpolicylab] Instruction: {instruction}")

        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )["input_ids"].to(device)

        with torch.no_grad():
            embeddings = text_encoder(tokens.view(1, -1)).last_hidden_state.detach().cpu()

        torch.save(
            {
                "name": task_name,
                "instruction": instruction,
                "embeddings": embeddings,
            },
            save_path,
        )
        print(f"[xpolicylab] saved {save_path} with shape {tuple(embeddings.shape)}")
        success_count += 1

    print(f"\n[xpolicylab] Batch encoding completed: {success_count}/{len(df)}")


if __name__ == "__main__":
    main()
