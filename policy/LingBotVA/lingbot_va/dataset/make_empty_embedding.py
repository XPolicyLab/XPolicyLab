
#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import T5TokenizerFast, UMT5EncoderModel


def parse_args():
    parser = argparse.ArgumentParser(description="Create empty_emb.pt for LingBot-VA training.")
    parser.add_argument("--model-root", required=True, help="Wan2.2 Diffusers model directory.")
    parser.add_argument("--output", required=True, help="Output .pt path.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def to_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@torch.no_grad()
def encode_text(tokenizer, text_encoder, text, max_length, device, dtype):
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_len = int(attention_mask.gt(0).sum(dim=1).item())

    hidden = text_encoder(input_ids, attention_mask).last_hidden_state[0]
    hidden = hidden[:seq_len].to(dtype=dtype)
    padded = torch.cat(
        [hidden, torch.zeros(max_length - seq_len, hidden.shape[-1], dtype=dtype, device=hidden.device)],
        dim=0,
    )
    return padded.cpu()


def main():
    args = parse_args()
    model_root = Path(args.model_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = to_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = T5TokenizerFast.from_pretrained(model_root / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_root / "text_encoder", torch_dtype=dtype).to(device)
    text_encoder.eval()

    empty_emb = encode_text(tokenizer, text_encoder, "", args.max_length, device, dtype)
    torch.save(empty_emb, output_path)
    print(f"Saved: {output_path}")
    print(f"Shape: {tuple(empty_emb.shape)}")
    print(f"Dtype: {empty_emb.dtype}")

if __name__ == "__main__":
    main()