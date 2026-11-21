#!/usr/bin/env python3
"""Serve a local vision-language model behind an OpenAI-compatible chat endpoint.

    CUDA_VISIBLE_DEVICES=4 python services/servers/serve_vlm.py --model /path/to/Qwen3-VL-8B-Instruct --port 8800

Runs in the harness-policy environment with transformers only (no vLLM), one
request at a time. Point a harness at it with backend = "openai_chat" and
base_url = "http://127.0.0.1:8800/v1". Meant for development runs that should
not spend API credit; throughput is modest.
"""

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import uuid


def load(model_path, dtype, quantize=None, max_memory=None, trust_remote_code=False,
         keep_modules=("visual", "lm_head")):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    options = {}
    if max_memory:
        # e.g. "0=14GiB,1=8GiB": spread layers so each GPU keeps room for the KV cache.
        options["max_memory"] = {int(k): v for k, v in (item.split("=") for item in max_memory.split(","))}
    if quantize == "4bit":
        # A 32B model in NF4 fits one 24 GB GPU; the vision tower stays in bf16.
        from transformers import BitsAndBytesConfig
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=list(keep_modules))
    elif quantize == "8bit":
        # LLM.int8: close to bf16 quality at about half the memory (an 8B model in ~10 GB).
        from transformers import BitsAndBytesConfig
        options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True,
                                                            llm_int8_skip_modules=list(keep_modules))
    if trust_remote_code:
        # Models whose code ships with the checkpoint (Molmo2).
        options["trust_remote_code"] = True
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=getattr(torch, dtype), device_map="auto", attn_implementation="sdpa", **options)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    return model.eval(), processor


def to_template(messages, max_side=None):
    """OpenAI chat messages -> Qwen-VL chat-template messages with PIL images.
    Images larger than max_side pixels on their long edge are downscaled, which
    cuts vision tokens roughly with the pixel count."""
    from PIL import Image

    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) or content is None:
            parts = [{"type": "text", "text": content or ""}]
        else:
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:") or ";base64," not in url:
                        raise ValueError("Only base64 data-URL images are accepted")
                    raw = base64.b64decode(url.split(";base64,", 1)[1])
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                    if max_side and max(image.size) > max_side:
                        scale = max_side / max(image.size)
                        image = image.resize((round(image.width * scale), round(image.height * scale)))
                    parts.append({"type": "image", "image": image})
                else:
                    raise ValueError(f"Unsupported content part: {part['type']}")
        converted.append({"role": message["role"], "content": parts})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", help="model id reported to clients (default: directory name)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--quantize", choices=["4bit", "8bit"],
                        help="load weights in 4-bit NF4 or 8-bit LLM.int8 (bitsandbytes)")
    parser.add_argument("--keep-modules", default="visual,lm_head",
                        help="modules left unquantized (name substrings, comma-separated)")
    parser.add_argument("--max-memory", help="per-GPU weight budget, e.g. 0=14GiB,1=8GiB; splitting a "
                        "4-bit model over two GPUs was about 5x slower per token in our tests")
    parser.add_argument("--max-image-side", type=int, help="downscale images to this long edge (pixels)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="run the model code shipped with the checkpoint (needed for Molmo2)")
    args = parser.parse_args()
    import torch

    name = args.name or args.model.rstrip("/").split("/")[-1]
    model, processor = load(args.model, args.dtype, args.quantize, args.max_memory, args.trust_remote_code,
                            tuple(filter(None, args.keep_modules.split(","))))
    lock = threading.Lock()
    print(json.dumps({"serving": name, "port": args.port}), flush=True)

    def complete(request):
        # A request may override the downscaling (perception sends 0 to keep full resolution).
        messages = to_template(request["messages"], request.get("max_image_side", args.max_image_side))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        max_new = min(int(request.get("max_tokens") or args.max_new_tokens), args.max_new_tokens)
        temperature = float(request.get("temperature") or 0.0)
        options = {"max_new_tokens": max_new, "do_sample": temperature > 0}
        if temperature > 0:
            options["temperature"] = temperature
        with lock, torch.inference_mode():
            try:
                output = model.generate(**inputs, **options)
            except torch.cuda.OutOfMemoryError:
                del inputs
                torch.cuda.empty_cache()
                raise RuntimeError("out of GPU memory for this context; shorten the conversation") from None
            finally:
                # Long multi-image contexts leave large cached blocks behind.
                torch.cuda.empty_cache()
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated = output[0, prompt_tokens:]
        text = processor.decode(generated, skip_special_tokens=True)
        finish = "length" if len(generated) >= max_new else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": int(len(generated)),
                      "total_tokens": prompt_tokens + int(len(generated)), "cost": 0.0},
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *items):
            pass

        def _send(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(size))
                started = time.perf_counter()
                body = complete(request)
                print(json.dumps({"prompt_tokens": body["usage"]["prompt_tokens"],
                                  "completion_tokens": body["usage"]["completion_tokens"],
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)
                self._send(200, body)
            except Exception as error:
                self._send(400, {"error": {"message": f"{type(error).__name__}: {error}"}})

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
