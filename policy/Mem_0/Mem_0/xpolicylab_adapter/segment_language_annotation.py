"""
Generate Mem_0 Mn sub-task annotations from XPolicyLab episodes with a VLM.

Self-contained, in-project adaptation of the Ego-X caption operator
(~/Desktop/zijian/ego/Ego-X_Operator/operators/caption: segment_v2t.py /
vlm_api.py). It reuses that operator's approach -- uniform frame sampling,
multimodal VLM request, robust JSON parsing, and merging of same-objective runs
into one captioned sub-task -- but drops the Ego-X framework dependencies and
emits exactly the annotation format the Mem_0 converter consumes.

Two segmentation modes:

* free mode (default, used by tasks like cover_blocks):
    1. read head-camera frames from the XPolicyLab HDF5 (same source as training);
    2. uniformly sample K frames spanning the episode (keeping their frame indices);
    3. one multimodal VLM call -> sequential sub-tasks [{instruction,start,end}];
    4. contiguous coverage over [0, episode_length); merge adjacent identical
       instructions so the SAME sub-task carries an IDENTICAL instruction.

* strict template mode (used by tasks with a fixed sub-task script, e.g. swap_T):
    1. load instruction/<task_name>.json (or --template PATH); the template fixes
       BOTH the count and the exact ordered list of sub-task instruction strings;
    2. uniform frame sampling as above;
    3. one VLM call that ONLY produces the start_frame of each sub-task -- the
       instruction texts come verbatim from the template;
    4. contiguous coverage; adjacent identical instructions are PRESERVED as
       distinct segments (the template encodes the desired phasing).

Output: language_annotation.json = {"episode_<i>": [[subtask_text, duration], ...]}
written under <Mem_0>/language_annotations/<dataset>/<task>/<env_cfg>/, where
process_data.sh ... Mn auto-discovers it. Source data/ stays untouched.

Run in an env with: openai (or httpx for Ark), opencv, h5py, numpy, and a VLM key.
  export VLM_API_PROVIDER=dashscope            # or volcengine_ark
  export DASHSCOPE_API_KEY=...                 # or ARK_API_KEY / VLM_API_KEY
  export VLM_MODEL=qwen3.5-flash               # optional override

  # free mode
  python segment_language_annotation.py RoboDojo cover_blocks arx_x5 50 \
      --global_task "cover the blocks left-to-right then uncover red,green,blue"

  # strict template mode (auto-discovers instruction/swap_T.json)
  python segment_language_annotation.py RoboDojo swap_T arx_x5 50
"""

import argparse
import base64
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ADAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.dirname(ADAPTER_DIR)
ROOT_DIR = os.path.abspath(os.path.join(UPSTREAM_DIR, "..", "..", "..", ".."))  # repo root
for p in (ROOT_DIR, UPSTREAM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from XPolicyLab.utils.load_file import load_hdf5  # noqa: E402
from XPolicyLab.utils.process_data import decode_image_bit  # noqa: E402

TARGET_W, TARGET_H = 640, 480
INSTRUCTION_DIR = os.path.join(ADAPTER_DIR, "instruction")
# Where this adapter writes its derived artifacts. Mirrors lerobot_datasets/ in spirit:
# all VLM annotations live next to the upstream Mem_0 sources, NOT inside data/.
ANNOTATIONS_ROOT = os.path.join(UPSTREAM_DIR, "language_annotations")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
PROVIDER_DEFAULT_MODELS = {"dashscope": "qwen3-vl-plus", "volcengine_ark": "doubao-seed-2-0-lite-260215"}


# --------------------------------------------------------------------------- #
# VLM client (compact OpenAI-compatible / Ark adaptation of caption/vlm_api.py)
# --------------------------------------------------------------------------- #
def _provider() -> str:
    return os.getenv("VLM_API_PROVIDER", "dashscope").strip().lower() or "dashscope"


def _api_key() -> str:
    if _provider() == "volcengine_ark":
        return os.getenv("ARK_API_KEY", "").strip() or os.getenv("VLM_API_KEY", "").strip()
    return os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("VLM_API_KEY", "").strip()


def _base_url() -> str:
    return DEFAULT_ARK_BASE_URL if _provider() == "volcengine_ark" else DEFAULT_BASE_URL


def _model(cli_model: str | None) -> str:
    return (cli_model or os.getenv("VLM_MODEL", "").strip()
            or PROVIDER_DEFAULT_MODELS.get(_provider(), "qwen3.5-plus"))


def _build_message(image_b64_list, prompt):
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
               for b in image_b64_list if b]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _vlm_chat(messages, model: str) -> str | None:
    """One multimodal chat completion. dashscope/openai-compatible via openai SDK; Ark via httpx."""
    key = _api_key()
    if not key:
        raise RuntimeError(
            f"{_provider()} API key not set (DASHSCOPE_API_KEY / ARK_API_KEY / VLM_API_KEY)."
        )
    if _provider() == "volcengine_ark":
        import httpx
        resp = httpx.Client(timeout=180.0).post(
            f"{_base_url().rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages},
        )
        resp.raise_for_status()
        body = resp.json()
        return (body.get("choices") or [{}])[0].get("message", {}).get("content")

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=_base_url())
    resp = client.chat.completions.create(model=model, messages=messages)
    choices = getattr(resp, "choices", None) or []
    return getattr(choices[0].message, "content", None) if choices else None


def _parse_json(raw: str | None):
    """Robust JSON extraction (adapted from caption/segment_v2t._parse_vlm_json)."""
    if not raw:
        return None
    cleaned = str(raw).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for m in re.finditer(r"[\{\[]", cleaned):
            try:
                value, _ = decoder.raw_decode(cleaned[m.start():])
                return value
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Strict template loader
# --------------------------------------------------------------------------- #
def _load_template(template_path: str | None, task_name: str):
    """Return (global_task_or_None, [instruction, ...]) or None if no template applies.

    Lookup order:
      1. explicit --template path (raise if missing/invalid);
      2. instruction/<task_name>.json next to this script (silent skip if absent).
    Template format: {"global_task": "...", "subtasks": ["...", "...", ...]}.
    """
    explicit = bool(template_path)
    if not template_path:
        candidate = os.path.join(INSTRUCTION_DIR, f"{task_name}.json")
        if not os.path.isfile(candidate):
            return None
        template_path = candidate
    elif not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    try:
        raw = json.loads(Path(template_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if explicit:
            raise
        return None  # malformed legacy file (e.g. excerpt) -> fall back to free mode
    if not isinstance(raw, dict) or "subtasks" not in raw:
        if explicit:
            raise ValueError(
                f"Template {template_path} must be a dict with a 'subtasks' list "
                "of instruction strings."
            )
        return None  # legacy format like cover_blocks.json -> fall back to free mode
    subtasks = [str(s).strip() for s in raw.get("subtasks") or []]
    if not all(subtasks):
        raise ValueError(f"Template {template_path}: every entry in 'subtasks' must be non-empty.")
    if len(subtasks) < 1:
        raise ValueError(f"Template {template_path}: 'subtasks' must contain at least one entry.")
    global_task = raw.get("global_task")
    if global_task is not None:
        global_task = str(global_task).strip() or None
    return global_task, subtasks


# --------------------------------------------------------------------------- #
# Frame sampling + prompts + segmentation
# --------------------------------------------------------------------------- #
def _sample_frames(colors, num_frames: int):
    """Uniformly sample frame indices; return (frame_ids, [b64 jpeg]). cv2 keeps BGR -> correct JPEG."""
    nframes = len(colors)
    frame_ids = np.linspace(0, nframes - 1, min(num_frames, nframes), dtype=int).tolist()
    b64 = []
    for fid in frame_ids:
        img = decode_image_bit(colors[fid])               # BGR
        img = cv2.resize(img, (TARGET_W, TARGET_H))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            b64.append(base64.b64encode(buf).decode("ascii"))
    return frame_ids, b64, nframes


def _build_prompt(global_task: str, frame_ids, nframes: int) -> str:
    mapping = ", ".join(f"image[{i}]=frame {fid}" for i, fid in enumerate(frame_ids))
    return f"""\
You are analyzing a robot manipulation video of the task: {global_task}
The full episode has {nframes} frames (indices 0..{nframes - 1}). You are given
{len(frame_ids)} frames sampled evenly across the episode, in order. The original
frame index of each image is: {mapping}.

Split the episode into the sequence of distinct sub-tasks that are performed.
Rules:
- Sub-tasks are contiguous and ordered; together they cover frames 0..{nframes - 1}.
- Give each sub-task ONE short imperative instruction.
- If the same kind of sub-task repeats, REUSE the IDENTICAL instruction string.
- start_frame/end_frame are ORIGINAL frame indices (0..{nframes - 1}).

Output ONLY valid JSON:
{{
  "subtasks": [
    {{"instruction": "...", "start_frame": 0, "end_frame": <int>}},
    ...
  ]
}}"""


def _build_prompt_strict(global_task: str, frame_ids, nframes: int, subtasks: list[str]) -> str:
    """Strict-template prompt: VLM only picks start_frame for each fixed sub-task."""
    mapping = ", ".join(f"image[{i}]=frame {fid}" for i, fid in enumerate(frame_ids))
    bullets = []
    for i, instr in enumerate(subtasks):
        suffix = ""
        if i > 0 and instr == subtasks[i - 1]:
            suffix = " (continuation/second phase of the previous sub-task; identify the natural mid-point split)"
        bullets.append(f"  {i + 1}. {instr}{suffix}")
    listing = "\n".join(bullets)
    n = len(subtasks)
    return f"""\
You are analyzing a robot manipulation video of the task: {global_task}
The full episode has {nframes} frames (indices 0..{nframes - 1}). You are given
{len(frame_ids)} frames sampled evenly across the episode, in order. The original
frame index of each image is: {mapping}.

The episode STRICTLY follows this fixed ordered sequence of {n} sub-tasks
(do NOT change, add, drop, reorder, or rephrase any of them):
{listing}

Identify the ORIGINAL start_frame index of each sub-task.
Rules:
- Output exactly {n} integers in "starts", one per sub-task above, in the SAME order.
- starts[0] must be 0.
- Each subsequent starts[i] must satisfy starts[i-1] < starts[i] < {nframes}.
- The final sub-task implicitly ends at frame {nframes - 1}.
- When two adjacent sub-tasks share the same instruction, they describe two
  consecutive phases of the same goal (e.g. main motion vs. ending/release);
  pick the natural transition frame between them.

Output ONLY valid JSON (no extra fields):
{{
  "starts": [0, <int>, <int>, ...]
}}"""


def _to_segments(parsed, nframes: int, global_task: str):
    """Normalize VLM output -> contiguous [(instruction, duration)] covering [0, nframes)."""
    subs = (parsed or {}).get("subtasks") if isinstance(parsed, dict) else None
    raw = []
    for s in subs or []:
        instr = str(s.get("instruction", "")).strip()
        try:
            start = int(s.get("start_frame"))
            end = int(s.get("end_frame"))
        except (TypeError, ValueError):
            continue
        if instr and end >= start:
            raw.append([instr, max(0, start), min(nframes - 1, end)])
    if not raw:
        return [(global_task, nframes)]  # fallback: whole episode as one sub-task

    raw.sort(key=lambda x: x[1])
    # Force contiguous coverage: first starts at 0, each starts where the previous ended.
    raw[0][1] = 0
    for i in range(1, len(raw)):
        raw[i][1] = raw[i - 1][2] + 1
        if raw[i][2] < raw[i][1]:
            raw[i][2] = raw[i][1]
    raw[-1][2] = nframes - 1

    # Merge adjacent identical instructions (one identical text per sub-task).
    merged = [list(raw[0])]
    for instr, start, end in raw[1:]:
        if instr == merged[-1][0]:
            merged[-1][2] = end
        else:
            merged.append([instr, start, end])
    return [(instr, end - start + 1) for instr, start, end in merged]


def _to_segments_strict(parsed, nframes: int, subtasks: list[str]):
    """Force contiguous coverage with EXACTLY len(subtasks) segments, instructions fixed.

    Adjacent identical instructions are kept separate (the template asked for them).
    Falls back to an even split if the VLM output is unusable.
    """
    n = len(subtasks)
    starts: list[int] | None = None
    if isinstance(parsed, dict):
        cand = parsed.get("starts")
        if isinstance(cand, list) and len(cand) == n:
            try:
                starts = [int(x) for x in cand]
            except (TypeError, ValueError):
                starts = None

    if starts is None:
        # Even split fallback: starts[i] = round(i * nframes / n).
        starts = [int(round(i * nframes / n)) for i in range(n)]

    # Sanitize: clamp to [0, nframes-1], starts[0]=0, strictly increasing, leave room for the tail.
    starts[0] = 0
    for i in range(1, n):
        lo = starts[i - 1] + 1
        hi = nframes - (n - i)               # ensure each remaining sub-task gets >=1 frame
        starts[i] = max(lo, min(int(starts[i]), hi))

    # ends[i] = starts[i+1] - 1; last ends at nframes-1.
    ends = [starts[i + 1] - 1 for i in range(n - 1)] + [nframes - 1]
    return [(subtasks[i], ends[i] - starts[i] + 1) for i in range(n)]


def _segment_episode(load_path: str, camera: str, num_frames: int, global_task: str, model: str,
                     template_subtasks: list[str] | None = None):
    data = load_hdf5(load_path)
    colors = data["vision"][camera]["colors"]
    frame_ids, b64, nframes = _sample_frames(colors, num_frames)
    if not b64:
        if template_subtasks:
            return _to_segments_strict(None, nframes, template_subtasks)
        return [(global_task, nframes)]
    if template_subtasks:
        prompt = _build_prompt_strict(global_task, frame_ids, nframes, template_subtasks)
        text = _vlm_chat(_build_message(b64, prompt), model)
        return _to_segments_strict(_parse_json(text), nframes, template_subtasks)
    prompt = _build_prompt(global_task, frame_ids, nframes)
    text = _vlm_chat(_build_message(b64, prompt), model)
    return _to_segments(_parse_json(text), nframes, global_task)


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM sub-task segmentation -> language_annotation.json")
    parser.add_argument("dataset_name")
    parser.add_argument("task_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("expert_data_num", type=int)
    parser.add_argument("--global_task", default=None, help="overall task description (default <task_name>)")
    parser.add_argument("--num_frames", type=int, default=24, help="frames sampled per episode for the VLM")
    parser.add_argument("--camera", default="cam_head")
    parser.add_argument("--model", default=None, help="VLM model (else VLM_MODEL env / provider default)")
    parser.add_argument("--max_workers", type=int, default=4, help="parallel VLM requests")
    parser.add_argument("--out", default=None,
                        help="output path (default: language_annotations/<dataset>/<task>/<env_cfg>/language_annotation.json)")
    parser.add_argument("--template", default=None,
                        help='strict-mode template JSON {"global_task":"...","subtasks":["...",...]}; '
                             "if omitted, instruction/<task_name>.json is auto-discovered")
    args = parser.parse_args()

    template = _load_template(args.template, args.task_name)
    template_subtasks: list[str] | None = None
    template_global: str | None = None
    if template is not None:
        template_global, template_subtasks = template
        tqdm.write(f"[segment] strict template: {len(template_subtasks)} sub-tasks "
                   f"({len(set(template_subtasks))} unique)")

    global_task = args.global_task or template_global or args.task_name
    model = _model(args.model)
    load_dir = os.path.join(ROOT_DIR, "final_data", args.dataset_name, args.task_name, args.env_cfg_type)
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Source data dir not found: {load_dir}")
    out_path = args.out or os.path.join(
        ANNOTATIONS_ROOT, args.dataset_name, args.task_name, args.env_cfg_type,
        "language_annotation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    episodes = []
    for idx in range(args.expert_data_num):
        p = os.path.join(load_dir, "data", f"episode_{idx:07d}.hdf5")
        if os.path.isfile(p):
            episodes.append((idx, p))
        else:
            tqdm.write(f"[segment] skip missing {p}")
    if not episodes:
        raise RuntimeError(f"No episodes found under {load_dir}/data")

    annotations: dict[str, list] = {}
    bar = tqdm(total=len(episodes), desc=f"segment {args.task_name} [{model}]", unit="ep", dynamic_ncols=True)

    def _work(item):
        idx, path = item
        return idx, _segment_episode(path, args.camera, args.num_frames, global_task, model,
                                     template_subtasks=template_subtasks)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(_work, it): it[0] for it in episodes}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _idx, segments = fut.result()
                annotations[f"episode_{idx}"] = [[instr, dur] for instr, dur in segments]
                bar.set_postfix(episode=idx, subtasks=len(segments))
            except Exception as exc:  # keep going; one bad episode shouldn't abort the batch
                tqdm.write(f"[segment] episode {idx} failed: {exc}")
            bar.update(1)
    bar.close()

    if not annotations:
        raise RuntimeError("No episodes segmented (all VLM requests failed?).")
    Path(out_path).write_text(json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8")
    tqdm.write(f"[segment] wrote {len(annotations)} episodes -> {out_path}")


if __name__ == "__main__":
    main()
