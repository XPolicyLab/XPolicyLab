"""Visually enhance instructions using prepared human/robot condition images.

This runs separately from Wan so Qwen can use a different Python environment.
Both emitted prompt manifests can be passed directly to scripts/infer_video.sh.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import time

from .inference import load_inputs
from .utils.conditioning import load_views, prepare_image


SUFFIXES = {
    "robot": "This is a composed robot video with two wrist-view in the right side.",
    "human": "This is an ego-view video, and the camera view keeps static.",
}

SYSTEM_PROMPT = """You enhance manipulation instructions for image-to-video prediction.
Use only the supplied initial image and instruction. Preserve the exact task, target
objects, spatial qualifiers, and desired final state. Ground object descriptions in
the image. Do not invent objects, tools, obstacles, extra tasks, or camera motion.
The instruction specifies the GOAL, not just the first motion. Include completion:
reaching/grasping alone does not complete putting, stacking, pouring, or wiping.
Preserve the requested METHOD too: sweeping means using a brush to push an object
along the surface, never grasping and carrying that object. Grasp the tool, not its
target. A sliding drawer closes by pushing inward, not downward. Do not invent
opening/closing a tissue box when extracting a tissue. Only mention grasping an
object if needed; set held objects down before releasing them.
Never replace opening/closing a physical lid or door with pressing a power button.

Choose which hand/arm can perform the task most naturally from the visible starting
positions: consider reach, which side is nearest, current grasp, and avoiding crossed
arms. Prefer one active hand when sufficient; use both when support or coordination
is needed. An instruction explicitly naming a hand takes precedence. For a side
that need not move, write exactly "keeps static". Do not make both hands active just
to fill the format, and do not make both static for an action instruction.

Describe the LEFT side first and RIGHT side second. Left/right identify the hands
or arms on the left/right of the MAIN image, not the small wrist-view panels.
Detail a short physically plausible motion: approach/grasp when needed, movement
or manipulation, and completion/release when appropriate. Coordinate simultaneous
support with the other side. Do not merely describe the initial image or speculate
about actions that already happened. Do not add unnecessary repositioning afterwards.

Before writing the two actions, explicitly identify the requested final state and
compare the two visible arms' reach to the target. Do not default to the left arm.
Return ONLY a JSON object with these four keys, in this order:
"task_goal": one sentence stating the instruction's exact intended final state;
"arm_choice": briefly locate the target in the MAIN image and justify which side
acts and whether the other side must support the object;
"left_action" and "right_action": detailed actions that jointly achieve task_goal.
Each action is an English verb phrase in third-person present tense that completes
"The left/right robot arm ..." or "The left/right hand ...". Do not repeat the subject
inside the value. Use a single sentence per side; aim for 50-120 words for the two
actions together and never exceed 180 words. Include concrete approach, manipulation,
and final placement/release as appropriate, not just a short restatement of the task.
Use "keeps static" for a side that need not move. Example action-pair syntax:
{"left_action": "keeps static", "right_action": "reaches toward the red block near
its gripper, closes around the block, lifts it clear of the table, carries it above
the nearby tray, then lowers and releases it inside the tray"}.
This example only illustrates syntax; choose sides and objects from the actual image
and instruction. Do not include the camera/view suffix. Check that the action clauses
include the requested destination and final state before returning the JSON.
"""

REVIEW_PROMPT = """Check an expanded robot/human manipulation instruction against its
short source instruction. Judge TASK FIDELITY, not writing style or visual grounding.
The drafting model has already viewed the image. Accept its choice of arms and
reasonable added visual attributes; do not invent objections about object presence,
colors, arm visibility, or whether extra support is necessary.
Return ONLY JSON with keys "valid" (boolean) and "feedback" (short string).
Use valid=false if the draft changes the requested manipulation method, omits the
destination or completed goal, invents unrelated steps, or contains a clear physical
contradiction. Examples: sweeping requires pushing with the brush, not picking up
the target; an open laptop closes by lowering its lid, not pressing a power button;
a drawer slides inward, not downward; a stationary hand cannot newly grasp an object.
Reject simply reaching for or grasping an object without completing the task.
Keep existing left/right and positional qualifiers from the source instruction.
Do not demand a different arm just from personal preference: either may act if
physically plausible. Do not invent problems or insist on additional steps. If
valid=false, explain the specific contradiction and the minimal correction.
"""


def parse_json_response(text):
    # Thinking may start in the chat template rather than the generated token span.
    text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return json.loads(text)


def format_prompt(actions, input_type):
    required = {"left_action", "right_action"}
    if not isinstance(actions, dict) or not required <= set(actions) or set(actions) - (
        required | {"task_goal", "arm_choice"}
    ):
        raise ValueError("Expected JSON with left_action and right_action, plus optional task_goal/arm_choice")
    clauses = []
    for side in ("left", "right"):
        action = actions[f"{side}_action"]
        if not isinstance(action, str) or not action.strip():
            raise ValueError(f"Missing {side} action")
        action = " ".join(action.strip().split()).rstrip(". ")
        # Some instruct models repeat the requested subject. Remove that redundant
        # prefix without asking the model to change an otherwise valid action.
        action = re.sub(
            rf"(?i)^(?:the\s+)?{side}\s+(?:(?:robot|human)\s+)?(?:arm|hand|gripper)\s+",
            "", action,
        )
        if not action:
            raise ValueError(f"Missing {side} action after subject")
        action = action[0].lower() + action[1:]
        if action.lower() in {"keep static", "keeps static", "remains static", "remains stationary"}:
            action = "keeps static"
        if "@@" in action or "<think>" in action or re.match(
            r"(?i)^(the\s+)?(left|right)\s+(robot|human|hand|arm)", action
        ):
            raise ValueError("Each action must be a verb phrase, without subject or formatting")
        subject = f"The {side} robot arm" if input_type == "robot" else f"The {side} hand"
        clauses.append(f"{subject} {action}.")
    if all(clause.endswith(" keeps static.") for clause in clauses):
        raise ValueError("Both sides were static; an action instruction must describe motion")
    if len(" ".join(clauses).split()) > 190:
        raise ValueError("Actions are too long; use at most 180 words")
    left, right = clauses
    return f"{left[:-1]}, while the {right.removeprefix('The ')} {SUFFIXES[input_type]}"


class QwenEnhancer:
    def __init__(self, model_dir, device="cuda:0", thinking=False, reasoning_effort="medium",
                 max_new_tokens=4096):
        import torch
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
        except ImportError as exc:
            raise RuntimeError(
                "Use Transformers >=4.57 for Qwen3-VL. Models requiring Transformers 5 "
                "should run in a separate Qwen environment."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Qwen enhancement requires CUDA; use --preprocess-only for CPU work")
        torch.cuda.set_device(self.device)
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_new_tokens = max_new_tokens
        print(f"Loading Qwen from {model_dir} on {self.device}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map={"": str(self.device)},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.generation_config = GenerationConfig(
            do_sample=False, use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

    def generate(self, image, system, user, *, thinking, max_new_tokens):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                *([{"type": "image", "image": image}] if image is not None else []),
                {"type": "text", "text": user},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking, reasoning_effort=self.reasoning_effort,
        )
        inputs = self.processor(
            text=[text], images=[image] if image is not None else None, return_tensors="pt",
        ).to(self.device)
        generation_config = copy.deepcopy(self.generation_config)
        generation_config.max_new_tokens = max_new_tokens
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, generation_config=generation_config, use_model_defaults=False,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        eos = self.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        complete = len(generated) > 0 and int(generated[-1]) in (eos or [])
        decoded = self.processor.decode(generated, skip_special_tokens=True)
        return decoded, complete, len(generated)

    def classify(self, image):
        text, complete, _ = self.generate(
            image,
            'Classify the manipulation scene. Return only JSON: {"input_type":"human"} '
            'for human hands in an ego view, or {"input_type":"robot"} for robot arms. '
            'If ambiguous, return {"input_type":"unknown"}.',
            "Identify the kind of manipulator in this main camera image.",
            thinking=False, max_new_tokens=128,
        )
        kind = parse_json_response(text).get("input_type")
        if not complete or kind not in SUFFIXES:
            raise ValueError(f"Cannot classify image; explicitly set --input-type human/robot: {text}")
        return kind

    def enhance(self, image, instruction, input_type, missing_views):
        scene = (
            "This is a robot composite: main camera on the left, wrist cameras stacked "
            "on the right. Wrist panels show the same scene, not extra arms. "
            f"Missing camera panels {missing_views} are black padding, not physical objects."
            if input_type == "robot" else
            "This is a single ego-view image of human hands. The camera stays fixed."
        )
        user = (
            f"{scene}\nInstruction: {instruction}\n"
            "Use task_goal and arm_choice to plan from the image, then provide detailed "
            "left_action and right_action that complete this instruction. Describe the "
            "entire short manipulation through its final state, not only the initial grasp."
        )
        attempts = []
        for attempt in range(3):
            start = time.monotonic()
            # Allow more reasoning tokens after a truncated or malformed response.
            response, complete, count = self.generate(
                image, SYSTEM_PROMPT, user, thinking=self.thinking,
                max_new_tokens=self.max_new_tokens * (2 if attempt else 1),
            )
            attempts.append({"response": response, "complete": complete,
                             "generated_tokens": count, "seconds": time.monotonic() - start})
            try:
                if not complete:
                    raise ValueError("Response hit the generation limit before EOS")
                actions = parse_json_response(response)
                prompt = format_prompt(actions, input_type)
                review_text, review_complete, _ = self.generate(
                    None, REVIEW_PROMPT,
                    f"Source instruction: {instruction}\nDraft: {prompt}",
                    thinking=False, max_new_tokens=1024,
                )
                if not review_complete:
                    raise ValueError("Task-fidelity review was truncated")
                review = parse_json_response(review_text)
                attempts[-1]["review"] = review
                if not isinstance(review, dict) or not isinstance(review.get("valid"), bool):
                    raise ValueError("Review must return a JSON boolean valid")
                if not review["valid"]:
                    raise ValueError(f"Task-fidelity correction: {review.get('feedback', '')}")
                return prompt, {"actions": actions, "attempts": attempts}
            except (ValueError, TypeError) as exc:
                attempts[-1]["error"] = str(exc)
                print(f"Retrying prompt enhancement: {exc}", flush=True)
                user += f"\nCorrection: {exc}. Return the required complete JSON only."
        raise ValueError(f"Qwen failed to produce a valid enhancement: {attempts[-1]}")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest_line(prompt, image_path, manifest):
    return f"{prompt}@@{Path(os.path.relpath(image_path, manifest.parent)).as_posix()}\n"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True,
                        help="UTF-8 lines: instruction@@image_path_or_view_directory")
    parser.add_argument("--input-type", choices=("auto", "robot", "human"), default="auto",
                        help="Auto uses Qwen to classify the main image; explicit types skip classification")
    parser.add_argument("--qwen-model-dir", type=Path,
                        default=os.environ.get("RecoverWAM_QWEN_ROOT", "weights/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable reasoning for models that support it (e.g. Qwen3.8)")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--output-file", type=Path,
                        help="Default: <input_stem>_prompt_enhanced.txt")
    parser.add_argument("--image-dir", type=Path, help="Default: <input_stem>_prepared beside input")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preprocess-only", action="store_true",
                        help="Prepare images and a baseline manifest without loading Qwen")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs from an earlier run")
    # load_inputs shares manifest parsing and path resolution with video inference.
    parser.set_defaults(prompt=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 128:
        parser.error("--max-new-tokens must be at least 128")
    if args.preprocess_only and args.input_type == "auto":
        parser.error("--preprocess-only requires --input-type human or robot")
    items = load_inputs(args, allow_directories=True)
    manifest = args.prompt_file.expanduser().resolve()
    output = (args.output_file or manifest.with_name(
        manifest.stem + "_prompt_enhanced.txt")).expanduser().resolve()
    image_dir = (args.image_dir or manifest.with_name(
        manifest.stem + "_prepared")).expanduser().resolve()
    baseline = output.with_name(output.stem.removesuffix("_prompt_enhanced") + "_preprocessed.txt")
    metadata_path = image_dir / "preparation.json"
    if len({manifest, output, baseline}) != 3:
        raise ValueError("Input, enhanced and baseline manifests must have distinct paths")
    for path in (image_dir, baseline, *([] if args.preprocess_only else [output])):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {path}; use --overwrite to regenerate")
    # Validate every source before loading a large model or writing outputs.
    for _, _, source in items:
        views, _ = load_views(source)
        if args.input_type != "auto":
            prepare_image(views, args.input_type)
    enhancer = None

    def get_enhancer():
        nonlocal enhancer
        if enhancer is None:
            enhancer = QwenEnhancer(
                args.qwen_model_dir.expanduser().resolve(), args.device, args.thinking,
                args.reasoning_effort, args.max_new_tokens,
            )
        return enhancer

    image_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"complete": False, "settings": {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }, "system_prompt": SYSTEM_PROMPT, "review_prompt": REVIEW_PROMPT,
        "output_format": "Left-side clause, while right-side clause. View suffix.", "samples": []}
    for line, instruction, source in items:
        views, paths = load_views(source)
        kind = args.input_type if args.input_type != "auto" else get_enhancer().classify(views[0])
        image, geometry = prepare_image(views, kind)
        destination = image_dir / f"{line:06d}.png"
        image.save(destination)
        metadata["samples"].append({
            "line": line, "source": str(source), "instruction": instruction,
            "views": [str(p) if p else None for p in paths],
            "image": str(destination), **geometry,
        })
    # The baseline uses exactly the same image paths as the enhanced manifest.
    baseline.write_text("".join(manifest_line(s["instruction"], Path(s["image"]), baseline)
                                for s in metadata["samples"]), encoding="utf-8")
    write_json(metadata_path, metadata)
    print(f"Prepared {len(items)} images at 416x240; baseline: {baseline}", flush=True)
    if not args.preprocess_only:
        from PIL import Image

        model = get_enhancer()
        partial = output.with_suffix(output.suffix + ".partial")
        with partial.open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(metadata["samples"], 1):
                with Image.open(sample["image"]) as image:
                    prompt, details = model.enhance(
                        image.convert("RGB"), sample["instruction"], sample["input_type"],
                        sample["missing_views"],
                    )
                sample.update(enhanced_prompt=prompt, enhancement=details)
                handle.write(manifest_line(prompt, Path(sample["image"]), output))
                handle.flush()
                write_json(metadata_path, metadata)
                print(f"[{index}/{len(items)}] {sample['source']}\n{prompt}", flush=True)
        partial.replace(output)
        print(f"Saved enhanced prompts: {output}", flush=True)
    metadata["complete"] = True
    write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
