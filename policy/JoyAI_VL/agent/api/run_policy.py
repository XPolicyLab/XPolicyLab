"""Run the model's tool conversation over Preview's robot client."""

import base64
from http.client import HTTPException
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

from PIL import Image

import robo

TOOLS = [
    {"type": "function", "function": {
        "name": "python",
        "description": ("Run Python code in your persistent session (`robot`, `np` and `math` are defined; "
                        "variables stay between calls). Returns what the code printed, the value of a final "
                        "expression, and any error. Pictures from robot.look() follow the result."),
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to run."}},
            "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": ("End the episode now; the scene is judged as it is. The episode ends by itself the moment "
                        "the task is complete, so only call this when nothing you can still do would raise the "
                        "score."),
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["no_budget", "stuck", "keep_partial"],
                       "description": ("no_budget: what is left cannot cover another attempt; stuck: no way "
                                       "forward; keep_partial: more attempts would risk the partial result.")},
            "summary": {"type": "string", "description": "What was achieved and what is missing."}},
            "required": ["status", "summary"]}}},
]

CAMERAS = {"head.png": "head", "wrist_l.png": "left_wrist", "wrist_r.png": "right_wrist"}
MAX_IMAGES_PER_TURN = 8
MAX_HISTORY_IMAGES = 90
KEEP_HISTORY_IMAGES = 60
MAX_HISTORY_IMAGE_BYTES = 20_000_000


def event(kind, **fields):
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), flush=True)


def image_parts(images, first_number):
    parts = []
    for number, image in enumerate(images, first_number):
        encoded = BytesIO()
        with Image.open(image["path"]) as picture:
            picture.convert("RGB").save(encoded, "JPEG", quality=90)
        data = base64.b64encode(encoded.getvalue()).decode("ascii")
        parts.extend([{"type": "text", "text": f"[picture {number}: {image['caption']}]"},
                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}])
    return parts


def trim_images(messages):
    # The initial camera views remain in messages[1].
    pictures = [(previous["text"].split(":", 1)[0], part) for message in messages[2:]
                if message["role"] == "user" and isinstance(message.get("content"), list)
                for previous, part in zip(message["content"], message["content"][1:])
                if part["type"] == "image_url"]
    initial_pictures = [part for part in messages[1]["content"] if part["type"] == "image_url"]
    count = len(initial_pictures) + len(pictures)
    size = sum(len(part["image_url"]["url"]) for part in initial_pictures)
    size += sum(len(part["image_url"]["url"]) for _, part in pictures)
    if count <= MAX_HISTORY_IMAGES and size <= MAX_HISTORY_IMAGE_BYTES:
        return
    for caption_prefix, part in pictures:
        if count <= KEEP_HISTORY_IMAGES and size <= MAX_HISTORY_IMAGE_BYTES * 0.7:
            break
        count -= 1
        size -= len(part["image_url"]["url"])
        part.clear()
        part.update(type="text", text=f"{caption_prefix} removed to keep the conversation small; call look() again for a current one]")


def streamed_message(response):
    """Assemble the first chat completion from its SSE deltas."""
    parts = []
    calls = {}
    finished = False
    message = {"role": "assistant", "content": "", "reasoning_content": ""}
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            parts.append(line[5:].lstrip(" "))
        if line or not parts:
            continue
        data = "\n".join(parts)
        parts.clear()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise OSError("model gateway returned a streaming error")
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            message["content"] += delta.get("content") or ""
            message["reasoning_content"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for piece in delta.get("tool_calls") or []:
                index = piece.get("index")
                if index is None:
                    index = next((i for i, call in calls.items() if piece.get("id") and call["id"] == piece["id"]),
                                 len(calls) if piece.get("id") or not calls else max(calls))
                call = calls.setdefault(index, {"id": None, "type": "function",
                                                "function": {"name": "", "arguments": ""}})
                if piece.get("id"):
                    call["id"] = piece["id"]
                function = piece.get("function") or {}
                if function.get("name") and not call["function"]["name"]:
                    call["function"]["name"] = function["name"]
                call["function"]["arguments"] += function.get("arguments") or ""
            finished = finished or bool(choice.get("finish_reason"))
    if not finished:
        raise OSError("model stream ended before its finish reason")
    message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return message


def tool_calls(reply, turn):
    calls = []
    for index, call in enumerate(reply.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        calls.append({"id": call.get("id") or f"call_{turn}_{index}", "type": "function",
                      "function": {"name": str(function.get("name") or ""),
                                   "arguments": arguments if isinstance(arguments, str)
                                   else json.dumps(arguments or {})}})
    return calls


def cell_output(result, status, turn, max_turns):
    output = result["output"].strip("\n") or "(no output)"
    if len(output) > 16000:
        output = f"{output[:10000]}\n[... {len(output) - 16000} characters cut ...]\n{output[-6000:]}"
    step = f"step {status['step']} of {status['step_limit']} · " if status.get("step") is not None else ""
    output += f"\n— {step}reply {turn} of {max_turns}"
    if result.get("stopped") or status.get("over"):
        output += f"\nThe episode is over: {status.get('end') or 'the episode is over'}."
    return output


def model_reply(settings, messages):
    trim_images(messages)
    request = urllib.request.Request(os.environ["MODEL_API_URL"],
                                     data=json.dumps(dict(settings, messages=messages)).encode(),
                                     headers={"Content-Type": "application/json"})
    timeout = float(os.environ.get("HEXASHELL_REQUEST_TIMEOUT_S", "900"))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if "text/event-stream" in response.headers.get("Content-Type", ""):
                    return streamed_message(response)
                return json.load(response)["choices"][0]["message"]
        except urllib.error.HTTPError as error:
            if attempt == 2 or (error.code != 429 and error.code < 500):
                raise
        except (OSError, HTTPException):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def main(prompt):
    max_turns = int(os.environ.get("HEXASHELL_MAX_TURNS", "100"))
    settings = {"model": os.environ.get("MODEL", "gpt-6-astra"), "tools": TOOLS, "tool_choice": "auto",
                "reasoning_effort": "max",
                "max_tokens": int(os.environ.get("HEXASHELL_MAX_TOKENS", "65536")),
                "temperature": float(os.environ.get("HEXASHELL_TEMPERATURE", "0")),
                "stream": os.environ.get("HEXASHELL_STREAM", "1") == "1"}
    if settings["stream"]:
        settings["stream_options"] = {"include_usage": True}
    template = json.loads(os.environ.get("HEXASHELL_CHAT_TEMPLATE_KWARGS", "{}"))
    if template:
        if "reasoning_effort" in template:
            template["reasoning_effort"] = settings["reasoning_effort"]
        settings["chat_template_kwargs"] = template
    keep_reasoning = os.environ.get("HEXASHELL_HISTORY_REASONING", "1") == "1"

    robo.download_obs()
    images = [{"path": Path(robo.OBS_DIR) / file, "caption": f"{camera} at the start"}
              for file, camera in CAMERAS.items() if (Path(robo.OBS_DIR) / file).is_file()]
    prompt_files = sorted(Path(__file__).with_name("prompts").joinpath("robodojo").glob("*.md"))
    if not prompt_files:
        raise RuntimeError("RoboDojo system prompt files are missing")
    system = "\n\n".join(path.read_text(encoding="utf-8").strip() for path in prompt_files) + "\n"
    budget_minutes = float(os.environ.get("HEXASHELL_BUDGET_S", "5400")) / 60
    prompt += (f"\nYou have up to {max_turns} replies and {budget_minutes:.0f} minutes for this episode; "
               "the simulation stands still while you think.\n")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": prompt}] + image_parts(images, 1)}]
    next_picture = len(images) + 1

    nudges = 3
    end_reason = "turn_limit"
    for turn in range(1, max_turns + 1):
        reply = model_reply(settings, messages)
        calls = tool_calls(reply, turn)
        assistant = {"role": "assistant", "content": reply.get("content") or ""}
        if calls:
            assistant["tool_calls"] = calls
        reasoning = reply.get("reasoning_content") or reply.get("reasoning")
        if keep_reasoning and reasoning:
            assistant["reasoning_content" if template else "reasoning"] = reasoning
        messages.append(assistant)
        event("reply", turn=turn, calls=calls, content=reply.get("content"))
        if not calls:
            if nudges == 0:
                end_reason = "no_tool_call"
                break
            nudges -= 1
            messages.append({"role": "user", "content":
                             "The episode is still running, so the task is not complete yet (it ends by itself "
                             "the moment it is). Continue with the python tool; call finish only if nothing "
                             "you can still do would raise the score."})
            continue

        images = []
        finished = False
        state = {"over": False}
        for call in calls:
            function = call["function"]
            try:
                args = json.loads(function["arguments"])
            except (TypeError, ValueError):
                args = None
            name = function["name"]
            output = "Invalid tool arguments. Use the declared schema."
            if finished:
                output = "Not run: finish was called earlier in this reply."
            elif state.get("over"):
                output = f"Not run: The episode is over: {state['end']}."
            elif isinstance(args, dict):
                code = args.get("code")
                if name == "python" and isinstance(code, str) and code.strip():
                    result = robo.python_cell(code, structured=True)
                    state = robo.post({"cmd": "robot15_status"})
                    if "error" in state:
                        raise RuntimeError(state["error"])
                    output = cell_output(result, state, turn, max_turns)
                    images.extend(result.get("images", []))
                elif (name == "finish" and args.get("status") in ("no_budget", "stuck", "keep_partial")
                      and isinstance(args.get("summary"), str)):
                    event("finish", status=args["status"], summary=args["summary"])
                    finished = True
                    output = "Finishing: the episode ends now and the scene is judged as it stands."
            event("tool_result", call_id=call["id"], output=output)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        if finished:
            finish_episode()
            return
        state = robo.post({"cmd": "robot15_turn"})
        if state.get("over"):
            event("end", reason="episode_over")
            robo.post({"cmd": "robot15_end_seen"})
            return
        if "error" in state:
            raise RuntimeError(state["error"])
        if images:
            skipped = max(0, len(images) - MAX_IMAGES_PER_TURN)
            text = "Pictures from robot.look() in this turn"
            if skipped:
                text += f" (the first {skipped} were left out: at most {MAX_IMAGES_PER_TURN} per turn)"
            content = [{"type": "text", "text": text + ":"}]
            messages.append({"role": "user", "content": content + image_parts(images[skipped:], next_picture + skipped)})
            next_picture += len(images)

    event("end", reason=end_reason)
    finish_episode()


def finish_episode():
    result = robo.post({"cmd": "done"})
    if result.get("exit_code") not in (robo.EXIT_OK, robo.EXIT_EPISODE_OVER):
        raise RuntimeError(f"Preview rejected finish: {result.get('error', result)}")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except urllib.error.HTTPError as error:
        event("error", error=f"HTTP {error.code}: {error.reason}")
        raise SystemExit(1)
    except (OSError, HTTPException, RuntimeError, KeyError, IndexError, TypeError, ValueError) as error:
        event("error", error=str(error))
        raise SystemExit(1)
