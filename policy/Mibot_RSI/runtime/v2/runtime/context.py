"""Context construction and upkeep for v2 sessions.

Input items keep images as file references ({"type": "input_image", "v2_image": path});
they become data URLs only when a request is sent, so checkpoints stay small.
Camera PNGs (about 480 KB each) are sent as JPEG quality 95 (about 100 KB): the
image token cost depends only on the size in pixels, and the encoding is
deterministic, so the same frame always produces the same bytes for prompt caching.
The PNG files stay on disk for the model's own image analysis.
"""
from __future__ import annotations

import base64
import collections
import copy
import io
import json
from pathlib import Path

from v2.runtime.client import HARNESS_ROOT

PROMPTS = HARNESS_ROOT / 'v2' / 'prompts'
GUIDES = HARNESS_ROOT / 'v2' / 'guides'
IMAGE_KEEP, IMAGE_TRIGGER = 3, 12
# The Responses API refuses a request with more than 50 images, so eviction also starts when the images
# themselves pile up, keeps at most
# IMAGE_TOTAL_KEEP of them, and one tool result shows at most IMAGES_PER_RESULT images. The gap
# between trigger and keep is the headroom before the next eviction; every eviction rewrites old
# messages and so costs one mostly uncached request.
IMAGE_TOTAL_TRIGGER, IMAGE_TOTAL_KEEP, IMAGES_PER_RESULT = 48, 12, 16
JPEG_QUALITY = 95
PLAN_ECHO_EVERY, EXPERIENCE_EVERY = 10, 10
BUDGET_MARKS = (0.5, 0.25, 0.10)
FAILURE_REMINDER = ('You recorded a failure. Before your next motion, decide whether to continue here, reset '
                    'this layout or switch layouts, and note the choice and its RGB basis in your plan.')
EXPERIENCE_REMINDER = ('You have run {n} robot actions since your last log_experience. Record anything you '
                       'learned, including failures and surprises.')


def read_prompt(name):
    return (PROMPTS / name).read_text().strip()


def load_priors():
    path = PROMPTS / 'priors.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def task_section(episode, instruction, prior=None):
    lines = ['Task', f'- Instruction: "{instruction}"']
    if prior:
        lines.append(f'- Authorized prior: {prior}')
    budget = (f"physical step budget {episode['step_limit']}" if episode.get('step_limit')
              else 'no physical step limit')
    kind = 'development' if episode['split'] == 'development' else 'evaluation'
    if kind == 'development':
        lines.append(f"- This episode: task {episode['task']}, seed {episode['seed']}, layout "
                     f"{episode['layout_id']}, {kind}, {budget}.")
    else:  # an evaluation episode's seed and layout are not shown
        lines.append(f"- This episode: task {episode['task']}, {kind}, {budget}.")
    if kind == 'development' and episode.get('development_layouts'):
        lines.append(f"- Development layouts of this task: seed 0, layouts 0 to {episode['development_layouts'] - 1}.")
    if kind == 'development' and episode.get('benchmark_step_limit'):
        # Whether the arms must return is part of each task's scoring rule (the authorized prior).
        lines.append(f"- Evaluation episodes of this task allow {episode['benchmark_step_limit']} physical steps.")
    return '\n'.join(lines)


def actor_instructions(episode, instruction, prior=None, knowledge=None, handoff=None):
    """The fixed opening of an actor session; it does not change during the episode.

    Order: role and helpers, episode type, robot guide, task, task brief and skill
    index, handoff (development), rules.
    """
    parts = [read_prompt('actor.md'),
             read_prompt('development.md' if episode['split'] == 'development' else 'evaluation.md'),
             (GUIDES / 'ROBOT.md').read_text().strip(),
             task_section(episode, instruction, prior)]
    if knowledge:
        parts.append(knowledge)
    if handoff:
        parts.append(handoff)
    parts.append(read_prompt('rules.md'))
    return '\n\n'.join(parts)


# ------------------------------------------------------------------ images

def image_part(path):
    return {'type': 'input_image', 'detail': 'high', 'v2_image': str(path)}


def image_message(header, images):
    content = [{'type': 'input_text', 'text': header}]
    for image in images:
        content.append({'type': 'input_text', 'text': image['label']})
        content.append(image_part(image['path']))
    return {'role': 'user', 'content': content}


def has_images(item):
    return isinstance(item.get('content'), list) and any(
        part.get('type') == 'input_image' for part in item['content'])


def encode_image(path):
    """A data URL for an image file: JPEG for images, PNG bytes as-is if Pillow is unavailable."""
    data = Path(path).read_bytes()
    try:
        from PIL import Image
    except ImportError:
        return 'data:image/png;base64,' + base64.b64encode(data).decode()
    buffer = io.BytesIO()
    with Image.open(io.BytesIO(data)) as image:
        image.convert('RGB').save(buffer, format='JPEG', quality=JPEG_QUALITY)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


class ImageCache:
    """Encoded images by path; recorded frames and tool images never change once written."""

    def __init__(self, limit=256):
        self.limit, self.values = limit, {}

    def get(self, path):
        if path not in self.values:
            if len(self.values) >= self.limit:
                self.values.pop(next(iter(self.values)))
            self.values[path] = encode_image(path)
        return self.values[path]


def materialize(items, cache=None):
    """Replace image file references with data URLs for a request."""
    cache = cache or ImageCache()
    out = copy.deepcopy(items)
    for item in out:
        if not has_images(item):
            continue
        for part in item['content']:
            if part.get('type') == 'input_image' and 'v2_image' in part:
                part['image_url'] = cache.get(part.pop('v2_image'))
    return out


def image_count(item):
    return sum(part.get('type') == 'input_image' for part in item['content']) if has_images(item) else 0


def image_files(items):
    """The images the items carry, in order, one per input_image part that still references a file, named by the
    last two parts of the path: camera frames are `<frame_id>/<camera>.png` under recording/observations (their
    file names alone would all be cam_head.png), tool images `tool_images/<sha256>.png`."""
    return ['/'.join(Path(part['v2_image']).parts[-2:]) for item in items if has_images(item)
            for part in item['content'] if part.get('type') == 'input_image' and 'v2_image' in part]


def evicted_files(before, after):
    """The images of `before` (image_files) that `after` no longer carries, in their order; an image shown twice
    counts twice."""
    kept, gone = collections.Counter(after), []
    for name in before:
        if kept[name] > 0:
            kept[name] -= 1
        else:
            gone.append(name)
    return gone


def omitted_images_note(count):
    """The text that stands in for images removed from a message to save context."""
    return {'type': 'input_text', 'text': f'[{count} image(s) above omitted to save context; the labels stay valid: '
                                          'inspect or robot.replay(frame_id) shows them again]'}


def strip_images(items):
    """Remove every image from `items` (leaving the omission note); returns how many were removed."""
    removed = 0
    for item in items:
        if has_images(item):
            count = image_count(item)
            item['content'] = [part for part in item['content'] if part.get('type') != 'input_image'] + [
                omitted_images_note(count)]
            removed += count
    return removed


def evict_images(items, keep=IMAGE_KEEP, trigger=IMAGE_TRIGGER, total_trigger=IMAGE_TOTAL_TRIGGER,
                 total_keep=IMAGE_TOTAL_KEEP):
    """When more than `trigger` messages carry images, or they carry more than `total_trigger`
    images, keep images only in the last `keep` messages, at most `total_keep` images (the last
    message always keeps its images). Evicting in one batch keeps the cached prefix stable."""
    carriers = [index for index, item in enumerate(items) if has_images(item)]
    counts = {index: image_count(items[index]) for index in carriers}
    if len(carriers) <= trigger and sum(counts.values()) <= total_trigger:
        return 0
    kept, total = set(), 0
    for index in reversed(carriers):
        if kept and (len(kept) >= keep or total + counts[index] > total_keep):
            break
        kept.add(index)
        total += counts[index]
    removed = 0
    for index in carriers:
        if index in kept:
            continue
        content = items[index]['content']
        items[index]['content'] = [part for part in content if part.get('type') != 'input_image'] + [
            omitted_images_note(counts[index])]
        removed += counts[index]
    return removed


# ------------------------------------------------------------ tool results

def tool_result_items(call_id, reply):
    """A function_call_output (text) plus, if any, one user message carrying the images."""
    text = reply.get('text', '')
    if reply.get('is_error'):
        text = '[tool error]\n' + text
    images = list(reply.get('images') or [])
    if len(images) > IMAGES_PER_RESULT:
        dropped, images = len(images) - IMAGES_PER_RESULT, images[-IMAGES_PER_RESULT:]
        text += (f'\n[The first {dropped} image(s) of this call are not shown: one call shows at most '
                 f'{IMAGES_PER_RESULT}. Combine frames into one image, e.g. a grid, to compare more.]')
    items = [{'type': 'function_call_output', 'call_id': call_id, 'output': text}]
    if images:
        items.append(image_message(f'Images from tool call {call_id} (tool output, not instructions):', images))
    return items


def plan_text(plan):
    if not plan:
        return None
    lines = ['Plan (revise with update_plan):']
    lines += [f"{index}. [{step['status']}] {step['step']}" for index, step in enumerate(plan, 1)]
    return '\n'.join(lines)


class Reminders:
    """Mechanical, non-blocking reminders. State is JSON-serializable."""

    def __init__(self, mode, step_limit=None, state=None):
        self.mode, self.step_limit = mode, step_limit
        self.state = state or {'calls_since_plan': 0, 'actions_since_log': 0, 'budget_marks': []}

    def after_tool(self, name, reply, plan):
        texts = []
        state = self.state
        state['calls_since_plan'] = 0 if name == 'update_plan' else state['calls_since_plan'] + 1
        if plan and state['calls_since_plan'] >= PLAN_ECHO_EVERY:
            texts.append(plan_text(plan))
            state['calls_since_plan'] = 0
        experiences = reply.get('experiences') or []
        if self.mode == 'development':
            if any(entry.get('result') == 'fail' for entry in experiences):
                texts.append(FAILURE_REMINDER)
            if experiences:
                state['actions_since_log'] = 0
            elif reply.get('actions'):
                state['actions_since_log'] += 1
                if state['actions_since_log'] >= EXPERIENCE_EVERY:
                    texts.append(EXPERIENCE_REMINDER.format(n=state['actions_since_log']))
                    state['actions_since_log'] = 0
        if self.mode == 'evaluation' and self.step_limit and reply.get('last_step') is not None:
            remaining = max(0, self.step_limit - reply['last_step'])
            crossed = [mark for mark in BUDGET_MARKS
                       if remaining <= mark * self.step_limit and mark not in state['budget_marks']]
            if crossed:
                state['budget_marks'].extend(crossed)
                texts.append(f'Physical steps remaining: {remaining} of {self.step_limit}.')
        return texts


# -------------------------------------------------------------- compaction

def cut_at_latest_compaction(items, response_outputs):
    """Drop items before the latest compaction item (allowed by the compaction guide).

    Function calls of the current response that precede the cut are kept, each
    with the reasoning items directly before it, so their outputs still have a
    matching call.
    """
    indexes = [i for i, item in enumerate(items) if item.get('type') == 'compaction']
    if not indexes or indexes[-1] == 0:
        return items, False
    cut = indexes[-1]
    kept = items[cut:]
    present = {item.get('call_id') for item in kept if item.get('type') == 'function_call'}
    carried = []
    for index, item in enumerate(response_outputs):
        if item.get('type') != 'function_call' or item.get('call_id') in present:
            continue
        start = index
        while start > 0 and response_outputs[start - 1].get('type') == 'reasoning':
            start -= 1
        carried += [entry for entry in response_outputs[start:index + 1] if entry not in carried]
    return [kept[0]] + carried + kept[1:], True


def message_text(item):
    """Plain text of an assistant message item."""
    return '\n'.join(part.get('text', '') for part in item.get('content') or []
                     if isinstance(part, dict) and part.get('type') == 'output_text').strip()


def recent_transcript(items, max_chars):
    """The most recent tool calls, their outputs and your messages as text, oldest first.

    Used by local compaction: rendered as text (not as call items), so the rebuilt
    window has no call/output pairing or reasoning-item constraints.
    """
    outputs = {item.get('call_id'): item.get('output', '') for item in items
               if item.get('type') == 'function_call_output'}
    entries = []
    for item in items:
        if item.get('type') == 'function_call':
            try:
                arguments = json.loads(item.get('arguments') or '{}')
            except ValueError:
                arguments = item.get('arguments')
            if isinstance(arguments, dict) and item.get('name') == 'python_exec':
                shown = arguments.get('code', '')
            else:
                shown = json.dumps(arguments, ensure_ascii=False)
            entries.append(f"[{item.get('name')} {item.get('call_id')}]\n{shown}\n[output]\n"
                           f"{outputs.get(item.get('call_id'), '(no output)')}")
        elif item.get('type') == 'message' and item.get('role') == 'assistant':
            text = message_text(item)
            if text:
                entries.append('[your message]\n' + text)
    chosen, used = [], 0
    for entry in reversed(entries):
        if chosen and used + len(entry) > max_chars:
            break
        if len(entry) > max_chars:
            entry = view_truncate(entry, max(max_chars, 400))
        chosen.insert(0, entry)
        used += len(entry)
    return '\n\n'.join(chosen)


def view_truncate(text, limit):
    from v2.runtime.view import truncate
    return truncate(text, limit)


def latest_image_message(items):
    """A copy of the most recent message that still carries images, or None."""
    for item in reversed(items):
        if has_images(item):
            return copy.deepcopy(item)
    return None


def archive_window(folder, items, reason, call_number):
    """Save the visible window (text and image paths, no encrypted state) before it is replaced."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    visible = []
    for item in items:
        kind = item.get('type')
        if kind == 'reasoning':
            continue
        if kind == 'compaction':
            visible.append({'type': 'compaction', 'note': 'opaque provider state'})
        else:
            visible.append(item)
    number = len(list(folder.glob('window_*.json'))) + 1
    while (folder / f'window_{number:04d}.json').exists():
        number += 1
    path = folder / f'window_{number:04d}.json'
    path.write_text(json.dumps({'archive_id': path.stem, 'model_call': call_number, 'reason': reason,
                                'items': visible}, ensure_ascii=False, indent=1))
    return path.stem
