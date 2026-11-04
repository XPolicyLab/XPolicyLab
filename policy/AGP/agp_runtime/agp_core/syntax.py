"""Short command strings for `run` steps and measurement queries.

A string is parsed into the same step or query object the JSON form uses, so every
action still goes through runner.expand_arm and the unchanged runtime checks.

Arm command:   <L|R|B> <verb> [target] [options] [key=evidence text ...]
               verbs: ee <x,y,z | label | label+dx,dy,dz>, d <dx,dy,dz>, open, close, home,
               hold [samples], joints <j1,...,j6>; width <mm> with OURS_GRIP_WIDTH (see below)
Step:          arm commands joined with "&" (shared clock); B = same command on both arms
Measure step:  ? [label:] <L|R> <top|lw|rw> <u,v>[/<u,v>...] [z=..] [pick=..] [zr=a,b] <target text>
               several queries joined with "&"
OURS_INTENT_MANIPULATE: with manipulate=True (the runner passes the switch) the intent word `manipulate` is accepted
as well (INTENTS_MANIPULATE); with it False (the default) the word is unknown.
OURS_WAIT_STILL: with still=True (the runner passes the switch) the option word `still` is accepted on `hold`
(`B hold 300 still`: spec "still": True; the samples are then the most the runner holds, see agp_core/runner.py); on any
other verb it is an error. With it False (the default) the word is unknown.
OURS_GRIP_WIDTH: with width=True (the runner passes the switch) the verb `width <mm>` is accepted (`L width 30`: spec
{"do": "width", "width_mm": 30.0}; the pad gap in mm, agp_core/grip_width.py WIDTH_MIN_MM .. WIDTH_MAX_MM); with it False
(the default) the verb is unknown.
"""
import re

ARMS = {"l": ("left",), "left": ("left",), "r": ("right",), "right": ("right",),
        "b": ("left", "right"), "both": ("left", "right")}
VERBS = {"ee": "ee", "move": "ee", "d": "delta", "delta": "delta", "open": "open", "close": "close",
         "home": "home", "hold": "hold", "joints": "joints"}
INTENTS = {"obs": "observe", "observe": "observe", "grasp": "grasp", "transfer": "transfer", "xfer": "transfer",
           "contact": "contact", "retreat": "retreat", "place": "place"}
INTENTS_MANIPULATE = dict(INTENTS, manipulate="manipulate")     # OURS_INTENT_MANIPULATE
SPEEDS = {"fast": "transit", "transit": "transit", "slow": "precision", "precision": "precision"}
CAMS = {"top": "top", "lw": "left_wrist", "left_wrist": "left_wrist", "rw": "right_wrist", "right_wrist": "right_wrist"}
OPTIONS = {"m": "m", "max": "max_samples", "settle": "settle_samples", "ptol": "position_tolerance_m",
           "rtol": "rotation_tolerance_rad", "drot": "drot_deg"}
PICKS = {"solid", "high", "low"}
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_KEY = re.compile(r"(?<!\S)([a-z_]+)=")
_LABEL = r"[A-Za-z_][\w.]*"


def numbers(token, count, what):
    parts = token.split(",")
    if len(parts) != count or not all(re.fullmatch(_NUM, p) for p in parts):
        raise ValueError(f"'{token}': {what} needs {count} comma-separated numbers")
    return [float(p) for p in parts]


def _target(token):
    if re.fullmatch(rf"{_NUM},{_NUM},{_NUM}", token):
        return numbers(token, 3, "a position")
    match = re.fullmatch(rf"({_LABEL})(?:\+(.+))?", token)
    if not match:
        raise ValueError(f"'{token}': target is x,y,z or a label with an optional +dx,dy,dz")
    label, offset = match.groups()
    return {"at": label, "d": numbers(offset, 3, "an offset") if offset else [0.0, 0.0, 0.0]}


def _split_evidence(text):
    """Head (verb, target, options) and evidence pairs; evidence starts at the first non-option key=."""
    for match in _KEY.finditer(text):
        if match.group(1) not in OPTIONS:
            head, tail = text[:match.start()], text[match.start():]
            keys = list(_KEY.finditer(tail))
            evidence = {}
            for i, key in enumerate(keys):
                end = keys[i + 1].start() if i + 1 < len(keys) else len(tail)
                value = tail[key.end():end].strip()
                if not value:
                    raise ValueError(f"evidence {key.group(1)}= needs text")
                evidence[key.group(1)] = value
            return head, evidence
    return text, {}


def parse_arm(text, manipulate=False, still=False, width=False):
    """Return (arm names, arm spec, expect) for one arm command (manipulate: the intent word `manipulate` too; still:
    the hold option `still`, OURS_WAIT_STILL; width: the verb `width <mm>`, OURS_GRIP_WIDTH)."""
    intents = INTENTS_MANIPULATE if manipulate else INTENTS
    head, evidence = _split_evidence(text.strip())
    tokens = head.split()
    if len(tokens) < 2:
        raise ValueError(f"'{text.strip()}': expected '<L|R|B> <verb> ...'")
    arms = ARMS.get(tokens[0].lower())
    verb = VERBS.get(tokens[1].lower())
    if verb is None and width and tokens[1].lower() == "width":     # grip width (OURS_GRIP_WIDTH)
        verb = "width"
    if arms is None or verb is None:
        if width:
            raise ValueError(f"'{text.strip()}': starts with L, R or B and a verb (ee, d, open, close, width, home, "
                             f"hold, joints)")
        raise ValueError(f"'{text.strip()}': starts with L, R or B and a verb (ee, d, open, close, home, hold, joints)")
    spec, expect, rest = {"do": verb}, {}, tokens[2:]
    need = {"ee": "a target", "delta": "dx,dy,dz", "joints": "six joint angles",
            "width": "the pad gap in mm"}.get(verb)
    if need and not rest:
        raise ValueError(f"'{text.strip()}': {verb} needs {need}")
    if verb == "width":                                               # grip width: the pad gap in mm
        from agp_core import grip_width
        spec["width_mm"] = numbers(rest.pop(0), 1, "width")[0]
        try:
            grip_width.width_mm(spec)
        except ValueError as exc:
            raise ValueError(f"'{text.strip()}': {exc}") from None
    elif verb == "ee":
        spec["to"] = _target(rest.pop(0))
        if isinstance(spec["to"], dict) and len(arms) == 2:
            raise ValueError(f"'{text.strip()}': a measurement belongs to one arm; use L and R, not B")
    elif verb == "delta":
        spec["d"] = numbers(rest.pop(0), 3, "d")
    elif verb == "joints":
        spec["joints"] = numbers(rest.pop(0), 6, "joints")
    elif verb == "hold" and rest and re.fullmatch(r"\d+", rest[0]):
        spec["samples"] = int(rest.pop(0))
    for token in rest:
        low = token.lower()
        if low in intents:
            spec["intent"] = intents[low]
        elif low in SPEEDS:
            spec["speed"] = SPEEDS[low]
        elif low in ("plan", "linear"):
            spec["mode"] = low
        elif low == "recovery":
            evidence.setdefault("release_purpose", "recovery")
        elif still and low == "still":                       # OURS_WAIT_STILL
            if verb != "hold":
                raise ValueError(f"'{text.strip()}': still is an option of hold only")
            spec["still"] = True
        elif match := re.fullmatch(rf"down({_NUM})", low):
            spec["rot"] = {"down": float(match.group(1))}
        elif match := re.fullmatch(rf"tilt({_NUM},{_NUM})", low):
            spec["rot"] = {"tilted": numbers(match.group(1), 2, "tilt")}
        elif match := re.fullmatch(rf"q({_NUM},{_NUM},{_NUM},{_NUM})", low):
            spec["rot"] = dict(zip("wxyz", numbers(match.group(1), 4, "q")))
        elif match := re.fullmatch(rf"gap([<>])({_NUM})", low):
            expect["gap_min_mm" if match.group(1) == ">" else "gap_max_mm"] = float(match.group(2))
        elif "=" in token and token.split("=", 1)[0] in OPTIONS:
            key, value = token.split("=", 1)
            if key == "m":
                spec["m"] = value
            elif key == "drot":
                spec["drot_deg"] = numbers(value, 3, "drot")
            else:
                spec[OPTIONS[key]] = numbers(value, 1, key)[0]
                if key in ("max", "settle"):
                    spec[OPTIONS[key]] = int(spec[OPTIONS[key]])
        else:
            raise ValueError(f"'{text.strip()}': unknown word '{token}' (evidence goes last as key=text)")
    if evidence:
        spec["ev"] = evidence
    return arms, spec, expect


def parse_query(text, pose=False):
    """Return (query object, label or None) for one measurement query string (pose: the option `pose`,
    OURS_POSE_QUERY; when False the word is target text)."""
    text = text.strip()
    label = None
    match = re.match(rf"({_LABEL}):\s*(.*)$", text, re.S)
    if match:
        label, text = match.groups()
    tokens = text.split()
    if len(tokens) < 4:
        raise ValueError(f"'{text}': expected '<L|R> <top|lw|rw> <u,v>[/<u,v>...] [options] <target text>'")
    arms = ARMS.get(tokens[0].lower())
    if arms is None or len(arms) != 1:
        raise ValueError(f"'{text}': a measurement names one arm, L or R")
    cam = CAMS.get(tokens[1].lower())
    if cam is None:
        raise ValueError(f"'{text}': camera is top, lw or rw")
    pixels = []
    for pixel in tokens[2].split("/"):
        if not re.fullmatch(r"\d+,\d+", pixel):
            raise ValueError(f"'{tokens[2]}': pixels are u,v integers, candidates separated by /")
        pixels.append([int(v) for v in pixel.split(",")])
    query = {"arm": arms[0], "cam": cam}
    if len(pixels) == 1:
        query.update(u=pixels[0][0], v=pixels[0][1])
    else:
        query["cands"] = pixels
    rest = tokens[3:]
    while rest:
        token = rest[0].lower()
        if match := re.fullmatch(rf"z=({_NUM})", token):
            query["plane_z"] = float(match.group(1))
        elif match := re.fullmatch(r"pick=(\w+)", token):
            if match.group(1) not in PICKS:
                raise ValueError(f"'{rest[0]}': pick is solid, high or low")
            query["pick"] = match.group(1)
        elif match := re.fullmatch(rf"zr=({_NUM},{_NUM})", token):
            query["zr"] = numbers(match.group(1), 2, "zr")
        elif pose and token == "pose":           # pose query (OURS_POSE_QUERY)
            query["pose"] = True
        else:
            break
        rest.pop(0)
    if not rest:
        raise ValueError(f"'{text}': add a few words naming the target")
    query["target"] = " ".join(rest)
    return query, label


def parse_step(text, manipulate=False, still=False, width=False, pose=False):
    """One step string -> the JSON step object (manipulate: the intent word `manipulate` too,
    OURS_INTENT_MANIPULATE; still: the hold option `still`, OURS_WAIT_STILL; width: the verb `width <mm>`,
    OURS_GRIP_WIDTH; pose: the query option `pose`, OURS_POSE_QUERY)."""
    text = text.strip()
    if text.startswith("?"):
        queries, labels = [], []
        for part in text[1:].split("&"):
            query, label = parse_query(part, pose=pose)
            queries.append(query)
            labels.append(label)
        return {"measure": queries, "as": labels}
    step = {}
    for part in text.split("&"):
        arms, spec, expect = parse_arm(part, manipulate, still, width)
        for arm in arms:
            if arm in step:
                raise ValueError(f"'{text}': the {arm} arm appears twice in one step")
            step[arm] = dict(spec)
            if expect:
                step.setdefault("expect", {})[arm] = dict(expect)
    return step


def parse_show(show):
    """show as {at, end, on_stop}, a list of end images, or a string like 'top lw 2:rw stop:top'."""
    if show is None or isinstance(show, dict):
        return show or {}
    tokens = show if isinstance(show, list) else str(show).replace(",", " ").split()
    out = {}
    for token in tokens:
        where, _, name = token.rpartition(":")
        name = CAMS.get(name.lower())
        if name is None:
            raise ValueError(f"show: unknown image '{token}' (top, lw, rw)")
        if not where:
            out.setdefault("end", []).append(name)
        elif where == "stop":
            out.setdefault("on_stop", []).append(name)
        elif where.isdigit():
            out.setdefault("at", {}).setdefault(where, []).append(name)
        else:
            raise ValueError(f"show: '{token}' is image, k:image or stop:image")
    return out
