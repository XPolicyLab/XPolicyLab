"""Checkpoint-owned image roles, temporal memory and Qwen prompts."""

from PIL import Image


CURRENT_TYPES = (1, 2, 3)
INITIAL_TYPES = (19, 20, 21)
HISTORY_TYPES = ((12, 22, 23), (13, 24, 25), (14, 26, 27))
CAMERA_NAMES = ("head", "left", "right")
MEMORY_CAMERA = {kind: camera for group in (INITIAL_TYPES, *HISTORY_TYPES)
                 for camera, kind in enumerate(group)}


def selected_views(value):
    views = ["head"] if value is None else value
    if (not isinstance(views, (list, tuple)) or "head" not in views or
            len(set(views)) != len(views) or any(v not in CAMERA_NAMES for v in views)):
        raise ValueError("Memory views must contain head and optionally left/right, without duplicates")
    return tuple(i for i, name in enumerate(CAMERA_NAMES) if name in views)


class ImageMemory:
    def __init__(self, first, history):
        first, history = first or {}, history or {}
        views = selected_views(first.get("views"))
        self.first_views = views if first.get("enable", False) else ()
        self.first_t = first.get("t", 0) if self.first_views else 0
        if type(self.first_t) is not int and self.first_t != "relative":
            raise ValueError("first_frame.t must be an integer or relative")
        if self.first_views and first.get("mode", "prepend") != "prepend":
            raise ValueError("Only prepend first-frame memory is supported")
        self.prompt_version = first.get("prompt_version", "auto")
        if self.prompt_version == "auto":
            self.prompt_version = "legacy" if views == (0,) else "multiview"
        if self.prompt_version not in ("legacy", "multiview"):
            raise ValueError("Unsupported first_frame.prompt_version")
        self.history_views = selected_views(history.get("views")) if history.get("enable", False) else ()
        offsets = history.get("offsets") if self.history_views else ()
        if offsets is None:
            n, step = history.get("n", 1), history.get("step", 25)
            if type(n) is not int or not 1 <= n <= 3 or type(step) is not int or step < 1:
                raise ValueError("History requires n in [1, 3] and a positive integer step")
            offsets = [-i * step for i in range(n, 0, -1)]
        self.offsets = tuple(offsets)
        if len(self.offsets) > 3 or any(type(n) is not int or n >= 0 for n in self.offsets):
            raise ValueError("Expected at most three negative native-frame history offsets")
        if self.prompt_version == "legacy" and any(i != 0 for i in self.first_views + self.history_views):
            raise ValueError("Wrist image memory requires the multiview prompt")
        self.history_length = 1 - min(self.offsets, default=0)

    def assemble(self, first_images, history):
        images = [first_images[i] for i in self.first_views]
        kinds = [INITIAL_TYPES[i] for i in self.first_views]
        # Checkpoints list history selectors oldest first; the last is HISTORY_0.
        for slot, offset in zip(reversed(range(len(self.offsets))), self.offsets):
            frame = history[max(0, len(history) - 1 + offset)]
            images.extend(frame[i] for i in self.history_views)
            kinds.extend(HISTORY_TYPES[slot][i] for i in self.history_views)
        return images + list(history[-1]), kinds + list(CURRENT_TYPES)

    def messages(self, images, kinds, instruction):
        multiview = self.prompt_version == "multiview"
        names = {1: "the first-person observation", 2: "the left-wrist close-up", 3: "the right-wrist close-up",
                 19: "the first-person observation at the episode start",
                 **{i: "the historical first-person observation" for i in (12, 13, 14)}}
        if multiview:
            for group in (INITIAL_TYPES, *HISTORY_TYPES):
                when = "at the first observation" if group == INITIAL_TYPES else "from about one second ago"
                for kind, camera in zip(group, ("head-camera view", "left-wrist close-up", "right-wrist close-up")):
                    names[kind] = f"the {camera} {when}"
        temporal = ""
        if multiview and any(kind in MEMORY_CAMERA for kind in kinds):
            temporal = (
                "Image labels identify the first, earlier and current observations when present. "
                "Earlier observations are ordered from oldest to newest. Within each group, views are ordered "
                "head camera, left wrist, right wrist. Compare the same camera across time to track "
                "object motion, grasp changes and task progress. Near the episode start, an earlier "
                "observation may repeat the first observation. Predict actions from the current state "
                "to carry out the task instruction; earlier views provide context, not the current pose. ")
        views = " followed by ".join(names[k] for k in kinds)
        system = temporal + (
            f"Describe the key features of {views} (objects, their locations and graspable surfaces, "
            "each robot end effector's current pose and contact state, scene layout), then explain how "
            "the user's instruction should move each robot end effector and modify its interaction with "
            "the scene. Predict the short-horizon trajectory of the robot end effectors that executes "
            "the instruction consistently with the observation.")
        content = []
        labeled = bool(self.first_views) or (multiview and any(k in MEMORY_CAMERA for k in kinds))
        for rgb, kind in zip(images, kinds):
            label = None
            if labeled and multiview:
                camera = ("Head-camera", "Left-wrist camera", "Right-wrist camera")[
                    MEMORY_CAMERA[kind] if kind in MEMORY_CAMERA else CURRENT_TYPES.index(kind)]
                if kind in INITIAL_TYPES:
                    label = f"{camera} view at the first observation"
                elif kind in MEMORY_CAMERA:
                    label = f"{camera} view from about one second ago"
                else:
                    label = f"Current {camera.lower()} view"
            elif labeled:
                label = {19: "Head-camera view at the first observation", 1: "Current head and wrist views",
                         **{k: "Head-camera view from about one second ago" for k in (12, 13, 14)}}.get(kind)
            if label is not None:
                content.append({"type": "text", "text": f"\n{label}:\n"})
            content.append({"type": "image", "image": Image.fromarray(rgb)})
        if multiview and labeled:
            content.append({"type": "text", "text": "\nTask instruction for the current observation:\n"})
        content.append({"type": "text", "text": instruction})
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]
