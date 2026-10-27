import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


def _optional_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return Path(text).expanduser()


def _stable_pick(items: Sequence[str], seed: str) -> str:
    if not items:
        raise ValueError("Cannot select from an empty instruction list.")
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(items)
    return str(items[index])


@lru_cache(maxsize=None)
def _load_json(path: Path) -> Any:
    """Load immutable instruction metadata once per worker process."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=None)
def _object_description_exists(path: Path) -> bool:
    """Cache immutable metadata lookups, including ordinary non-object values."""
    return path.exists()


def _extract_placeholders(instruction: str) -> list[str]:
    return re.findall(r"{([^}]+)}", instruction)


def _filter_instruction_templates(
    instructions: Sequence[str], episode_params: dict[str, str]
) -> list[str]:
    filtered = []
    stripped_params = {key.strip("{}"): value for key, value in episode_params.items()}
    param_keys = set(stripped_params.keys())
    arm_params = {key for key in param_keys if len(key) == 1 and "a" <= key <= "z"}

    for instruction in instructions:
        placeholders = set(_extract_placeholders(str(instruction)))
        if placeholders == param_keys:
            filtered.append(str(instruction))
            continue
        if (
            arm_params
            and placeholders.union(arm_params) == param_keys
            and not arm_params.intersection(placeholders)
        ):
            filtered.append(str(instruction))
    return filtered


def _description_list_for_type(
    payload: dict[str, Any], instruction_type: str
) -> list[str]:
    return [str(item) for item in payload.get(instruction_type, [])]


def _replace_placeholders(
    instruction: str,
    episode_params: dict[str, str],
    *,
    objects_description_dir: Path | None,
    instruction_type: str,
    seed: str,
) -> str:
    stripped_params = {key.strip("{}"): value for key, value in episode_params.items()}
    result = str(instruction)

    for key, raw_value in stripped_params.items():
        value = str(raw_value)
        if objects_description_dir is not None:
            json_path = objects_description_dir / f"{value}.json"
            if _object_description_exists(json_path):
                descriptions = _description_list_for_type(
                    _load_json(json_path), instruction_type
                )
                if not descriptions:
                    raise ValueError(
                        f"Object description file has no `{instruction_type}` entries: {json_path}"
                    )
                value = f"the {_stable_pick(descriptions, f'{seed}|{key}|{value}')}"
            elif "/" in value or "\\" in value:
                raise FileNotFoundError(
                    f"Object description file not found: {json_path}"
                )

        if len(key) == 1 and "a" <= key <= "z":
            value = f"the {value} arm"

        result = result.replace("{" + key + "}", value)
    return result


def _instruction_types(instruction_type: str) -> list[str]:
    text = str(instruction_type).strip().lower()
    if text in {"all", "both"}:
        return ["seen", "unseen"]
    if text not in {"seen", "unseen"}:
        raise ValueError(
            f"Unsupported instruction_type={instruction_type!r}; expected seen, unseen, or all."
        )
    return [text]


def generate_instruction_candidates_from_scene_info(
    *,
    task_name: str,
    episode_params: dict[str, str],
    instruction_type: str,
    task_instruction_dir: Path,
    objects_description_dir: Path | None,
    max_instructions_per_episode: int | None = None,
) -> list[str]:
    task_instruction_path = task_instruction_dir / f"{task_name}.json"
    if not task_instruction_path.exists():
        raise FileNotFoundError(
            f"Task instruction file not found: {task_instruction_path}"
        )

    task_payload = _load_json(task_instruction_path)
    candidates: list[str] = []
    for kind in _instruction_types(instruction_type):
        templates = _filter_instruction_templates(
            task_payload.get(kind, []), episode_params
        )
        for template in templates:
            seed = f"{task_name}|{kind}|{template}|{sorted(episode_params.items())}"
            candidates.append(
                _replace_placeholders(
                    template,
                    episode_params,
                    objects_description_dir=objects_description_dir,
                    instruction_type=kind,
                    seed=seed,
                )
            )
            if max_instructions_per_episode is not None and len(candidates) >= int(
                max_instructions_per_episode
            ):
                return candidates
    return candidates


def _task_text(task_name: str, humanize_task_name: bool) -> str:
    return str(task_name).replace("_", " ") if humanize_task_name else str(task_name)


def _load_instruction_file(
    *,
    root_dir: Path,
    task_name: str,
    phase: str,
    episode_index: int,
    instruction_type: str,
    max_instructions_per_episode: int | None,
) -> list[str]:
    path = (
        root_dir / task_name / phase / "instructions" / f"episode{episode_index}.json"
    )
    if not path.exists():
        return []
    payload = _load_json(path)
    candidates: list[str] = []
    for kind in _instruction_types(instruction_type):
        candidates.extend(str(item) for item in payload.get(kind, []))
    if max_instructions_per_episode is not None:
        candidates = candidates[: int(max_instructions_per_episode)]
    return candidates


def _load_scene_info_candidates(
    *,
    root_dir: Path,
    task_name: str,
    phase: str,
    episode_index: int,
    instruction_type: str,
    task_instruction_dir: Path | None,
    objects_description_dir: Path | None,
    max_instructions_per_episode: int | None,
) -> list[str]:
    if task_instruction_dir is None:
        return []
    scene_info_path = root_dir / task_name / phase / "scene_info.json"
    if not scene_info_path.exists():
        return []
    scene_info = _load_json(scene_info_path)
    episode_payload = scene_info.get(f"episode_{episode_index}", {})
    episode_params = (
        episode_payload.get("info", {}) if isinstance(episode_payload, dict) else {}
    )
    if not isinstance(episode_params, dict):
        episode_params = {}
    return generate_instruction_candidates_from_scene_info(
        task_name=task_name,
        episode_params={str(key): str(value) for key, value in episode_params.items()},
        instruction_type=instruction_type,
        task_instruction_dir=task_instruction_dir,
        objects_description_dir=objects_description_dir,
        max_instructions_per_episode=max_instructions_per_episode,
    )


def load_raw_robotwin_instruction_candidates(
    *,
    root_dir: str | Path,
    task_name: str,
    phase: str,
    episode_index: int,
    instruction_source: str = "task",
    instruction_type: str = "seen",
    task_instruction_dir: str | Path | None = None,
    objects_description_dir: str | Path | None = None,
    humanize_task_name: bool = False,
    max_instructions_per_episode: int | None = None,
) -> list[str]:
    root = Path(root_dir).expanduser()
    source = str(instruction_source).strip().lower()
    if source in {"", "none", "null", "task", "task_name"}:
        return [_task_text(task_name, humanize_task_name)]

    task_dir = _optional_path(task_instruction_dir)
    objects_dir = _optional_path(objects_description_dir)
    candidates: list[str] = []

    if source in {"auto", "instructions", "instruction_file", "file"}:
        candidates = _load_instruction_file(
            root_dir=root,
            task_name=str(task_name),
            phase=str(phase),
            episode_index=int(episode_index),
            instruction_type=instruction_type,
            max_instructions_per_episode=max_instructions_per_episode,
        )
        if candidates or source != "auto":
            if candidates:
                return candidates
            raise FileNotFoundError(
                f"No instruction file found for {task_name}/{phase}/episode{episode_index} under {root}."
            )

    if source in {"auto", "scene_info", "generated", "robo_twin", "robotwin"}:
        candidates = _load_scene_info_candidates(
            root_dir=root,
            task_name=str(task_name),
            phase=str(phase),
            episode_index=int(episode_index),
            instruction_type=instruction_type,
            task_instruction_dir=task_dir,
            objects_description_dir=objects_dir,
            max_instructions_per_episode=max_instructions_per_episode,
        )
        if candidates or source != "auto":
            if candidates:
                return candidates
            raise FileNotFoundError(
                f"No scene-info instructions found for {task_name}/{phase}/episode{episode_index} under {root}."
            )

    if source == "auto":
        return [_task_text(task_name, humanize_task_name)]
    raise ValueError(f"Unsupported instruction_source={instruction_source!r}.")


def select_instruction_candidate(
    candidates: Sequence[str],
    *,
    instruction_selection: str,
    task_name: str,
    phase: str,
    episode_index: int,
    start_index: int = 0,
) -> str:
    selection = str(instruction_selection).strip().lower()
    if selection == "first":
        return str(candidates[0])
    if selection in {"hash", "episode_hash"}:
        seed = f"{task_name}|{phase}|{episode_index}"
        return _stable_pick(candidates, seed)
    if selection in {"sample_hash", "window_hash"}:
        seed = f"{task_name}|{phase}|{episode_index}|{start_index}"
        return _stable_pick(candidates, seed)
    raise ValueError(
        f"Unsupported instruction_selection={instruction_selection!r}; expected first, hash, or sample_hash."
    )


class RobotWinRawInstructionMixin:
    def _init_instruction_config(
        self,
        *,
        prompt_template: str,
        humanize_task_name: bool,
        instruction_source: str,
        instruction_type: str,
        instruction_selection: str,
        task_instruction_dir: str | Path | None,
        objects_description_dir: str | Path | None,
        max_instructions_per_episode: int | None,
    ) -> None:
        self.prompt_template = str(prompt_template)
        self.humanize_task_name = bool(humanize_task_name)
        self.instruction_source = str(instruction_source)
        self.instruction_type = str(instruction_type)
        self.instruction_selection = str(instruction_selection)
        self.task_instruction_dir = _optional_path(task_instruction_dir)
        self.objects_description_dir = _optional_path(objects_description_dir)
        self.max_instructions_per_episode = (
            None
            if max_instructions_per_episode is None
            else int(max_instructions_per_episode)
        )

    def _attach_instruction_candidates_to_episodes(self) -> None:
        for episode in self.episodes:
            episode["instruction_candidates"] = tuple(
                load_raw_robotwin_instruction_candidates(
                    root_dir=self.root_dir,
                    task_name=episode["task"],
                    phase=episode["phase"],
                    episode_index=int(episode["episode_index"]),
                    instruction_source=self.instruction_source,
                    instruction_type=self.instruction_type,
                    task_instruction_dir=self.task_instruction_dir,
                    objects_description_dir=self.objects_description_dir,
                    humanize_task_name=self.humanize_task_name,
                    max_instructions_per_episode=self.max_instructions_per_episode,
                )
            )

    def _instruction_for_episode(
        self,
        episode: dict[str, Any],
        start: int = 0,
    ) -> str:
        candidates = episode.get("instruction_candidates")
        if not candidates:
            return _task_text(str(episode["task"]), self.humanize_task_name)
        return select_instruction_candidate(
            candidates,
            instruction_selection=self.instruction_selection,
            task_name=str(episode["task"]),
            phase=str(episode["phase"]),
            episode_index=int(episode["episode_index"]),
            start_index=int(start),
        )

    def _prompt_for_episode(self, episode: dict[str, Any], start: int = 0) -> str:
        return self.prompt_template.format(
            task=self._instruction_for_episode(episode, start=start)
        )
