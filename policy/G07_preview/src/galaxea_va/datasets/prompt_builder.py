import random
from typing import Optional

DEFAULT_PROMPT = "A video of a robot doing: {task}"

EGOCENTRIC_PROMPT = "A video recorded from a person's point of view doing: {task}"
EGOCENTRIC_EMBODIMENTS = frozenset({"egocentric", "ego_video", "ego_pose"})

DEFAULT_EMBODIMENT_PROMPT = (
    "A video of a robot doing: {task}\n"
    "Robot Embodiment: {embodiment}."
)

ARM_CONTROL_TEXT = {
    "relative_joint_angle": "relative joint angle",
    "absolute_joint_angle": "absolute joint angle",
    "relative_ee_pose": "relative ee pose",
    "delta_ee_pose": "delta ee pose",
    "absolute_ee_pose": "absolute ee pose",
}


def build_text_prompt(
    task: str,
    *,
    embodiment_type: Optional[str] = None,
    include_embodiment: bool = False,
    include_control_type: bool = False,
    use_zh_instruction_prob: float = 0.0,
) -> str:
    """Robot 和有动作的 Ego 共用 ``构型@控制方式``；纯视频 Ego 不追加 control。"""
    if not 0.0 <= use_zh_instruction_prob <= 1.0:
        raise ValueError(
            f"`use_zh_instruction_prob` must be in [0, 1], got {use_zh_instruction_prob}."
        )
    task_text = str(task)
    if "@" in task_text:
        zh_instruction, en_instruction = task_text.split("@")
        task_text = (
            zh_instruction
            if random.random() < use_zh_instruction_prob
            else en_instruction
        )
    embodiment_name = embodiment_type.partition("@")[0] if embodiment_type is not None else None
    is_egocentric = embodiment_name in EGOCENTRIC_EMBODIMENTS
    template = EGOCENTRIC_PROMPT if is_egocentric else DEFAULT_PROMPT
    prompt = template.format(task=task_text)
    if is_egocentric and (not include_control_type or embodiment_type in {"ego_video", "egocentric"}):
        return prompt

    if include_control_type:
        if embodiment_type is None:
            raise ValueError("Control prompt requires embodiment_type='<embodiment>@<control>'.")
        embodiment_name, separator, control_type = embodiment_type.partition("@")
        if not separator or not embodiment_name.strip() or control_type not in ARM_CONTROL_TEXT:
            raise ValueError(
                "Control prompt requires embodiment_type='<embodiment>@<control>' with a "
                f"supported control type, got {embodiment_type!r}."
            )
        details = []
        if include_embodiment and not is_egocentric:
            details.append(f"Robot Embodiment: {embodiment_name}.")
        details.append(f"Control: {ARM_CONTROL_TEXT[control_type]}.")
        return prompt + "\n" + " ".join(details)

    if not include_embodiment:
        return prompt

    if embodiment_type is None or str(embodiment_type).strip() == "":
        raise ValueError("`embodiment_type` is required when include_embodiment=True.")
    return DEFAULT_EMBODIMENT_PROMPT.format(
        embodiment=str(embodiment_type),
        task=task_text,
    )
