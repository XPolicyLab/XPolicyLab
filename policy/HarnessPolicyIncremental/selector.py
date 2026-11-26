"""Semantic task-skill selector driven solely by natural-language instruction (and optional RGB cues).

Never reads task_name or benchmark metadata. Selects one of the 42 base manipulation skills
from the episode's instruction string using structured intent rules + token-overlap fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    description: str
    required_all: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


# 42 canonical manipulation skills, keyed by semantic intent rather than benchmark task_name.
SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        "align_blocks",
        "Use the set square to push the three blocks into a straight aligned row",
        required_all=("set square", "block"),
    ),
    SkillSpec(
        "arrange_largest_number",
        "Arrange the numbers from left to right to form the largest possible number on the pad",
        required_all=("largest", "number"),
    ),
    SkillSpec(
        "build_tower",
        "Build a tower using the wooden blocks and wooden boards",
        required_all=("tower", "board"),
    ),
    SkillSpec(
        "classify_objects_by_language",
        "Put specified object categories into the left, middle, and right baskets",
        required_all=("left", "right", "basket"),
        required_any=("middle", "center basket", "centre basket"),
    ),
    SkillSpec(
        "classify_objects",
        "Sort the objects by category into the three baskets",
        required_all=("category", "basket"),
    ),
    SkillSpec(
        "cover_blocks",
        "Cover the blocks from left to right, remember their colors, then uncover them",
        required_all=("cover", "uncover"),
    ),
    SkillSpec(
        "deposit_coin",
        "Pick up the coin from the holder and insert it precisely into the coin bank",
        required_all=("coin", "coin bank"),
    ),
    SkillSpec(
        "fasten_screws",
        "Screw each nut onto the bolt of the same color",
        required_all=("nut", "bolt"),
    ),
    SkillSpec(
        "fill_egg_holder",
        "Place the four eggs from the basket into the egg holder then close the lid",
        required_all=("egg", "egg holder"),
    ),
    SkillSpec(
        "fill_pen_holder",
        "Hold the pen holder with one hand and place all pens into it with the other hand",
        required_all=("pen holder", "pen"),
    ),
    SkillSpec(
        "fold_clothes",
        "Fold the clothes neatly",
        required_all=("fold",),
        required_any=("clothes", "sleeve", "garment", "shirt"),
    ),
    SkillSpec(
        "general_pickup",
        "Pick up the target object by 10 cm",
        required_any=("designated target", "by 10 cm", "pick it up", "keep it held", "keep it in the gripper"),
        forbidden=("conveyor",),
    ),
    SkillSpec(
        "hang_mugs",
        "Hang all the mugs on the mug rack",
        required_all=("mug", "rack"),
    ),
    SkillSpec(
        "imitate_sorting_sequence",
        "Observe the object placement order, remember it, then place the corresponding objects into the basket",
        required_all=("observe", "placement order"),
    ),
    SkillSpec(
        "insert_key",
        "Pick up the key, hand it over, insert it into the keyhole, then turn it",
        required_all=("key", "keyhole"),
    ),
    SkillSpec(
        "insert_tubes",
        "Insert the three tubes into the rack one by one",
        required_all=("tube", "rack"),
    ),
    SkillSpec(
        "make_kong",
        "Wait for the opponent to discard a tile then declare a kong with the matching tiles",
        required_all=("kong", "tile"),
    ),
    SkillSpec(
        "make_toast",
        "Pick up two slices of bread, place them into the toaster, and press the lever down",
        required_all=("bread", "toaster"),
    ),
    SkillSpec(
        "match_and_pick_from_conveyor",
        "Remember the first object on the conveyor then pick the matching object when it appears again",
        required_all=("conveyor",),
        required_any=("first object", "first item", "keeps moving", "appears again"),
        forbidden=("display board", "image on the board"),
    ),
    SkillSpec(
        "organize_table",
        "Place the alarm clock on the drawer, figurine on the stand, mouse on mouse pad, keyboard into frame",
        required_all=("alarm clock", "keyboard"),
    ),
    SkillSpec(
        "pack_objects_into_box",
        "Place all the objects into the box with their front sides facing left",
        required_all=("box",),
        required_any=("facing left", "toothbrush"),
    ),
    SkillSpec(
        "pick_from_conveyor_by_image",
        "Lift the basket, identify the target object on the conveyor according to the image on the board",
        required_all=("conveyor",),
        required_any=("image on the board", "display board", "photo"),
    ),
    SkillSpec(
        "play_Xylophone",
        "Pick up the mallet and strike all xylophone keys from left to right",
        required_all=("mallet", "xylophone"),
    ),
    SkillSpec(
        "play_stacking_toy",
        "Place all stacking toy pieces onto the correct pegs",
        required_all=("peg", "pieces"),
    ),
    SkillSpec(
        "play_tic_tac_toe",
        "Play tic-tac-toe as the first player and fill the board with the opponent",
        required_all=("tic-tac-toe",),
    ),
    SkillSpec(
        "plug_in_charger",
        "Plug the charger into the power strip",
        required_all=("charger", "power strip"),
    ),
    SkillSpec(
        "pour_balls_into_vase",
        "Pour all the balls from the cup into the vase",
        required_all=("pour", "balls", "vase"),
    ),
    SkillSpec(
        "pour_by_language",
        "Pour the liquid from the first, second, and third colored bottles into the specified bowls",
        required_all=("bottle", "bowl"),
        required_any=("pour", "required order"),
    ),
    SkillSpec(
        "pour_liquid_into_cup",
        "Pour the liquid from the bottle into the cup",
        required_all=("pour", "liquid", "cup"),
        forbidden=("bowl",),
    ),
    SkillSpec(
        "press_by_number",
        "Press each red button as many times as its card shows, tapping blue to confirm each count",
        required_all=("red button", "card"),
    ),
    SkillSpec(
        "push_T",
        "Push the T-shaped block to align it precisely with the gray T-shaped pad",
        required_all=("push", "t-shaped", "pad"),
    ),
    SkillSpec(
        "put_bottles_into_dustbin",
        "Pick up the bottles and throw them into the dustbin",
        required_all=("bottle", "dustbin"),
    ),
    SkillSpec(
        "solve_equation",
        "Complete the equation by selecting the correct missing number or operator and placing it on the pad",
        required_all=("equation",),
    ),
    SkillSpec(
        "sort_nesting_dolls_by_size",
        "Arrange the five nesting dolls in a row from left to right from smallest to largest",
        required_all=("nesting doll",),
    ),
    SkillSpec(
        "stack_blocks_by_language",
        "Stack the blocks from bottom to top in the specified color order",
        required_all=("stack", "block"),
        required_any=("bottom to top", "top to bottom", "color order", "colour order", "order of"),
    ),
    SkillSpec(
        "stack_blocks",
        "Stack the three blocks with different textures",
        required_all=("stack", "block"),
    ),
    SkillSpec(
        "stack_bowls",
        "Stack the three bowls together",
        required_all=("bowl",),
        required_any=("stack", "nest", "pile"),
    ),
    SkillSpec(
        "store_laptop_and_headphones",
        "Hang the headphones on the headphone stand, close the laptop, place it into the vertical laptop stand",
        required_all=("headphone", "laptop"),
    ),
    SkillSpec(
        "store_tools_in_toolbox",
        "Place each tool into its matching position in the toolbox",
        required_all=("tool", "toolbox"),
    ),
    SkillSpec(
        "swap_T",
        "Pick up the two T-shaped blocks, swap their positions, and place them back with the correct orientations",
        required_all=("t-shaped", "swap"),
    ),
    SkillSpec(
        "swap_blocks",
        "Swap the two blocks using the empty mat, pressing the button after each move",
        required_all=("block", "empty mat"),
    ),
    SkillSpec(
        "sweep_blocks",
        "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep",
        required_all=("dustpan",),
        required_any=("broom", "brush", "sweep"),
    ),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[\u2010-\u2014]", "-", str(text or "").lower()).split()).replace(
        "bottom-to-top", "bottom to top").replace("top-to-bottom", "top to bottom")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(_normalize(text)))


def select_skill(instruction: str, obs: dict[str, Any] | None = None) -> str:
    """Select the manipulation skill purely from the natural-language instruction (and optional obs).

    Raises ValueError if the instruction is empty or cannot be matched.
    """
    norm = _normalize(instruction)
    if not norm:
        raise ValueError("Cannot select skill from empty instruction")

    # 1. Exact semantic rule match
    matches: list[tuple[int, SkillSpec]] = []
    for spec in SKILL_SPECS:
        if any(f in norm for f in spec.forbidden):
            continue
        if spec.required_all and not all(req in norm for req in spec.required_all):
            continue
        if spec.required_any and not any(req in norm for req in spec.required_any):
            continue
        specificity = sum(map(len, spec.required_all)) + max(
            (len(req) for req in spec.required_any if req in norm), default=0)
        matches.append((specificity, spec))

    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1].skill_id

    # 2. Token-overlap fallback for paraphrased instructions
    q_tokens = _tokens(norm)
    best_score = 0.0
    best_id: str | None = None
    for spec in SKILL_SPECS:
        d_tokens = _tokens(spec.description)
        if not d_tokens:
            continue
        overlap = len(q_tokens & d_tokens)
        score = overlap / (len(q_tokens | d_tokens) ** 0.5)
        if score > best_score:
            best_score = score
            best_id = spec.skill_id

    if best_id is None or best_score <= 0.0:
        raise ValueError(f"Unable to match instruction to any known skill: {instruction!r}")
    return best_id
