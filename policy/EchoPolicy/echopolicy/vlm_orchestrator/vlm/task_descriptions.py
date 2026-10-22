# SPDX-License-Identifier: Apache-2.0

"""RoboDojo simulation-task references used by VLM planning prompts.

The benchmark descriptions are deliberately kept in a small local catalogue.
The VLM cannot browse the benchmark site during an evaluation, and a task
description gives planning calls task-level sequence and terminal-state details
that are not always present in a short generated instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskReference:
    slug: str
    name: str
    instruction: str
    description: str
    aliases: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"https://robodojo-benchmark.com/doc/sim-tasks/{self.slug}/"

    @property
    def full_score_condition(self) -> str:
        return FULL_SCORE_CONDITIONS.get(self.slug, "")


def _ref(
    slug: str,
    name: str,
    instruction: str,
    description: str,
    *aliases: str,
) -> TaskReference:
    return TaskReference(slug, name, instruction, description, aliases)


# The catalogue mirrors the 43 simulation-task pages listed at
# https://robodojo-benchmark.com/doc/sim-tasks/.
TASK_REFERENCES: tuple[TaskReference, ...] = (
    _ref(
        "stack-bowls",
        "Stack Bowls",
        "Stack the three bowls together.",
        "There are three bowls. The robot needs to stack all the bowls together.",
    ),
    _ref(
        "push-t",
        "Push T",
        "Push the T-shaped block to align it precisely with the gray T-shaped pad.",
        "There is a thin gray T-shaped pad and a T-shaped block. The robot needs to push the T-shaped block until it is precisely aligned with and fitted onto the gray pad.",
    ),
    _ref(
        "pack-objects-into-box",
        "Pack Objects Into Box",
        "Place all the objects into the box with their front sides facing left.",
        "There are several objects on the table and a box. The robot needs to pick up all the objects, place them into the box, and ensure that each object is oriented with its front side facing left.",
    ),
    _ref(
        "fold-clothes",
        "Fold Clothes",
        "Fold the clothes neatly.",
        "There is a piece of clothing. The robot needs to fold it neatly.",
    ),
    _ref(
        "hang-mugs",
        "Hang Mugs",
        "Hang all the mugs on the mug rack.",
        "There are three mugs and one mug rack. The robot needs to pick up each mug and hang all of them on the rack.",
    ),
    _ref(
        "sweep-blocks",
        "Sweep Blocks",
        "Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.",
        "There is a broom and a dustpan on the left side, and the blocks are on the right side. The robot needs to pick up the broom, hand it over to the right hand, grasp the dustpan with the left hand, and sweep the blocks into the dustpan.",
    ),
    _ref(
        "pour-liquid-into-cup",
        "Pour Liquid Into Cup",
        "Pour the liquid from the bottle into the cup.",
        "There is a bottle containing liquid and a cup. The robot needs to pick up the bottle and pour the liquid into the cup.",
        "into the cup",
    ),
    _ref(
        "make-toast",
        "Make Toast",
        "Pick up two slices of bread, place them into the toaster, and press the lever down.",
        "There is a basket containing multiple slices of bread and a toaster. The robot needs to pick up two slices one by one, place them into the toaster, and then press the lever down to start toasting.",
    ),
    _ref(
        "arrange-largest-number",
        "Arrange Largest Number",
        "Arrange the numbers from left to right to form the largest possible number, and place them on the pad.",
        "There are several number tiles on the table and a pad for placement. The robot needs to determine the order that forms the largest possible number, then place the four numbers on the pad from left to right in that order.",
        "largest possible number",
        "arrange the numbers",
    ),
    _ref(
        "sort-nesting-dolls-by-size",
        "Sort Nesting Dolls By Size",
        "Arrange the five nesting dolls in a row from left to right, from smallest to largest.",
        "There are five nesting dolls of different sizes on the table. The robot needs to sort them by size and arrange them in a straight row from left to right, from smallest to largest.",
    ),
    _ref(
        "store-laptop-and-headphones",
        "Store Laptop And Headphones",
        "Hang the headphones on the headphone stand, close the laptop, then place it into the vertical laptop stand.",
        "There is an open laptop on a laptop stand, a vertical laptop stand, a pair of headphones, and a headphone stand. The laptop opening angle is random but greater than 30 degrees. The robot needs to first hang the headphones on the headphone stand, then close the laptop, pick it up from the stand, and insert it into the vertical laptop stand.",
    ),
    _ref(
        "stack-blocks",
        "Stack Blocks",
        "Stack the three blocks with different textures.",
        "There are three blocks with different textures on the table. The robot needs to pick them up and stack them into a stable pile.",
        "different textures",
    ),
    _ref(
        "cover-blocks",
        "Cover Blocks",
        "Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, and blue.",
        "There are three covers and three blocks arranged in a random order. The blocks are red, green, and blue. The robot needs to cover the blocks from left to right, remember the color under each cover, and then uncover the blocks in the order of red, green, and blue.",
        "cover the blocks",
    ),
    _ref(
        "match-and-pick-from-conveyor",
        "Match And Pick From Conveyor",
        "Remember the first object on the conveyor, then pick the matching object when it appears again.",
        "An object first appears on the conveyor and is carried away. The robot needs to remember this object, observe the following objects on the conveyor, and pick the one that matches the first object.",
    ),
    _ref(
        "swap-blocks",
        "Swap Blocks",
        "Swap the two blocks using the empty mat, pressing the button after each move.",
        "There are three mats, two blocks placed on two of the mats, and one button. The robot needs to swap the positions of the two blocks by using the empty mat as a temporary place. After each move, the robot must press the button.",
        "swap the two blocks",
    ),
    _ref(
        "swap-t",
        "Swap T",
        "Pick up the two T-shaped blocks, swap their positions, and place them back with the correct orientations.",
        "There are two T-shaped blocks on the table. The robot needs to grasp them with both hands, swap their positions, and place them back so that each block matches the original pose of the other one, including both position and orientation. This is a memory-based task.",
    ),
    _ref(
        "press-by-number",
        "Press By Number",
        "Press the two red buttons the required number of times according to the number cards, then press the blue button to confirm.",
        "There are two number cards, two red buttons, and one blue confirmation button. The robot needs to read the numbers, press each red button the corresponding number of times, and then press the blue button to confirm. Pressing the blue button ends the task immediately.",
    ),
    _ref(
        "imitate-sorting-sequence",
        "Imitate Sorting Sequence",
        "Observe the object placement order, remember it, then place the corresponding objects into the basket in the same order.",
        "There are five categories of objects, with five objects on each side. The opposite robot places its objects into the basket on the right side in a certain order. The robot needs to observe and remember this sequence, then place its corresponding objects into the basket in the same order. This is a memory-based imitation task.",
        "same order",
        "opposite robot",
    ),
    _ref(
        "fasten-screws",
        "Fasten Screws",
        "Insert and tighten each screw into the nut of the same color.",
        "There are three screws and three nuts on the table. The screws and nuts come from five possible colors: red, blue, gray, yellow, and purple. In each episode, three different colors are selected, and the robot needs to match each screw with the nut of the same color, insert it, and tighten it. The screw positions vary within a small range, while the nut positions vary within a larger range. If a handover is needed, the robot can first place the screw in the middle and then let the other arm pick it up and fasten it.",
    ),
    _ref(
        "plug-in-charger",
        "Plug In Charger",
        "Plug the charger into the power strip.",
        "There is a charger plug and a power strip. The robot needs to pick up the charger plug and insert it into the power strip.",
    ),
    _ref(
        "insert-tubes",
        "Insert Tubes",
        "Insert the three tubes into the rack one by one.",
        "There is a tube rack and three tubes. The robot needs to pick up each tube in sequence and insert all tubes into the rack.",
    ),
    _ref(
        "pour-balls-into-vase",
        "Pour Balls Into Vase",
        "Pour all the balls from the cup into the vase.",
        "There is a cup containing many small balls and a vase. The robot needs to pick up the cup and pour all the balls into the vase.",
    ),
    _ref(
        "play-xylophone",
        "Play Xylophone",
        "Pick up the mallet and strike all xylophone keys from left to right.",
        "There is a xylophone and a mallet. The robot needs to pick up the mallet with one hand and strike all the xylophone keys from left to right.",
    ),
    _ref(
        "deposit-coin",
        "Deposit Coin",
        "Pick up the coin from the holder and insert it precisely into the coin bank.",
        "There is a coin placed on a holder and a coin bank. The robot needs to pick up the coin and accurately insert it into the slot of the coin bank.",
    ),
    _ref(
        "insert-key",
        "Insert Key",
        "Pick up the key, hand it over to the other hand, insert it into the keyhole, then turn it.",
        "There is a key and a keyhole on the table. The robot needs to pick up the key, hand it over to the other hand for pose adjustment, insert it accurately into the keyhole, and then turn it.",
    ),
    _ref(
        "build-tower",
        "Build Tower",
        "Build a stable multi-layer tower with the wooden blocks and boards.",
        "There are wooden blocks and wooden boards on the table. The robot needs to place each board or block on the supports below it, keep each layer centered over the layer beneath it, and keep all pieces upright.",
    ),
    _ref(
        "fill-pen-holder",
        "Fill Pen Holder",
        "Hold the pen holder with one hand, place all pens into it with the other hand, then put it back down.",
        "There is a pen holder and several pens. The robot needs to grasp the pen holder with one hand, use the other hand to place the pens into the holder one by one, and finally put the filled pen holder back on the table.",
    ),
    _ref(
        "classify-objects",
        "Classify Objects",
        "Sort the objects by category into the three baskets.",
        "There are three categories of objects and three baskets. The robot needs to group the objects by category and place each category into a separate basket. Any basket can be used for any category, as long as objects of the same category are placed together.",
    ),
    _ref(
        "put-bottles-into-dustbin",
        "Put Bottles Into Dustbin",
        "Pick up the bottles and throw them into the dustbin, using handover when needed.",
        "There are four bottles on the table, and they may be either standing or lying down. A dustbin is placed beside the table. The robot needs to pick up the bottles and throw them into the dustbin. Because of the shifted table layout, bottles on the right side require a handover between the two hands before being discarded.",
    ),
    _ref(
        "play-tic-tac-toe",
        "Play Tic-Tac-Toe",
        "Play tic-tac-toe as the first player and fill the board with the opponent.",
        "There is a 3-by-3 tic-tac-toe board. The robot plays as the first player, while the opponent follows a random strategy. The robot and the opponent take turns placing their marks until the board is filled.",
    ),
    _ref(
        "fill-egg-holder",
        "Fill Egg Holder",
        "Place the four eggs from the basket into the egg holder, then close the lid.",
        "There is a woven basket containing four eggs and an egg holder. The robot needs to pick up the eggs one by one, place all four into the egg holder, and then close the lid.",
    ),
    _ref(
        "organize-table",
        "Organize Table",
        "Place the mouse on the mouse pad, push the keyboard into the frame, put the figurine on the stand, place the alarm clock on the drawer, then open the drawer and put all remaining miscellaneous items inside.",
        "There are a computer, a keyboard, a mouse, an alarm clock, a cartoon figurine, three miscellaneous items, and a drawer. The robot needs to organize the table by placing the mouse on the mouse pad, pushing the keyboard into the frame, putting the figurine on the stand, placing the alarm clock on top of the drawer, opening the drawer, and putting all remaining miscellaneous items into it.",
    ),
    _ref(
        "make-kong",
        "Make Kong",
        "Wait for the opponent to discard a tile, then declare a kong with the matching tiles.",
        "There is a Mahjong setup. The opponent first pushes out a tile. The robot needs to observe the discarded tile, identify the matching tiles on its side, and perform a valid kong action. The setup guarantees that a kong is possible.",
    ),
    _ref(
        "play-stacking-toy",
        "Play Stacking Toy",
        "Place all stacking toy pieces onto the correct pegs.",
        "There is a stacking toy with four pegs and four types of pieces. The numbers of pieces in the four types are 4, 3, 2, and 1, and each peg matches one type. The robot needs to place all pieces onto their corresponding pegs correctly.",
    ),
    _ref(
        "align-blocks",
        "Align Blocks",
        "Use the set square to push the three blocks into a straight, aligned row, then reset the robot arm.",
        "There is a set square and three blocks. The robot needs to use the set square to push the blocks until they are aligned in parallel in a straight row.",
    ),
    _ref(
        "general-pickup",
        "General Pickup",
        "Pick up the target object by 10 cm.",
        "There are multiple objects. The robot needs to understand the language instruction, identify the target object, and pick it up.",
        "pick up the target object",
        "by 10 cm",
    ),
    _ref(
        "solve-equation",
        "Solve Equation",
        "Complete the equation by selecting the correct missing number or operator and placing it on the pad, then reset the robot arm.",
        "There is an arithmetic equation on the table and a pad for the answer. The equation is missing either a number or an operator. The robot needs to choose the correct missing item from the randomly arranged numbers and operators on the table and place it on the pad to complete the equation.",
        "complete the equation",
        "missing number or operator",
    ),
    _ref(
        "stack-blocks-by-language",
        "Stack Blocks By Language",
        "Stack the three blocks on top of each other in the order of color_1, color_2, and color_3, then reset the robot arm.",
        "There are several colored blocks. The robot needs to understand the language instruction and stack the blocks in the specified color order.",
        "specified color order",
        "in the order of",
    ),
    _ref(
        "classify-objects-by-language",
        "Classify Objects By Language",
        "Put category_1 objects into the left basket, category_2 objects into the middle basket, and category_3 objects into the right basket, then reset the robot arm.",
        "There are three baskets and three categories of unseen objects. The robot needs to understand the language instruction, identify the category of each object, and place the objects into the specified baskets from left to right.",
        "left basket",
        "middle basket",
        "category_1",
    ),
    _ref(
        "pick-from-conveyor-by-image",
        "Pick From Conveyor By Image",
        "Lift the basket more than 8 cm, identify the target object on the conveyor according to the image on the board, pick it up, and place it into the basket.",
        "There is a board displaying an image of the target object, a basket, and a conveyor carrying multiple objects. The robot needs to first lift the basket more than 8 cm, identify the target object on the conveyor based on the image shown on the board, pick up the target object, and place it into the basket.",
        "image on the board",
    ),
    _ref(
        "store-tools-in-toolbox",
        "Store Tools In Toolbox",
        "Place each tool into its matching position in the toolbox, then reset the robot arm.",
        "There are several tools and a toolbox with designated slots. The robot needs to pick up each tool and place it into the corresponding position in the toolbox.",
    ),
    _ref(
        "pour-by-language",
        "Pour By Language",
        "Pour the liquid from the first color_1 bottle into the first color_2 bowl, from the second color_3 bottle into the second color_4 bowl, and from the third color_5 bottle into the third color_6 bowl. Then reset the robot arm.",
        "There are three colored bottles and three colored bowls on the table. The robot needs to follow the language instruction and pour the liquid from each specified bottle into the corresponding specified bowl.",
        "colored bottles",
        "colored bowls",
        "bowl",
    ),
    _ref(
        "dlc",
        "DLC",
        "Arrange the letters to spell \"RoboDojo\" in a row.",
        "There are multiple separated letters on a cluttered tabletop with a complex background. The robot needs to identify the letters that form \"RoboDojo\" and arrange them in the correct order in a straight row.",
        "spell robodojo",
    ),
)


# The 100-point conditions published in each task page's Scoring table.
# DLC is intentionally absent because its official page is train-only and has
# no Scoring section as of 2026-09-19.
FULL_SCORE_CONDITIONS: dict[str, str] = {
    "stack-bowls": "All three bowls are stacked together, all bowls are upright, the bottom bowl is strictly and stably placed on the table, and the robot returns to origin.",
    "push-t": "The T-shaped block is within the target xy threshold, its orientation matches the target, the robot returns to origin, and the block is never lifted above the allowed height.",
    "pack-objects-into-box": "All four objects are placed in the aligned box with correct facing direction, and the robot returns to origin.",
    "fold-clothes": "Both sleeves are folded inward, the hems are close to the shoulder points in x/y, the hem line is aligned with the shoulder line within the angle threshold, and the robot returns to origin.",
    "hang-mugs": "All three mugs are hung on the rack and the robot returns to origin.",
    "sweep-blocks": "The dustpan is positioned left of the broom, stably placed upright on the table, all blocks are inside the dustpan, and the robot returns to origin.",
    "pour-liquid-into-cup": "The cup contains the required amount of liquid and the bottle is upright.",
    "make-toast": "Both toaster slots are filled, exactly two bread slices remain on the shelf, the toaster control is pressed down, and the robot returns to origin.",
    "arrange-largest-number": "All digits are correctly ordered on the pads, have valid orientation, and the robot returns to origin.",
    "sort-nesting-dolls-by-size": "All five nesting dolls stand in a one horizontal straight row, are ordered from left to right from smallest to largest, and the robot returns to origin.",
    "store-laptop-and-headphones": "Both the headphones and laptop conditions are satisfied and the robot returns to origin.",
    "stack-blocks": "All three blocks are stacked together and the robot returns to origin.",
    "cover-blocks": "All three blocks are first covered from left to right, then uncovered in red, green, and blue order; cup orientation remains valid and the robot returns to origin.",
    "match-and-pick-from-conveyor": "The matching target object is lifted at least 10 cm.",
    "swap-blocks": "The two target blocks are swapped using the empty mat, the button is pressed once after each of the three moves, the blocks finish on each other's original mats, and the robot returns to origin.",
    "swap-t": "The two T-shaped blocks swap their original xy positions, their orientations match the swapped targets, and the robot returns to origin.",
    "press-by-number": "Button `0` is pressed exactly the number shown by `num0`, the blue confirm button is pressed, button `1` is pressed exactly the number shown by `num1`, the blue confirm button is pressed again, and the robot returns to origin.",
    "imitate-sorting-sequence": "All five target objects are placed in sequence, all demonstration objects remain in the demo basket, the gripper is open, and the robot returns to origin.",
    "fasten-screws": "All three nut-bolt pairs are fastened, the gripper is open, and the robot returns to origin.",
    "plug-in-charger": "The charger is inside the socket, inserted to the required depth, upright, and the robot returns to origin.",
    "insert-tubes": "All three tubes are inserted correctly and the robot returns to origin.",
    "pour-balls-into-vase": "All seven balls are inside the vase, the cup is upright, and the robot returns to origin.",
    "play-xylophone": "The mallet tip hits all xylophone keys from left to right, and after each hit the mallet is lifted by at least 0.025 m.",
    "deposit-coin": "The coin bounding box is inside the coin bank slot region, and the robot returns to origin.",
    "insert-key": "The key is inserted deep enough into the slot, aligned closely in x/y, kept upright, and successfully turned after insertion.",
    "build-tower": "The full tower is completed with the top pieces, all eight pieces remain upright, each layer stays centered over the layer below it, the gripper is open, and the robot returns to origin.",
    "fill-pen-holder": "All four pens are inserted, the pen holder is upright, and the robot returns to origin.",
    "classify-objects": "All three categories are separated into three baskets, the objects are settled inside the baskets, and the robot returns to origin.",
    "put-bottles-into-dustbin": "All four bottles are inside the dustbin and the robot returns to origin.",
    "play-tic-tac-toe": "All five player pieces are on board cells at the correct height and the gripper is open.",
    "fill-egg-holder": "All four eggs are in the holder, the holder lid is correctly and fully closed, and the robot returns to origin.",
    "organize-table": "All four target items are organized correctly and the robot returns to origin.",
    "make-kong": "The three matching Mahjong tiles are pushed down, the non-matching tiles remain upright on the table, the target tile is correctly grasped and placed at the required position, the gripper is open, and no invalid robot motion is detected.",
    "play-stacking-toy": "All four peg groups are completed and the robot returns to origin.",
    "align-blocks": "The three cubes are in one straight aligned row, the robot returns to origin, and no cube has been lifted above the allowed height during the episode.",
    "general-pickup": "The target object is lifted at least 10 cm.",
    "solve-equation": "The correct missing number/operator piece is placed on the missing mat and the robot returns to origin.",
    "stack-blocks-by-language": "All three specified blocks are stacked in order and the robot returns to origin.",
    "classify-objects-by-language": "All three specified categories are placed into the left, middle, and right baskets as instructed, no basket contains objects from another category, and the robot returns to origin.",
    "pick-from-conveyor-by-image": "The image-specified target object is in the basket and lifted at least 8 cm.",
    "store-tools-in-toolbox": "All four tools are placed in their matching slots, covered by the toolbox, and the robot returns to origin.",
    "pour-by-language": "All three specified liquids are poured into their corresponding bowls.",
}


def _normalize(text: str) -> str:
    text = str(text or "").casefold().replace("_", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


_NORMALIZED_INSTRUCTIONS = {
    _normalize(reference.instruction): reference for reference in TASK_REFERENCES
}


def find_task_reference(instruction: str) -> TaskReference | None:
    """Match an official instruction or a distinctive task phrase."""
    normalized = _normalize(instruction)
    if not normalized:
        return None
    exact = _NORMALIZED_INSTRUCTIONS.get(normalized)
    if exact is not None:
        return exact

    candidates: list[tuple[int, TaskReference]] = []
    for reference in TASK_REFERENCES:
        phrases = (reference.slug.replace("-", " "), reference.name, *reference.aliases)
        for phrase in phrases:
            phrase_normalized = _normalize(phrase)
            if phrase_normalized and phrase_normalized in normalized:
                candidates.append((len(phrase_normalized), reference))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def format_task_reference(instruction: str) -> str:
    """Format task-level benchmark context for an observe/check user prompt."""
    reference = find_task_reference(instruction)
    if reference is None:
        return (
            "RoboDojo benchmark task reference: no catalogue entry matched this "
            "instruction. Use the supplied instruction and images only; do not "
            "invent a task description."
        )
    return (
        f"RoboDojo benchmark task reference ({reference.name}, {reference.slug}):\n"
        f"Official instruction: {reference.instruction}\n"
        f"Official description: {reference.description}\n"
        f"Official full-score condition: {reference.full_score_condition or '(not published)'}\n"
        f"Reference page: {reference.url}\n"
        "Use this as task-level context for the terminal state and required "
        "sequence. The supplied instruction and visual evidence remain "
        "authoritative; do not infer objects or actions absent from the images."
    )


__all__ = [
    "FULL_SCORE_CONDITIONS",
    "TASK_REFERENCES",
    "TaskReference",
    "find_task_reference",
    "format_task_reference",
]
