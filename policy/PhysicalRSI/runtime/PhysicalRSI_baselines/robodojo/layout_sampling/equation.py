"""Independent solvable arithmetic curriculum from public task semantics."""

from copy import deepcopy

from . import catalog as c
from .placement import _unit


def complete_requests(requests, *, assets_root, seed, counter):
    # Enumerate the task's full single-digit operand domain, including two
    # digit results. No policy results or historical layouts condition this.
    equations = []
    for a in range(10):
        for b in range(10):
            values = {"plus": a + b, "minus": a - b, "multiplication": a * b}
            if b and a % b == 0:
                values["division"] = a // b
            equations.extend(
                (a, op, b, value) for op, value in values.items() if value >= 0
            )

    def choose(values, field):
        return values[
            min(
                int(_unit(seed, "solve_equation", counter, field) * len(values)),
                len(values) - 1,
            )
        ]

    a, op, b, answer = choose(equations, "equation")
    values = {0: a, 1: op, 2: b, 4: answer}
    rows = deepcopy(requests)
    by_label = {row["label"]: row for row in rows}
    if answer >= 10:
        values[4], values[5] = divmod(answer, 10)
        pad = deepcopy(by_label["mat4"])
        pad["label"] = "mat5"
        spacing = (
            by_label["mat4"]["common"]["xlim"][0]
            - by_label["mat3"]["common"]["xlim"][0]
        )
        pad["common"]["xlim"] = [value + spacing for value in pad["common"]["xlim"]]
        target = deepcopy(by_label["t4"])
        target.update(label="t5")
        target["common"]["relative_plane"] = "mat5"
        rows.extend([pad, target])
    missing = choose(sorted(values), "missing-slot")
    result = []
    for row in rows:
        label = row["label"]
        if label == f"t{missing}":
            continue
        if label.startswith("t") and label[1:].isdigit():
            value = values[int(label[1:])]
            row["category"] = "number" if isinstance(value, int) else value
            indices = c._allowed_indices(
                assets_root, "Rigid", {"name": row["category"]}
            )
            if isinstance(value, int):
                indices = [index for index in indices if index % 10 == value]
            if not indices:
                raise c.TrainLayoutError("Missing public equation symbol assets")
            row["index"] = choose(indices, label + ":appearance")
        plane = row["common"]["relative_plane"]
        if plane.startswith("mat"):
            row["common"]["relative_plane"] = plane + "/default/0"
        result.append(row)
    # Parents precede supported symbols. Fixed operators and pads are placed
    # before loose answer choices, so retries never move a fixed fixture.
    result.sort(
        key=lambda row: (
            row["common"]["relative_plane"] not in {"Table", "Ground"},
            row["common"].get("xlim", [0, 1])[0]
            != row["common"].get("xlim", [0, 1])[-1],
        )
    )
    return result, dict(
        operands=[a, b],
        operator=op,
        result=answer,
        missing_slot=missing,
        distribution="uniform_legal_single_digit_operand_equations_then_uniform_missing_slot",
    )
