"""Constant-space native holds; the evaluator can stop before their planned end."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RepeatedAction:
    action: dict
    count: int

    def __post_init__(self):
        if type(self.count) is not int or self.count < 1:
            raise ValueError("A repeated native action needs a positive integer count")
