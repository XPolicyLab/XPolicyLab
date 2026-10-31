"""What an episode needs from the robot side, whatever the benchmark.

:class:`~hexashell.runtime.episode.Episode` runs the conversation, the cell kernel and the
records.  Everything that depends on the benchmark comes from a Session:
- how observations arrive;
- what the ``robot`` object in a cell can do;
- the task card and the pictures the model starts from;
- when the episode is over, and the verdict.

RoboDojo's is :class:`hexashell.xpl.session.RoboDojoSession` (XPolicyLab drives the simulator).
The other benchmarks run their simulator in a server process of their own; their sessions are
under :mod:`hexashell.benches`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

#: attach(kind, image, caption): show a picture to the model after the current cell.
Attach = Callable[[str, np.ndarray, str], None]


class Session:
    #: Benchmark name, written to result.json.
    bench = "robodojo"
    #: Module whose ``Robot`` class the sandboxed worker hands to cells as ``robot``.
    proxy = "hexashell.robot.proxy"
    #: Whether cells get ``Env_ICL`` (proxy :mod:`hexashell.robot.env_icl`, and its part of the system prompt).
    env_icl = False

    def __init__(self, info: dict[str, Any]):
        #: task_name, eval_seed and the benchmark's other episode facts (records, script paths)
        self.info = dict(info or {})
        #: problems that did not stop the episode (perception unavailable ...), for result.json
        self.errors: list[str] = []
        #: whether find / measure work in this episode
        self.perception = False

    # -- names -----------------------------------------------------------------------------------
    def prompt_folder(self) -> str:
        """The system prompt's folder under prompts/: the benchmark's name."""
        return self.bench

    def record_name(self) -> str | None:
        """The record folder's name before the time stamp; None: Records' own default."""
        return None

    def labels(self) -> dict[str, Any]:
        """Fields a script path may use ({task}, {seed}, {layout})."""
        return {"task": self.info.get("task_name"), "seed": self.info.get("eval_seed"),
                "layout": self.info.get("layout")}

    # -- lifetime ------------------------------------------------------------------------------
    def open(self, settings, rec, attach: Attach) -> None:
        """Take the first observation and set up the robot calls.  Raises if there is nothing to start from."""
        raise NotImplementedError

    def conclude(self, timeout_s: float) -> dict | None:
        """End the episode if it is still running and return the verdict (None if none came)."""
        raise NotImplementedError

    def close(self) -> None:
        """Release what the session holds (a simulator process ...).  Called once, after conclude."""

    # -- during the episode ----------------------------------------------------------------------
    @property
    def over(self) -> bool:
        raise NotImplementedError

    @property
    def verdict(self) -> dict | None:
        raise NotImplementedError

    def dispatch(self, name: str, args: tuple, kwargs: dict) -> Any:
        """Answer one ``robot.*`` call of a cell (see :class:`hexashell.kernel.host.Kernel`)."""
        raise NotImplementedError

    def card(self, settings) -> tuple[str, dict]:
        """The task card's text and what went into it (for the record)."""
        raise NotImplementedError

    def start_pictures(self) -> list[tuple[str, np.ndarray]]:
        """(camera, picture) the model gets with the task card; they stay in the conversation."""
        raise NotImplementedError

    @property
    def step(self) -> int | None:
        raise NotImplementedError

    @property
    def step_limit(self) -> int | None:
        raise NotImplementedError

    @property
    def instruction(self) -> str | None:
        raise NotImplementedError

    def frames(self) -> dict[str, np.ndarray]:
        """The cameras now, kept after a cell that moved the robot."""
        return {}

    def pictures_shown(self) -> None:
        """This turn's results, with the pictures robot.look took in it, have reached the model.  A host
        that keeps track of that hears of it (RobotHost: to_world checks picked pixels against the
        pictures the model has seen, and the driver offers a hand-over at most once per turn)."""
        host = getattr(self, "host", None)
        if host is not None and hasattr(host, "pictures_shown"):
            host.pictures_shown()

    def over_text(self) -> str:
        """One sentence for the model: why the episode is over."""
        raise NotImplementedError

    def result_fields(self, verdict: dict) -> dict[str, Any]:
        """The leading fields of result.json: which episode this was."""
        return {"task": self.info.get("task_name"), "seed": self.info.get("eval_seed"),
                "instruction": self.instruction}
