"""Grounding and CLIP composition; observable pixels, not world geometry.

Designed to live inside an owned perception worker. All models, prompts, memory
and thresholds must be included in the caller's frozen dependency closure.
"""

import math
from copy import deepcopy

from .clip_ranking import AmbiguousGrounding, CLIPRanking, select_unique
from .rgb_grounding import RGBGrounding


class RGBTargets:
    def __init__(self, *, grounding, ranking, selection):
        self.models = []
        self.closed = False
        if grounding["queries"] != ranking["queries"]:
            raise ValueError("Grounding and ranking queries must match")
        if set(selection) != set(grounding["queries"]):
            raise ValueError("Declare selection thresholds for every query")
        for thresholds in selection.values():
            if set(thresholds) != {"minimum_score", "minimum_margin", "duplicate_iou"}:
                raise ValueError("Explicit selection thresholds required")
            if (
                not all(math.isfinite(v) for v in thresholds.values())
                or thresholds["minimum_margin"] <= 0
                or not 0 < thresholds["duplicate_iou"] <= 1
            ):
                raise ValueError("Invalid selection thresholds")
        self.selection = deepcopy(selection)
        try:
            self.detector = RGBGrounding(**deepcopy(grounding))
            self.models.append(self.detector)
            self.ranker = CLIPRanking(**deepcopy(ranking))
            self.models.append(self.ranker)
        except BaseException:
            self.close()
            raise

    def predict(self, rgb, *, deadline):
        if self.closed:
            raise RuntimeError("RGB target backend is closed")
        self.detector._check(deadline)
        # One immutable-by-convention snapshot for both inference stages; callers
        # may update their observation buffer while this worker is running.
        frame = rgb.copy()
        rows = self.detector.predict(frame, deadline=deadline)
        ranked = self.ranker.rank(frame, rows, deadline=deadline)
        selected, unresolved = {}, {}
        for query, thresholds in self.selection.items():
            try:
                selected[query] = select_unique(ranked, query=query, **thresholds)
            except AmbiguousGrounding as error:
                unresolved[query] = str(error)
        self.detector._check(deadline)
        return dict(
            schema="physicalrsi.rgb-targets/v1",
            selected=selected,
            unresolved=unresolved,
            candidates=ranked,
        )

    def close(self):
        self.closed = True
        failures = []
        for model in list(reversed(self.models)):
            try:
                model.close()
                self.models.remove(model)
            except Exception as error:
                failures.append(error)
        if failures:
            raise RuntimeError("RGB target model cleanup incomplete") from failures[0]


def create(configuration):
    return RGBTargets(**configuration)
