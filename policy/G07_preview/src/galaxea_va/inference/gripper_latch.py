class GripperLatch:
    """Latch denormalized commands independently by gripper key, across chunks.

    Commands run from 0 (closed) to ``full_scale`` (open); thresholds are percentages of it. The checkpoint's
    ``inference_config.yaml`` gives ``full_scale`` (from the gripper action q99 of its training stats).
    """

    CLOSE_BELOW_PERCENT = 70
    OPEN_FROM_PERCENT = 85
    RELEASE_DELTA_PERCENT = 15

    def __init__(self, full_scale: float) -> None:
        self.open_value = full_scale
        self.close_below = full_scale * self.CLOSE_BELOW_PERCENT / 100
        self.open_from = full_scale * self.OPEN_FROM_PERCENT / 100
        self.release_delta = full_scale * self.RELEASE_DELTA_PERCENT / 100
        self._states: dict[str, tuple[bool, float]] = {}

    def process(self, key: str, value: float) -> float:
        """Process one value in action time order; low values mean closed."""
        if key not in self._states:
            self._states[key] = (False, value)
        closed, extreme = self._states[key]
        if closed:
            extreme = min(extreme, value)
            if value > extreme + self.release_delta or value >= self.close_below:
                closed, extreme = False, value
        else:
            extreme = max(extreme, value)
            if value < self.close_below and value <= extreme - self.release_delta:
                closed, extreme = True, value
        self._states[key] = (closed, extreme)
        return 0.0 if closed else self.open_value if value >= self.open_from else value

    def reset(self) -> None:
        self._states.clear()
