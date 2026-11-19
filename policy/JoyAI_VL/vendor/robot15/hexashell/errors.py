"""Errors shared by the host-side tools."""


class Refused(ValueError):
    """A request the tools turn down (bad or impossible arguments); the cell sees a ValueError."""


# Errors returned by the robot command endpoint.
class CallFailed(Exception):
    def __init__(self, kind, text):
        super().__init__(text)
        self.kind = kind
        self.text = text


class Stop(Exception):
    """The episode ended during a tool call."""
