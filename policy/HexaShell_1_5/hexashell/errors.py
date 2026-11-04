"""Errors shared by the host-side tools."""


class Refused(ValueError):
    """A request the tools turn down (bad or impossible arguments); the cell sees a ValueError."""
