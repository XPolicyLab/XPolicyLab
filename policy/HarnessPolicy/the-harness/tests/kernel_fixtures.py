"""Kernel bootstraps for driver tests (imported by the kernel process from tests/)."""


class HandedSession:
    """A plain session in which a program marks the task handed to a learned policy
    (handed()), and the driver's hook __policy_to_end__ records that it ran."""

    def __init__(self, options=None):
        self.strategy = None
        self.namespace = {"__name__": "__program__", "handed": self._handed, "__policy_to_end__": self._to_end}

    def _handed(self):
        self.strategy = {"strategy": "C", "kind": "stacking", "policy": "simate", "policy_ran": True}

    def _to_end(self):
        print("the policy ran on to the end")
        self.strategy["ran_to_end"] = True

    def drain(self):
        return {"strategy": dict(self.strategy)} if self.strategy else {}

    def close(self):
        pass
