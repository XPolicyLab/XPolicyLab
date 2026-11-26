"""Handlers that resolve events without a model call."""


class Handlers:
    def on(self, event, handler):
        """Register handler(trace) for an event kind. Return True (or None) to resolve
        it, or False to let it escalate."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        self.handlers[str(event)] = handler
