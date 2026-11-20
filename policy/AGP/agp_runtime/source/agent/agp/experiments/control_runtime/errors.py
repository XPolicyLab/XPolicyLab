"""Shared control errors without importing the transport or camera stack."""


class NotDispatched(RuntimeError):
    """The attempted action has no durable reservation or published request.

    The caller may record a rejection. Other actors may have changed the scene,
    so new visual evidence is still required before preparing another action.
    """
