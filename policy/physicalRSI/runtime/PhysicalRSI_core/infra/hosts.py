"""Adapter-owned semantics with shared environment/model service lifecycles."""

from .rpc import ServiceHost
from .rpc.main_thread_serve import FixedThreadDispatch


class EnvironmentHost(FixedThreadDispatch, ServiceHost):
    def __init__(self, environment, *, metadata, journal):
        super().__init__(metadata=metadata, journal=journal)
        self.environment = environment
        self._rpc.update(
            {
                "env.describe": lambda: dict(metadata),
                "env.observe": environment.observe,
                "env.reset": environment.reset,
                "env.step": environment.step,
                "env.chunk_step": environment.chunk_step,
            }
        )
        self._readonly_methods.update({"env.describe", "env.observe"})

    def close(self):
        self.environment.close()


class ModelHost(ServiceHost):
    def __init__(self, model, *, metadata, journal, session_timeout_s=3600):
        super().__init__(
            metadata=metadata,
            journal=journal,
            enable_sessions=True,
            session_timeout_s=session_timeout_s,
        )
        self.model = model
        self._rpc.update(
            {
                "model.describe": lambda session_id: dict(metadata),
                "model.predict": self.predict,
                "model.reset": self.reset,
            }
        )
        self._readonly_methods.add("model.describe")

    def predict(self, observation, options, *, session_id):
        if "session_id" in options or "session_ids" in options:
            raise ValueError("Session identity cannot be supplied in model options")
        return self.model.predict(observation, options, session_id=session_id)

    def reset(self, *, session_id):
        self.model.reset(session_id)
        return {"reset": True}

    def _on_session_drop(self, session_id):
        self.model.reset(session_id)

    def close(self):
        self.model.close()
