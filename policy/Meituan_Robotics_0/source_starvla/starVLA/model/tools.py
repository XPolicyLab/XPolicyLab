"""Inference model registration only."""
class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """Decorator: register a builder function or class"""

        def decorator(framework_class):
            if key in self._registry:
                # print(ImportWarning(f"{key} already registered to {self.name}"))
                pass
            self._registry[key] = framework_class
            return framework_class

        return decorator

    def __getitem__(self, key):
        return self._registry[key]

    def list(self):
        """
        List currently registered keys; if with_values=True (not used here) return mapping {key: value_obj}.
        Using class name as value is also intuitive, e.g., framework.__name__.
        """
        return {k: v for k, v in self._registry.items()}

FRAMEWORK_REGISTRY = Registry("framework")
