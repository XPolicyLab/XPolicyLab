"""Inference capability metadata for externally trained skill checkpoints."""


class ExternalPolicyMetadata:
    policy_family = "external_vla"
    training_available = False

    @classmethod
    def descriptor(cls):
        return {
            "name": cls.policy_name,
            "family": cls.policy_family,
            "training_available": False,
            "evaluation_only": True,
            "weights": "external checkpoint required for inference",
        }
