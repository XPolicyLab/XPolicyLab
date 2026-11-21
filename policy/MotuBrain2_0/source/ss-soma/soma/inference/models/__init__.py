"""Inference-owned model definitions.

Deliberately empty of eager imports: Falcon's model code imports
``soma.inference.models.operators`` for its portable leaf operators, so
re-exporting ``MotubrainInferModel`` here would make importing Falcon first
re-enter a half-initialised Falcon package.  Import the model from its own
module instead::

    from soma.inference.models.motubrain import MotubrainInferModel
"""
