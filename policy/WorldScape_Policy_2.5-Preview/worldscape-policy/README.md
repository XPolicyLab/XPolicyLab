# WorldScape Policy 2.5 Preview inference runtime

This directory contains the runtime used by the enclosing RoboDojo adapter:
Wan2.2 5B, visual history, online T5, and joint T5 + VLM conditioning.
Use the parent [policy README](../README.md) for installation and launch commands.

Model loading, preprocessing, sampling and parameter names match the validated
inference implementation. Training CLIs, dataset conversion, standalone robot
drivers and HDF5 replay tools are not included in this inference-only submission.
The shared evaluation protocols and observation/normalization adapters are retained.
Model assets are downloaded separately and are not tracked in this repository.
