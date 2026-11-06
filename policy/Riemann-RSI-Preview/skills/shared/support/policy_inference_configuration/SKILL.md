---
name: policy_inference_configuration
description: "Loads the first sorted *.toml model configuration from exp_dir, then from the parent of model_path if needed, and returns the parsed dict. A missing TOML is fatal instead of silently falling back to embodiment defaults."
---

# Policy inference configuration

## Function

Loads the first sorted *.toml model configuration from exp_dir, then from the parent of model_path if needed, and returns the parsed dict. A missing TOML is fatal instead of silently falling back to embodiment defaults.

## Inputs

- model_path: checkpoint path string used to derive a secondary search directory
- exp_dir: preferred directory to search for *.toml

## Outputs

- dict parsed from the first found TOML
- Prints the selected configuration path.

## Callable interface

Implementation: [implementation.py](implementation.py).

- `_load_train_config(model_path: str, exp_dir: str)`

## Constraints

- Search order is exp_dir first, then os.path.dirname(os.path.normpath(model_path)), skipping duplicate/empty candidates
- Raises FileNotFoundError listing searched directories when no TOML is found
- The TOML carries num_latents, video size, and norm_mode; missing config is treated as fatal
- Adapter-bound os global and exception behavior are preserved
