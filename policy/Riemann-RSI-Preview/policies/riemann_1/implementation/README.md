# WAM inference implementation

This subtree contains the selected RoboDojo inference path, with no training
registration, dataset loaders or other robot implementations. Model modules,
checkpoint types, embodiment IDs, normalization, image assembly and EE action
semantics retain their original definitions. The inference dependency
`skyrobot/train/utils.py` keeps its module path for existing model imports.

External checkpoint and encoder identities are declared in `../policy.json`
and `../../../configs/models/downloads.json`. Install the pinned inference
requirements in a dedicated model environment; do not modify the evaluator
environment.
