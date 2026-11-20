---
name: own_robot_kinematic_model
description: "Cached loader reads configs/mature-trajectory.json from the repository root, verifies the declared own-robot URDF SHA-256, and returns an OwnURDFKinematics instance for the source URDF."
---

# Own robot kinematic model

## Function

Cached loader reads configs/mature-trajectory.json from the repository root, verifies the declared own-robot URDF SHA-256, and returns an OwnURDFKinematics instance for the source URDF.

## Inputs

- configs/mature-trajectory.json fields robot_urdf and robot_urdf_sha256
- URDF source bytes read from root / settings['robot_urdf']

## Outputs

- OwnURDFKinematics(source) instance
- ValueError('The declared own-robot numerical model changed') when the SHA-256 does not match

## Callable interface

Implementation: [implementation.py](implementation.py).

- `run()`

## Constraints

- run() is wrapped with functools.lru_cache(maxsize=1).
- Repository root is Path(__file__).resolve().parents[4].
- The loader contains no task assets, object dimensions, layout fields, or simulator handles.
- Numerical solutions estimate the robot's own commands, never object poses or contact.
