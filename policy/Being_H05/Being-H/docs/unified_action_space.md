# Unified Action Space

Being-H uses a **200-dimensional unified action space** that enables cross-embodiment generalization. This design allows a single model to control different robot types by mapping each robot's state and action dimensions to a shared semantic space.

## Core Concept

The unified space has two fixed dimensions:
- **unified_state_dim = 200**: Robot proprioceptive state
- **unified_action_dim = 200**: Robot action commands

Each robot embodiment maps its specific state/action components to designated **slots** within this 200-dimensional space. Unused dimensions are filled with zeros during inference.

## Slot Layout

The 200-dimensional space is semantically organized into slots. Each slot has a specific purpose:

### Right Arm End-Effector (Dims 0-8)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 0-2 | `eef_position` | 3 | Right arm end-effector position (x, y, z) |
| 3-5 | `eef_rotation` | 3 | Right arm end-effector rotation (axis-angle) |
| 6-8 | Reserved | 3 | Reserved for future use |

### Left Arm End-Effector (Dims 9-17)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 9-11 | `left_eef_position` | 3 | Left arm end-effector position (x, y, z) |
| 12-14 | `left_eef_rotation` | 3 | Left arm end-effector rotation (axis-angle) |
| 15-17 | Reserved | 3 | Reserved for future use |

### Grippers (Dims 18-19)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 18-19 | `gripper_position` | 1 | Right gripper open/close (0=closed, 1=open) |
| 19-20 | `left_gripper_position` | 1 | Left gripper open/close |

### Dexterous Hands (Dims 20-44)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 20-26 | `dexhand_position` | 6 | Right dexterous hand joints |
| 26-32 | Reserved | 6 | Right hand extension |
| 32-38 | `left_dexhand_position` | 6 | Left dexterous hand joints |
| 38-44 | Reserved | 6 | Left hand extension |

### Legacy/Special Slots (Dims 44-50)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 44-46 | `libero_gripper_position` | 2 | LIBERO-specific gripper state |
| 46-50 | Reserved | 4 | Reserved for future use |

### Arm Joints (Dims 50-70)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 50-57 | `arm_joint_position` | 7 | Right arm joint positions (7-DoF) |
| 57-64 | `left_arm_joint_position` | 7 | Left arm joint positions (7-DoF) |
| 64-66 | `head_position` | 2 | Head pan/tilt joints |
| 66-69 | `waist_position` | 3 | Waist/torso joints |
| 69-70 | Reserved | 1 | Reserved for future use |

### Mobile Base (Dims 70-76)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 70-73 | `base_position` | 3 | Mobile base position (x, y, z) |
| 73-74 | `base_motion` | 1 | Base motion command |
| 74-75 | `control_mode` | 1 | Control mode flag |
| 75-76 | Reserved | 1 | Reserved for future use |

### Reserved (Dims 76-90)

| Slot Range | Description |
|------------|-------------|
| 76-90 | Reserved for future embodiments and extensions |

### Human Hands (Dims 90-200)

| Slot Range | Semantic Name | Dimensions | Description |
|------------|---------------|------------|-------------|
| 90-100     | `right_beta`  | 10         | Right Hand Shape        (only for state, MANO parameter $\beta$)  |
| 100-110    | `left_beta`   | 10         | Left  Hand Shape        (only for state, MANO parameter $\beta$)  |
| 110-155    | `right_theta` | 45         | Right Hand Articulation (axis-angle,     MANO parameter $\theta$) |
| 155-200    | `left_theta`  | 45         | Left  Hand Articulation (axis-angle,     MANO parameter $\theta$) |

## UNIFIED_MAPPING

Each `DataConfig` class defines a `UNIFIED_MAPPING` dictionary that specifies how robot-specific modalities map to the unified space:

```python
UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
    'state.eef_position':     (0, 3),    # Maps to dims 0-2
    'state.eef_rotation':     (3, 6),    # Maps to dims 3-5
    'action.eef_position':    (0, 3),    # Maps to dims 0-2
    'action.eef_rotation':    (3, 6),    # Maps to dims 3-5
    'action.gripper_position':(18, 19),  # Maps to dim 18
}
```

The tuple `(start, end)` defines the slice indices (exclusive end) in the 200-dim vector.

## Configuration Guide

### Single-Arm Manipulator (EEF Control)

For robots like Franka Panda with end-effector control:

```python
UNIFIED_MAPPING = {
    'state.eef_position': (0, 3),      # EEF xyz position
    'state.eef_rotation': (3, 6),      # EEF rotation (axis-angle)
    'action.eef_position': (0, 3),     # EEF position delta
    'action.eef_rotation': (3, 6),     # EEF rotation delta
    'action.gripper_position': (18, 19),  # Gripper open/close
}
```

### Single-Arm with Dexterous Hand

For robots with dexterous hands instead of grippers:

```python
UNIFIED_MAPPING = {
    'state.arm_joint_position': (50, 57),   # 7-DoF arm joints
    'state.dexhand_position': (20, 26),     # 6-DoF hand joints
    'action.arm_joint_position': (50, 57),
    'action.dexhand_position': (20, 26),
}
```

### Dual-Arm Humanoid

For humanoid robots with two arms:

```python
UNIFIED_MAPPING = {
    # Right arm
    'state.arm_joint_position': (50, 57),
    'state.dexhand_position': (20, 26),
    # Left arm
    'state.left_arm_joint_position': (57, 64),
    'state.left_dexhand_position': (32, 38),
    # Head and waist
    'state.head_position': (64, 66),
    'state.waist_position': (66, 69),
    # Actions mirror state slots
    'action.arm_joint_position': (50, 57),
    'action.dexhand_position': (20, 26),
    'action.left_arm_joint_position': (57, 64),
    'action.left_dexhand_position': (32, 38),
    'action.head_position': (64, 66),
    'action.waist_position': (66, 69),
}
```

### Mobile Manipulator

For robots with a mobile base (like RoboCasa):

```python
UNIFIED_MAPPING = {
    'state.eef_position': (0, 3),
    'state.eef_rotation': (3, 6),
    'state.gripper_qpos': (44, 46),
    'state.base_position': (70, 73),
    'state.base_rotation': (73, 76),
    'action.eef_position': (0, 3),
    'action.eef_rotation': (3, 6),
    'action.gripper_position': (18, 19),
    'action.base_motion': (70, 74),
    'action.control_mode': (74, 75),
}
```

## Existing Embodiment Examples

### LIBERO (7-DoF Franka)

```python
# LiberoNoNormDataConfig
UNIFIED_MAPPING = {
    'state.eef_position': (0, 3),
    'state.eef_rotation': (3, 6),
    'state.libero_gripper_position': (44, 46),
    'action.eef_position': (0, 3),
    'action.eef_rotation': (3, 6),
    'action.gripper_position': (18, 19),
}
```

### RoboCasa (Mobile Manipulator)

```python
# RobocasaHumanDataConfig
UNIFIED_MAPPING = {
    'state.eef_position': (0, 3),
    'state.eef_rotation': (3, 6),
    'state.gripper_qpos': (44, 46),
    'state.base_position': (70, 73),
    'state.base_rotation': (73, 76),
    'action.eef_position': (0, 3),
    'action.eef_rotation': (3, 6),
    'action.gripper_position': (18, 19),
    'action.base_motion': (70, 74),
    'action.control_mode': (74, 75),
}
```

## Benefits

1. **Cross-embodiment transfer**: Train once, deploy on multiple robots
2. **Semantic alignment**: Similar components (e.g., EEF position) share the same dimensions
3. **Scalability**: New embodiments can be added without retraining from scratch
4. **Unified statistics**: Normalization computed per-dimension across embodiments
