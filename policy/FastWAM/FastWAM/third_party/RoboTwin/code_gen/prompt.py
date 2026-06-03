# ====================== PROMPT =============================

BASIC_INFO = '''
In this environment, distance 1 indicates 1 meter long. Pose is representated as 7 dimention, [x, y, z, qw, qx, qy, qz].
For a 7-dimensional Pose object, you can use Pose.p to get the [x, y, z] coordinates and Pose.q to get the [qw, qx, qy, qz] quaternion orientation.
All functions which has parameter actor, and all of actor should be in the Actor object.
In the world coordinate system, the positive directions of the xyz coordinate axes are right, front, and upper respectively, so the direction vectors on the right, front,
and upper sides are [1,0,0], [0,1,0], [0,0,1] respectively. In the same way, we can get the unit vectors of the left side, back side and down side.
Each actor in the environment has one or more functional points, which are specific locations designed for interactions. 
Access functional points using actor.get_functional_point(point_id, return_type), where return_type can be "pose", "p", or "q".
'''

CODE_TEMPLATE = '''
from envs._base_task import Base_Task  
from envs.$TASK_NAME$ import $TASK_NAME$
from envs.utils import *
import sapien

class gpt_$TASK_NAME$($TASK_NAME$):
    def play_once(self):
        pass
'''

AVAILABLE_ENV_FUNCTION = {
    "open_gripper":
    "def open_gripper(self, arm_tag: ArmTag, pos=1.) -> tuple[ArmTag, list[Action]].\
        Opens the gripper of the specified arm.\
        Returns: tuple[ArmTag, list[Action]] containing the gripper-open action.\
        Args:\
        arm_tag: Which arm's gripper to open\
        pos: Gripper position (1 = fully open)",
    "close_gripper":
    "def close_gripper(self, arm_tag: ArmTag, pos=0.) -> tuple[ArmTag, list[Action]].\
        Closes the gripper of the specified arm.\
        Returns: tuple[ArmTag, list[Action]] containing the gripper-close action.\
        Args:\
        arm_tag: Which arm's gripper to close\
        pos: Gripper position (0 = fully closed)",
    "move":
    "def move(self, actions_by_arm1: tuple[ArmTag, list[Action]], actions_by_arm2: tuple[ArmTag, list[Action]] = None).\
        Executes action sequences on one or both robotic arms simultaneously.\
        No Return.\
        Args:\
        actions_by_arm1: Action sequence for the first arm, formatted as (arm_tag, [action1, action2, ...])\
        actions_by_arm2: Optional, action sequence for the second arm",

    # "move_to_pose":
    #     "def move_to_pose(self, arm_tag: ArmTag, target_pose: list) -> tuple[ArmTag, list[Action]].\
    #     Moves the end-effector of the specified arm to a specific absolute pose.\
    #     Returns: tuple[ArmTag, list[Action]] containing the move-to-pose actions.\
    #     Args:\
    #     arm_tag: The arm to control\
    #     target_pose: Absolute position and/or orientation, length 3 or 7 (xyz + optional quaternion)",

    # "move_by_displacement":
    #     "def move_by_displacement(self, arm_tag: ArmTag, x=0., y=0., z=0., quat=None, move_axis='world') -> tuple[ArmTag, list[Action]].\
    #     Moves the end-effector of the specified arm along relative directions and sets its orientation.\
    #     Returns: tuple[ArmTag, list[Action]] containing the move-by-displacement actions.\
    #     Args:\
    #     arm_tag: The arm to control\
    #     x, y, z: Displacement along each axis (in meters)\
    #     quat: Optional quaternion specifying the target orientation; if not set, uses current orientation\
    #     move_axis: 'world' means displacement is in world coordinates, 'arm' means displacement is in local coordinates",\
    "move_by_displacement":
    "def move_by_displacement(self, arm_tag: ArmTag, z=0., move_axis='world') -> tuple[ArmTag, list[Action]].\
        Moves the end-effector of the specified arm along relative directions and sets its orientation.\
        Returns: tuple[ArmTag, list[Action]] containing the move-by-displacement actions.\
        Args:\
        arm_tag: The arm to control\
        z: Displacement along the z-axis (in meters)\
        move_axis: 'world' means displacement is in world coordinates, 'arm' means displacement is in local coordinates",
    "grasp_actor":
    "def grasp_actor(self, actor: Actor, arm_tag: ArmTag, pre_grasp_dis=0.1, grasp_dis=0, gripper_pos=0., contact_point_id=None) -> tuple[ArmTag, list[Action]].\
        Generates a sequence of actions to pick up the specified Actor.\
        Returns: tuple[ArmTag, list[Action]] containing the grasp actions.\
        Args:\
        actor: The object to grasp\
        arm_tag: Which arm to use\
        pre_grasp_dis: Pre-grasp distance (default 0.1 meters), the arm will move to this position first\
        grasp_dis: Grasping distance (default 0 meters), the arm moves from the pre-grasp position to this position and then closes the gripper\
        gripper_pos: Gripper closing position (default 0, fully closed)\
        contact_point_id: Optional list of contact point IDs; if not provided, the best grasping point is selected automatically",
    "place_actor":
    "def place_actor(self, actor: Actor, arm_tag: ArmTag, target_pose: list | np.ndarray, functional_point_id: int = None, pre_dis=0.1, dis=0.02, is_open=True, **kwargs) -> tuple[ArmTag, list[Action]].\
        Places a currently held object at a specified target pose.\
        Returns: tuple[ArmTag, list[Action]] containing the place actions.\
        Args: \
        actor: The currently held object\
        arm_tag: The arm holding the object\
        target_pose: Target position/orientation, It is recommended to use the return value of actor.get_functional_point(..., 'pose') or pose in actor_list as target_pose\
        functional_point_id: Optional ID of the functional point; if provided, aligns this point to the target, otherwise aligns the base of the object\
        pre_dis: Pre-place distance (default 0.1 meters), arm moves to this position first\
        dis: Final placement distance (default 0.02 meters), arm moves from pre-place to this location, then opens the gripper\
        is_open: Whether to open the gripper after placing (default True), Set False if you need to keep gripper closed to maintain hold of the object\
        **kwargs: Other optional parameters:\
            constrain : {'free', 'align', 'auto'}, default='auto' Alignment strategy:\
                'free': Only forces the object's z-axis to align with the target point's z-axis, other axes are determined by projection.\
                'align': Forces all axes of the object to align with all axes of the target point.\
                'auto': Automatically selects a suitable placement pose based on grasp direction (vertical or horizontal).\
            pre_dis_axis : {'grasp', 'fp'} or np.ndarray or list, default='grasp'. Specifies the pre-placement offset direction.",
    "back_to_origin":
    "def back_to_origin(self, arm_tag: ArmTag) -> tuple[ArmTag, list[Action]].\
        Returns the specified arm to its predefined initial position.\
        Returns: tuple[ArmTag, list[Action]] containing the return-to-origin action.\
        Args:\
        arm_tag: The arm to return to origin",

    # "get_arm_pose":
    #     "def get_arm_pose(self, arm_tag: ArmTag) -> list[float].\
    #     Gets the current pose of the end-effector of the specified arm.\
    #     Returns: A list of 7 floats: [x, y, z, qw, qx, qy, qz], representing position and orientation.\
    #     Args:\
    #     arm_tag: Which arm to query",
}

FUNCTION_EXAMPLE = '''
You can directly use the actors provided in the actor_list:
```python
# For example, if actor_list contains ["self.object1", "self.object2"]
# You can directly use:
object1 = self.hammer
object2 = self.block
```

# Using ArmTag class to represent arms:
arm_tag = ArmTag("left")  # Left arm
arm_tag = ArmTag("right")  # Right arm

# Example of selecting an arm based on conditions:
arm_tag = ArmTag("left" if actor_position[0] < 0 else "right")

# Each actor in the environment may have multiple functional points that are useful for different interactions.
# Functional points provide precise locations for interactions like grasping, placing, or aligning objects.

# To get a functional point from an actor:
```python
functional_point_pose = actor.get_functional_point(point_id, "pose")  # Returns a complete 7-dimensional Pose object with p (position) and q (orientation)
position = functional_point_pose.p  # Get [x, y, z] position of the functional point
orientation = functional_point_pose.q  # Get [qw, qx, qy, qz] quaternion orientation of the functional point
```
Note: The pose from a functional point is already set according to the expected alignment/direction for the task. For placement, use get_functional_point(point_id, "pose") directly—do NOT construct or rotate your own quaternion.

# When stacking one object on top of another (for example, placing blockA on top of blockB):
target_pose = self.last_actor.get_functional_point(point_id, "pose")
# Use this target_pose in place_actor to place the object exactly on top of last_actor at the specified functional point.
```python
self.move(
    self.place_actor(
        actor=self.current_actor,            # The object to be placed
        target_pose=target_pose,             # The pose acquired from last_actor
        arm_tag=arm_tag,
        functional_point_id=0,               # Align functional point 0, or specify as needed
        pre_dis=0.1,
        dis=0.02,
	    pre_dis_axis="fp",    # Use functional point direction for pre-displacement, if the functional point is used
    )
)
```

For all actors in `actor_list` that are of type `pose`, such as `middle_pose` or `actor_target_pose`, these are already `Pose` objects (or lists of `Pose`), so you do **not** need to call `.get_pose()` again. You can pass them directly as `target_pose`.
Example:
```python
# Place the actor at actor_pose (already a Pose object)
self.move(
    self.place_actor(
        self.box,
        target_pose=self.actor_pose,  # already a Pose, no need for get_pose()
        arm_tag=grasp_arm_tag,
        functional_point_id=0,     # functional_point_id can be retrived from the actor list if the actor has functional points
        pre_dis=0,
        dis=0,  # set dis to 0 if is_open is False, and the gripper will not open after placing. Set the `dis` to a small value like 0.02 if you want the gripper to open after placing.
        is_open=False, # if is_open is False, pre_dis and dis will be 0, and the gripper will not open after placing.
        constrain="free", # if task requires the object to be placed in a specific pose that mentioned in the task description (like "the head of the actor should be toward xxx), you can set constrain to "align", in all of other cases, you should set constrain to "free".
        pre_dis_axis='fp',  # Use functional point direction for pre-displacement, if the functional_point_id is used
    )
)
```
Note: For the `target_actor`, It's a actor not a Pose, so you need to call `get_pose()` to get its pose. or call `get_functional_point()` to get its functional point.


For the grasping of a certain actor, you can check its position to decide which arm to use:
```python
# Get the actor's pose
actor_pose = self.actor.get_pose()  # Use actor_pose.p for position, actor_pose.q for orientation
actor_position = actor_pose.p  # [x, y, z]

# Example of selecting an arm based on conditions:
arm_tag = ArmTag("left" if actor_position[0] < 0 else "right")

# Grasp actor with selected arm
self.move(
    self.grasp_actor(actor=self.actor, arm_tag=arm_tag)
)
```

Here are some APIs and examples of grasping objects:
If you want to grasp an actor, you typically execute the following code:
```python
# Or grasp with arm_tag
self.move(
    self.grasp_actor(
        actor=self.actor, 
        arm_tag=arm_tag,        # arm_tag can be ArmTag("left") or ArmTag("right")
        pre_grasp_dis=0.1, 
        grasp_dis=0
    )
)
```

If you want to pick up an actor and lift it, you can refer to the following sample code:
```python
# Grasp the object
self.move(
    self.grasp_actor(
        actor=self.actor, 
        arm_tag=arm_tag,  # arm_tag can be ArmTag("left") or ArmTag("right") 
        pre_grasp_dis=0.1, 
        grasp_dis=0
    )
)

# Lift the object up by moving relative to current position, you should lift the arm up evrery time after grasping an object to avoid collision.
self.move(
    self.move_by_displacement(
        arm_tag=arm_tag,
        z=0.07,  # Move 7cm upward
        move_axis='world'
    )
)
```
The code for grasping with the right arm is similar to the above code.

Here are some examples of gripper control:
```python
# Open gripper fully
self.move(
    self.open_gripper(arm_tag=arm_tag, pos=1.0)  # arm_tag can be ArmTag("left") or ArmTag("right")
)

# Open gripper halfway
self.move(
    self.open_gripper(arm_tag=arm_tag, pos=0.5) # arm_tag can be ArmTag("left") or ArmTag("right")
)

# Close gripper fully
self.move(
    self.close_gripper(arm_tag=arm_tag, pos=0.0) # arm_tag can be ArmTag("left") or ArmTag("right")
)

# Close gripper halfway
self.move(
    self.close_gripper(arm_tag=arm_tag, pos=0.5) # arm_tag can be ArmTag("left") or ArmTag("right")
)
```

Here are some APIs and examples of placing objects:
To place an object at a target location, you typically execute the following code:
```python
# Place the object at a specific target pose
self.move(
    self.place_actor(
        actor=self.actor,
        arm_tag=arm_tag,
        target_pose=self.target_pose, # self.target_pose can be retrived from the actor list. 
        functional_point_id=0,  # functional_point_id can be retrived from the actor list if the actor has functional points
        pre_dis=0.1,
        dis=0.02, # set dis to 0 if is_open is False, and the gripper will not open after placing. Set the `dis` to a small value like 0.02 if you want the gripper to open after placing.
        is_open=True,  # Controls gripper state after placing: True to release object (default), False to maintain grip on object
        pre_dis_axis='fp',  # Use functional point direction for pre-displacement, if the functional_point_id is used
    )
)

# Lift the gripper up after placing to avoid collision with the object. (Only needed if is_open is True when placing, which means the object is released)
self.move(
    self.move_by_displacement(
        arm_tag=arm_tag,
        z=0.07,  # Move 7cm upward
        move_axis='world'  # Move in world coordinates
    )
```

If you want to align a functional point of the object with the target, you can specify the functional_point_id:
```python
# Place the object by aligning functional point 0 with the target pose
self.move(
    self.place_actor(
        actor=self.actor,
        arm_tag=arm_tag,
        target_pose=target_pose,
        functional_point_id=0, # functional_point_id can be retrived from the actor list if the actor has functional points
        pre_dis=0.1,
        dis=0.02,  # set dis to 0 if is_open is False, and the gripper will not open after placing.
        pre_dis_axis='fp'  # Use functional point direction for pre-displacement, if the functional_point_id is used
    )
)
```

If both arms need to work together simultaneously, use the move() function with two arm actions:
```python
# Move both arms simultaneously
left_arm_tag = ArmTag("left")
right_arm_tag = ArmTag("right")
self.move(
    self.grasp_actor(actor=self.left_actor, arm_tag=left_arm_tag),
    self.grasp_actor(actor=self.right_actor, arm_tag=right_arm_tag)
)

# Lift both actors up after grasping
self.move(
    self.move_by_displacement(arm_tag=left_arm_tag, z=0.07),  # Move left arm up by 10cm
    self.move_by_displacement(arm_tag=right_arm_tag, z=0.07)  # Move right arm up by 10cm
)
```


Place left object while moving right arm back to origin
```python
move_arm_tag = ArmTag("left")  # Specify which arm is placing the object
back_arm_tag = ArmTag("right")  # Specify which arm is moving back to origin
self.move(
    self.place_actor(
        actor=self.left_actor,
        arm_tag=move_arm_tag,
        target_pose=target_pose,
        pre_dis_axis="fp",
    ),
    self.back_to_origin(arm_tag=back_arm_tag)
)
```
The code for placing with the right arm is similar to the above code.

To return arms to their initial positions:
```python
# Return arm to origin
self.move(self.back_to_origin(arm_tag=arm_tag))

# Return both arms to origin simultaneously
left_arm_tag = ArmTag("left")
right_arm_tag = ArmTag("right")
self.move(
    self.back_to_origin(arm_tag=left_arm_tag),
    self.back_to_origin(arm_tag=right_arm_tag)
)
```
'''
