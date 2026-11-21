"""The code branch: one program per task, driven by the runtime in this package.

A program is a Python generator. Each `yield` produces one control step; a
program may also yield a request that gives the rest of the episode to the
learned branch. `gm.model` picks the program for a task by name, `gm.runtime`
runs it and records what happened, and `gm.lib` holds the geometry, kinematics,
motion and perception helpers the programs share.

Nothing here reads a reward, a success flag or a layout identifier. A program
sees the robot's own state, three camera images and the instruction.
"""
