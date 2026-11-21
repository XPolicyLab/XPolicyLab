"""One program per task, named after the task.

`gm.model` imports the module whose name is the task it was asked to run, falling
back to the base task for a `<task>_random` name that has no module of its own. A
program is a generator; the package above says what it receives and what it may
yield.

Modules directly beside this file are programs. `support/` holds the geometry,
measurement and motion helpers several programs share, and `data/` holds the arrays
and tables those helpers read. Nothing in `support/` is ever loaded by task name.
"""
