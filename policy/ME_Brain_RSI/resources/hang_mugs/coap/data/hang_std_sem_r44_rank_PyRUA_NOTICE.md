PyRUA-Lean
Copyright 2026 PyRUA-Lean Contributors

Licensed under the Apache License, Version 2.0 (see LICENSE).

This product builds on RPent (https://github.com/RLinf/RPent),
Copyright 2026 The RPent Authors, licensed under the Apache License, Version 2.0.
RPent's robot stacks (LIBERO-PRO, RoboTwin 2.0, RoboCasa365), its services (the
simulator servers and the frozen VLA and perception models it serves) and its
primitive implementations do the work behind the `robo` object: PyRUA-Lean calls
them through RPent's Toolkit and does not redistribute them.  The tool-calling
baseline of the comparison is RPent itself.

Parts of this repository are derived from RPent and were modified:
  - src/pyrualean/mcp_server.py is adapted from RPent's HttpMcpServer;
  - src/pyrualean/knowledge/*.md distil the operating knowledge of RPent's
    prompts and guides for a program author;
  - the robot classes in src/pyrualean/ mirror RPent's tool schemas (tool
    names, argument names and defaults).

RPent is described in: Harness VLA: Steering Frozen VLAs into Reliable
Manipulation Primitives via Memory-Guided Agents, arXiv:2607.08448
(https://arxiv.org/abs/2607.08448).
