# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import this module FIRST in every Isaac Sim example.

Why this exists
---------------

Isaac Sim 5.1 ships ``omni.warp.core 1.8.2`` via ``isaacsim/extscache/`` and
prepends that directory to ``sys.path`` when the kit boots. That older Warp
does **not** support the ``wp.func(..., module=...)`` kwarg that cuRobo 2.0
uses for overload registration in
``curobo._src.geom.collision.wp_collision_kernel`` (and sibling kernel
modules), which causes every Isaac Sim launch of a cuRobo example to crash
with::

    TypeError: func() got an unexpected keyword argument 'module'

How the fix works
-----------------

Importing ``warp`` BEFORE ``SimulationApp()`` runs pins the pip-installed
``warp-lang`` (≥ 1.12) in ``sys.modules``. Once cached, every later
``import warp`` — including the one inside cuRobo's collision kernels and
anything Isaac Sim itself pulls in via ``omni.warp.core`` — short-circuits
to the already-imported module object, so the extscache Warp 1.8.2 never
shadows the newer one.

Usage
-----

At the **top** of every Isaac Sim example script, above the
``SimulationApp(...)`` call::

    from curobo.examples.isaacsim import bootstrap  # noqa: F401

The ``# noqa`` is important — static analyzers will otherwise flag the import
as unused. The import is what does the work.

Verification
------------

To confirm the pin holds, after ``SimulationApp()`` runs::

    import warp, inspect
    assert "module" in inspect.signature(warp.func).parameters
"""

# Side-effect imports: cache the pip warp-lang in sys.modules before any
# Isaac Sim code can insert its extscache directory onto sys.path OR modify
# ``warp.__path__``.
#
# We must pin EVERY submodule cuRobo touches (``warp.torch``, ``warp.context``,
# ``warp.config``), not just the top-level package. In 1.12.x these are
# lazy-loaded deprecation shims; once we've imported them, ``sys.modules``
# contains the pip versions and later ``wp.torch.<x>`` attribute access hits
# the cache instead of triggering a fresh ``warp.__path__`` lookup.
import warp  # noqa: F401
import warp.torch  # noqa: F401
import warp.context  # noqa: F401
import warp.config  # noqa: F401

# Also initialize Warp now so ``warp.context.runtime`` is set (it stays None
# until ``wp.init()`` runs). cuRobo eventually calls ``init_warp()`` itself,
# but some code paths (e.g. ``MeshData.__post_init__`` via
# ``wp.torch.device_from_torch``) read ``warp.context.runtime`` *during*
# construction of the scene collision data structures. Forcing init here
# makes that read always succeed.
warp.init()
