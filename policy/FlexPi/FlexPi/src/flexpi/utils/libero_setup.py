"""Make ``import libero`` behave deterministically.

LIBERO ships as a plain source tree, not a pip package (its ``setup.py``
``find_packages()`` returns ``[]``), so which copy a process gets is decided
entirely by ``PYTHONPATH``. Two upstream behaviours can silently break that
contract. Call :func:`prepare_libero` before importing ``libero.libero`` to
close both.

Every producer *and* consumer of LIBERO data must call this, not just the eval
entry points: whatever writes ``camera_intrinsics.json`` probes the same mujoco
scene the eval later reads back, so if it bound a different package than the
eval does, the skew would be invisible.
"""

import functools
import importlib.util
import os

import yaml

__all__ = ["prepare_libero", "bind_libero_config_path", "patch_torch_load"]


def patch_torch_load():
    """Let LIBERO load its numpy-pickled init states under torch>=2.6.

    Upstream ``Benchmark.get_task_init_states()`` calls bare
    ``torch.load(init_states_path)`` (``benchmark/__init__.py:164``). torch 2.6
    flipped ``weights_only`` to True, so that raises ``UnpicklingError`` on
    ``numpy.core.multiarray._reconstruct`` and the eval dies before its first
    rollout. Patching here instead of in ``third_party/LIBERO`` keeps that
    submodule bit-identical to upstream Lifelong-Robot-Learning/LIBERO.

    Idempotent, and a caller-supplied ``weights_only=`` still wins (partial
    keywords are overridable).
    """
    import torch

    if getattr(torch.load, "_flexpi_weights_only_patch", False):
        return
    unpatched = torch.load
    torch.load = functools.partial(unpatched, weights_only=False)
    torch.load._flexpi_weights_only_patch = True


def bind_libero_config_path(verbose=True):
    """Pin LIBERO's data dirs to the package actually on ``sys.path``.

    ``libero.libero.get_libero_path()`` reads ``~/.libero/config.yaml`` and
    nothing else. Upstream writes that file once, on the very first import, and
    never refreshes it — so once it exists it silently overrides PYTHONPATH. A
    config left behind by an earlier checkout (or by a LIBERO variant such as
    LIBERO-plus/PRO) then points this run at another tree's bddl/init/asset
    files, which no log surfaces because every suite name still resolves.

    Fix: give each package its own config dir keyed by the package root's
    directory name, via LIBERO's own ``LIBERO_CONFIG_PATH`` env var, and write
    the yaml here. Writing it also sidesteps upstream's interactive ``input()``
    bootstrap, which has no tty under sbatch/nohup.

    A caller-set ``LIBERO_CONFIG_PATH`` is respected as an explicit override.
    """
    if os.environ.get("LIBERO_CONFIG_PATH"):
        return None
    # find_spec resolves the origin WITHOUT importing libero.libero, so the
    # upstream bootstrap (and its input() prompt) never runs first.
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.origin:
        return None
    benchmark_root = os.path.dirname(os.path.abspath(spec.origin))
    # .../third_party/<PKG>/libero/libero/__init__.py -> tag = <PKG>
    tag = os.path.basename(os.path.dirname(os.path.dirname(benchmark_root)))
    cfg_dir = os.path.expanduser(os.path.join("~/.libero", tag))
    os.environ["LIBERO_CONFIG_PATH"] = cfg_dir

    wanted = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "bddl_files"),
        "init_states": os.path.join(benchmark_root, "init_files"),
        "datasets": os.path.join(os.path.dirname(benchmark_root), "datasets"),
        "assets": os.path.join(benchmark_root, "assets"),
    }
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    try:
        with open(cfg_file) as f:
            current = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        current = {}
    # Rewrite whenever the stored root drifts from the package on sys.path, so
    # the binding is self-healing rather than sticky like upstream's.
    if current.get("benchmark_root") != benchmark_root:
        os.makedirs(cfg_dir, exist_ok=True)
        with open(cfg_file, "w") as f:
            yaml.dump(wanted, f)
    if verbose:
        print(f"[libero-bind] {tag}: {benchmark_root} (config {cfg_file})", flush=True)
    return benchmark_root


REQUIRED_MUJOCO = "3.3.2"


def assert_mujoco_pin():
    """Refuse to run an eval on a mujoco other than the one that rendered the data.

    The published dataset (``libero_mujoco3.3.2_depth``) was rendered under
    mujoco 3.3.2. Under 3.8.0 the libero_object agentview shifts by ~35 RGB
    levels across ~80% of pixels, so the policy sees an out-of-distribution
    image and the resulting success rates are not comparable to any published
    number — while nothing crashes and no log says so.

    Deliberately a raise, not an ``assert``: asserts are stripped under
    ``python -O``, and this is a correctness gate, not a debug check.

    NOT called from :func:`prepare_libero` — diagnosing a wrong mujoco requires
    importing and rendering under it, which this would forbid. Call it from entry
    points that PRODUCE NUMBERS instead. Training does not need it: it reads
    rendered frames off disk and never invokes the renderer.
    """
    import importlib.metadata as md

    import mujoco

    imported = getattr(mujoco, "__version__", None)
    try:
        declared = md.version("mujoco")
    except md.PackageNotFoundError:
        declared = None
    # Both, because a shadowing install makes the two disagree and the declared
    # one is the liar.
    if imported == REQUIRED_MUJOCO and declared == REQUIRED_MUJOCO:
        return

    raise RuntimeError(
        f"LIBERO eval requires mujoco {REQUIRED_MUJOCO}, found "
        f"imported={imported} declared={declared} "
        f"(from {getattr(mujoco, '__file__', '?')}).\n"
        f"The published dataset was rendered under {REQUIRED_MUJOCO}; another "
        f"version silently changes what the policy sees (libero_object agentview "
        f"shifts ~35 RGB levels under 3.8.0), so success rates would not be "
        f"comparable to published numbers.\n"
        f"Fix:  pip install --no-deps mujoco=={REQUIRED_MUJOCO}"
    )


def prepare_libero(verbose=True):
    """Apply both fixes. Safe to call repeatedly and from any entry point."""
    patch_torch_load()
    return bind_libero_config_path(verbose=verbose)
