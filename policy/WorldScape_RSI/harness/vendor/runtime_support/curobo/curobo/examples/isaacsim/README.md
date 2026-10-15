# cuRobo 2.0 — Isaac Sim Examples

Ports of the `MagicCurobo/examples/isaac_sim/` demos to the v2 solver API.
Every example runs inside an Isaac Sim Python environment (`SimulationApp`)
and reuses:

- `bootstrap.py` — pins pip `warp-lang` 1.12 into `sys.modules` before Isaac
  Sim loads its extscache `omni.warp.core 1.8.2`. Must be imported **before**
  `SimulationApp()` on every script.
- `helper.py` — `add_extensions`, `add_robot_to_scene`, and the v1→v2
  obstacle-sync shim `stage_obstacles_as_scene`.
- `batch_planning_utils.py` — `VariableBatchPlanner` wrapper around
  `BatchMotionPlanner` that accepts any `B ≤ max_batch_size` and pads the
  call with retract-state + retract-FK + empty-scene dummy problems.
- `scenes/` — two collision YAMLs pasted verbatim from MagicCurobo
  (`collision_test.yml`, `collision_thin_walls.yml`) so the batch reachers
  exercise the same obstacles the v1 demos did.

See the top-level [`BATCH_INTERFACES.md`](../../../BATCH_INTERFACES.md) for
the full v2 variant map and [`MIGRATION_V1_TO_V2.md`](../../../MIGRATION_V1_TO_V2.md)
for the call-by-call translation table.

---

## How to run

Every example is also wired as a `uv run` entrypoint (see
`[project.scripts]` in `pyproject.toml`), so the two invocation styles are
interchangeable — pick whichever you prefer:

```bash
# Short form (recommended)
uv run <script-name>

# Explicit form — works in any Python env that has curobo installed
uv run python -m curobo.examples.isaacsim.<module>
```

All examples accept `--max_steps N` for a bounded headless run and, where
relevant, `--headless_mode native` to skip the GUI. Argparse flags for
robots / scenes / visualisation are documented in each script's module
docstring.

> **Tip.** Plain `uv run <script>` re-resolves dependencies first and may
> fail on a transitive VCS dep. Pass `--no-sync` to skip that pre-flight:
> ```bash
> uv run --no-sync isaacsim-motion-gen-reacher
> ```
> Or activate the venv once and call the script directly:
> ```bash
> source .venv/bin/activate
> isaacsim-motion-gen-reacher        # → .venv/bin/isaacsim-motion-gen-reacher
> ```

---

## Examples

### Single-arm motion planning

| Script | What it demos | Run |
|---|---|---|
| **Pose-to-pose reacher** — one Franka, one draggable target, `MotionPlanner.plan_pose` fires every time the cube settles. Good first test to confirm the Isaac Sim ⇄ cuRobo round-trip works. | GUI / drag cube to retrigger plan | `uv run isaacsim-motion-gen-reacher` |
| **Multi-arm reacher** — a dual-EEF robot (`dual_ur10e.yml` by default) with **L** tool frames, **L** target cubes. One `plan_pose` call tracks every frame simultaneously. | GUI / drag either cube | `uv run isaacsim-multi-arm-reacher` |

### Batched motion planning (multi-env)

| Script | What it demos | Run |
|---|---|---|
| **Batch-env reacher** — 2 Franka robots at different world offsets with **different** collision scenes; both plans in one `BatchMotionPlanner.plan_pose` call. | 2 envs, shared cycle trigger | `uv run isaacsim-batch-motion-gen-reacher` |
| **Batched multi-arm reacher** — combines the two above: N dual-arm robots in N offset envs, L cubes per env. Single batched solve handles all `N × L` constraints. | Default N=2, each env has 2 arms | `uv run isaacsim-batched-multi-arm-reacher` |
| **Dynamic-batch reacher** — 4 envs, but the active "planning subset" rotates (3 → 2 → 4) every N sim steps. Uses `VariableBatchPlanner`, so the main loop can call `plan_pose` at any B ≤ max_batch_size; pad slots get an empty scene + retract-state + retract-FK problem. | Watch the `[cycle]` log to see per-slot scene reloads | `uv run isaacsim-dynamic-batch` |

### Inverse kinematics

| Script | What it demos | Run |
|---|---|---|
| **Reachability grid** — one Franka, a 10×10×5 grid of pose targets fed to a single batched IK solve. Points rendered green/red by success; animated through reachable solutions. | GUI with sphere cloud | `uv run isaacsim-ik-reachability` |
| **Batch-env IK** — v2 `solve_batch_env` in IK-only form: 2 Franka arms, 2 distinct collision worlds, one `ik.solve_pose` call per cube update. | 2 envs, shared cycle trigger | `uv run isaacsim-ik-batched-env` |
| **Humanoid batch-env IK** — **both** axes at once: N floating-base Unitree G1 bodies, each with L=2 palm tool frames, each in its own collision env. Exercises `max_batch_size=N, multi_env=True, L>1`. | `--mobile` for drag-and-reach mode | `uv run isaacsim-ik-humanoid-batched-env` |

### MPC

| Script | What it demos | Run |
|---|---|---|
| **Reactive MPC** — Franka continuously tracks a draggable cube with `ModelPredictiveControl.optimize_action_sequence`. Cube spawns on the retract-config EE so the first drag delta is zero and MPC can follow incrementally. Supports `--use_mppi` for MPPI + L-BFGS two-stage plus an on-screen planned-EE trajectory. | GUI / drag cube slowly | `uv run isaacsim-mpc` |
| **Batched MPC (multi-env)** — two Franka robots at different world offsets, each with its own collision scene (same `collision_test.yml` / `collision_thin_walls.yml` pair as `batch-motion-gen-reacher`). One batched `optimize_action_sequence` call tracks both cubes simultaneously. `max_batch_size=2, multi_env=True`. | 2 envs, drag either cube | `uv run isaacsim-batch-mpc` |

### Related (not Isaac Sim, but useful)

| Script | What it demos | Run |
|---|---|---|
| **Build robot model** — generate a cuRobo YAML / XRDF from a URDF (fits collision spheres, computes self-collision ignore matrix). See [`../../../BUILD_ROBOT_MODEL.md`](../../../BUILD_ROBOT_MODEL.md). | Standalone CLI | `uv run curobo-build-robot-model --help` |

---

## Headless smoke tests

Every reacher accepts `--headless_mode native --max_steps N` for a
non-interactive smoke run. Useful when you want to confirm an example
boots, plans, and exits cleanly — no display required:

```bash
uv run isaacsim-motion-gen-reacher       --headless_mode native --max_steps 150
uv run isaacsim-batch-motion-gen-reacher --headless_mode native --max_steps 300
uv run isaacsim-mpc                      --headless_mode native --max_steps 300
uv run isaacsim-batch-mpc                --headless_mode native --max_steps 300
uv run isaacsim-dynamic-batch            --headless_mode native --max_steps 2500 --cycle_switch_interval 300
```

Isaac Sim will log `Running in headless mode: native` and `Simulation App
Shutting Down` at the end — any other traceback is a real problem worth
investigating.

---

## Shared gotchas (read before filing bugs)

1. **Warp pin.** If a new example is added, its very first line after the
   future import must be
   `from curobo.examples.isaacsim import bootstrap  # noqa: F401`
   ahead of `SimulationApp()`. Without it, `cuRobo._src/geom/collision/…`
   collision kernels crash with
   `TypeError: func() got an unexpected keyword argument 'module'`
   (Isaac Sim 5.1 ships Warp 1.8.2 via extscache which lacks that kwarg).

2. **USD subroot paths.** `UsdWriter.add_subroot("/World", "/World/world_i", pose)`
   silently creates `/World/World/world_i` because `join_usd_path` strips
   the leading `/`. For multi-env stage layout, use the USD API directly:
   ```python
   from pxr import UsdGeom, Gf
   env_root = UsdGeom.Xform.Define(stage, f"/World/world_{i}")
   env_root.AddTranslateOp().Set(Gf.Vec3d(0, i * offset_y, 0))
   ```

3. **`BatchMotionPlanner.plan_pose` is rigid.** The input batch dim must
   equal `max_batch_size`. For variable-B cycles, use
   `VariableBatchPlanner` (see `dynamic_batch_motion_gen_reacher.py`).

4. **`JointState.stack` is deprecated.** Build batched states by stacking
   `numpy` arrays once and constructing one `JointState` directly — see
   `batch_motion_gen_reacher.py`.

5. **Interpolated plan shape.** v2 returns
   `(B, 1, max_H, dof_full)` with full-dof joints (locked fingers included).
   Slice `[s].squeeze(0)` → `(max_H, dof_full)`, then
   `trim_joint_state_trajectory(env_traj, 0, last_tstep[s])` to strip
   padding. No `get_full_js` call needed — v2 already includes locked
   joints.

6. **Replan trigger.** The canonical pattern is:
   `moved_since_last_plan AND settled_since_last_tick AND robot_static`
   (and for batch: `no_pending` across envs). Do **not** retrigger on
   `cmd_plan is None`; that fires every time a trajectory finishes and
   burns GPU on duplicate plans.
