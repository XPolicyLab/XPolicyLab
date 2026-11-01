# Benches

A bench is a simulator worker: one task per process, served over the
[worker protocol](worker-protocol.md). Isaac, torch, cuRobo and the simulator are
imported only in the worker process.

## RoboDojo

The dual-arm ARX X5 of RoboDojo, at the revision in `third_party/sources.toml`.

```text
benches/robodojo/
├── adapter.py     RoboDojoEnvironment: native reset and take_action, cameras, action packing, reward
├── general.py     RoboDojoTaskEnvironment: any one task, evaluation or development layouts
├── runtime.py     RoboDojoRuntimeEnvironment: the robot API, the shield
├── kinematics.py  cuRobo FK, IK and plans in the env frame (also the kinematics service)
├── devlayouts.py  development layouts, disjoint from the evaluation packs
└── codec.py, tasks.py, video.py
```

```bash
harness serve --factory benches.robodojo.runtime:create_environment \
  --options configs/workers/robodojo_runtime_dev.toml --port 8771
```

`harness run` starts and stops workers itself. By hand, set `HARNESS_ROBODOJO_RUNTIME=1`,
an absolute `HARNESS_ROBODOJO_ROOT` (the worker changes directory before it imports
RoboDojo), `CUDA_VISIBLE_DEVICES`, a writable `TMPDIR` (Isaac Lab logs under
`$TMPDIR/isaaclab`) and `OMNI_KIT_ACCEPT_EULA=YES` (`source environment/activate.sh` sets
the last). Isaac ignores SIGTERM: stop a worker with SIGINT, then SIGKILL.

### Action and observation

- A command is 14 values: components 0-5 and 7-12 are absolute `joint1`-`joint6`
  targets in radians (left arm, then right), 6 and 13 normalized gripper openings in
  [0, 1] (1 open; the tasks count at least 0.8 as open).
- One tick is one policy step at 25 Hz; RoboDojo interpolates it over its control steps.
- Reward follows RoboDojo's episode score: 1 when the success flag passes, otherwise the
  process score / 100. Reaching the step limit without success is truncation.
- Cameras `cam_head`, `cam_left_wrist`, `cam_right_wrist`, 640×480 RGB; no depth (the
  evaluation configuration renders none). End-effector poses are link6 poses.
- An unstable layout, or an episode a scripted support arm marks unstable, is an
  environment error: neither a success nor a failure of the policy.

### Layouts

- **Evaluation**: `layout_source = "eval"`, `layout_pack_seed` 0/1/2, the official packs.
- **Development**: `layout_source = "procedural"` with `dev_layout_dir` (default
  `<checkout>/harness_dev_layouts/arx_x5`) replays
  `<dir>/<task>_<seed>.json` for any seed, generating it on first use
  (`scripts/robodojo_make_dev_layouts.py` pre-generates them without Isaac Sim). The
  public RoboDojo has no layout generator, so `devlayouts.py` re-places a template
  evaluation layout with RoboDojo's own `ClutteredGenerator` within each instance's
  config limits, redraws object choices and clutter, and rejects any layout that repeats
  an evaluation layout. The same (task, seed) gives a byte-identical file. On the two
  conveyor tasks the objects on the belt are re-placed in its slots and turned a little;
  objects off it keep their pose. Not supported: `cover_blocks`, `play_tic_tac_toe`,
  `swap_blocks`, whose configs leave no placement freedom beyond the evaluation packs.

Scene configurations follow RoboDojo's `task/RoboDojo/config/_task.yml` (`conveyor` for
the conveyor tasks, a second scripted robot for `imitate_sorting_sequence`, `make_kong`,
`play_tic_tac_toe`); the policy always drives the two X5 arms. As RoboDojo's `run_eval`
does, the adapter loads the scripted arm's trajectory after reset, before the policy's
first step (the opponent's first move, the demonstration). The PhysX crash monitor
stays off: a crash ends the worker, the driver records the error, and `harness run
--rerun-errors` runs the episode again.

### Measured facts the prompts use

- Control at 25 Hz; a tick takes about 0.14 s of wall time (0.19 s with the head camera
  at 5 Hz in a controller, 0.4 s at 25 Hz). The simulator waits while programs think.
- link6 reaches forward only to y ≈ +0.1; pointing straight down
  (`[0.5, -0.5, 0.5, 0.5]`) needs z ≥ 0.9 and y ≤ 0.0 (`scripts/robodojo_reach_probe.py`).
- Table top z = 0.765. Fingertips are the centre of the gripping faces, 0.150 m along
  link6 +x (the jaws end 0.0075 m further: pointing down they touch the table at link6
  z ≈ 0.923); fingers slide along link6 y, 0.044 m each; an empty closed gripper reads
  an opening of about 0.185.
- Motions are timed at the planner's joint velocity and acceleration limits (3.14 rad/s,
  10 rad/s²); a 15 cm move takes about 33 ticks at the planner's timing, 22 at
  `speed=0.3`.
- The camera calibration is the same for every task, layout and arm pose (head camera
  fixed in the world, wrist cameras fixed on link6): `embodiments/arx_x5`.

## RoboTwin 2.0

`benches/robotwin/runtime.py`: the same robot API at 10 Hz, 50 tasks, seeds checked by
RoboTwin's expert, its own cuRobo 0.7.8. [Run on RoboTwin 2.0](../how-to/robotwin.md).

```bash
harness serve --factory benches.robotwin.runtime:create_environment \
  --options configs/workers/robotwin_runtime_dev.toml --port 8772
```
