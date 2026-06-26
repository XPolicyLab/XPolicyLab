# EventVLA RoboTwin-Mem Batch Eval

Run the default 8-task legacy QwenOFT pure-image checkpoint:

```bash
cd /mnt/workspace/yangganlin/tzz_workspace/final/EventVLA
bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh
```

Run the legacy Qwen3OFT raw-anchors-only checkpoint:

```bash
cd /mnt/workspace/yangganlin/tzz_workspace/final/EventVLA
bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_raw_anchors_only_qwen3oft_0423.sh
```

Dry-run the wiring without launching policy servers or simulation:

```bash
DRY_RUN=1 bash examples/RoboTwin-Mem/eval_batch/run_batch_eval.sh \
  examples/RoboTwin-Mem/eval_batch/weights_8tasks_pure_image_keyframe_memory_teacher_qwenoft.sh
```

Outputs are written under:

```text
examples/RoboTwin-Mem/eval_batch/logs/batch_eval/<RUN_TAG>/summary.csv
/mnt/workspace/yangganlin/tzz_workspace/final/RoboTwin-Mem/eval_result/<task>/<policy>/<task_config>/<ckpt_setting>/<timestamp>/
```

Task name mapping from the old Robotwin eval batch to RoboTwin-Mem:

```text
cover_blocks_hard             -> cover_blocks_hard
observe_and_pickup_hard       -> pick_the_unhidden_block
find_seal_and_seal_stamp      -> find_seal_and_seal_stamp
observe_and_pickup_object     -> pick_objects_in_order
put_back_block_hard           -> put_back_block_hard
press_button_keyframe         -> press_button_keyframe
rearrange_blocks_hard         -> rearrange_blocks_hard
reproduct_route               -> reproduce_route
```

`run_batch_eval.sh` also normalizes those three old names automatically if a config still uses them.
