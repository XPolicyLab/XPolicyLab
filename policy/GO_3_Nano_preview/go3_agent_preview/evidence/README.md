# Evidence

## Download evidence

Evidence documentation and reproduction scripts are included here. The 667 data files
(131.39 MiB uncompressed) are hosted in the public
[GO-3 Nano Preview evidence dataset](https://huggingface.co/datasets/skwang123/go3-nano-preview-evidence).
No Hugging Face account or access token is required.

From the XPolicyLab repository root, restore the original data paths before running the checks:

```bash
python3 policy/GO_3_Nano_preview/go3_agent_preview/evidence/download_evidence.py
```

The helper uses Python's standard library, downloads a pinned dataset revision, verifies the
archive and every file against `data_manifest.json`, and restores the data into this directory.
It leaves matching existing files unchanged and refuses to overwrite files with different contents.
The downloaded data are ignored by Git. Runtime assets in `../gm/programs/data/` remain bundled
with the policy and do not require this download.

## Reproduction checks

Each folder below reproduces the stored values of one task program from that task's own recorded
robot data. What ships is the method plus a small sample, not a corpus: the scripts are the
deliverable, `data/` holds the compact tables they produce, and `frames/` holds a few recorded
frames so the pipeline can be run end to end from inside the folder. The method was applied to far
more runs than the sample holds; each folder's README says how many, and none of them claims that
everything it ships can be regenerated from everything it ships.

The checks on the shipped sample compare with small tolerances rather than bit for bit, so they reproduce
across platforms and library versions.

Every folder has the same shape:

- `README.md` — the item table: what is reproduced, from which recorded run, and the tolerance it meets.
- `check.py` — prints the stored value, the value recomputed from the data, the difference and the
  tolerance, and fails when the difference exceeds it.
- `scripts/` — stage-1/stage-2 scripts that read only the folder's own data, the camera and kinematics
  helpers under `gm/lib`, and `TABLE_Z`.
- `data/` — the compact recorded data the scripts read.
- `frames/` — a sample of the recorded camera frames, as RGB arrays.
- `exclusion.json` — items this folder does not reproduce, each with a one-line reason.

Run one with `python check.py` from inside its folder.

These checks compare against a stored extraction, so they are sensitive to the numerical stack:
they were recorded with one set of library versions and a newer OpenCV or NumPy can shift a fitted
centre past its tolerance. When a check fails, reproduce it against the commit named in the folder's
README before treating it as a change in the code.

| Task |
| --- |
| align_blocks |
| arrange_largest_number |
| build_tower |
| classify_objects |
| classify_objects_by_language |
| cover_blocks |
| deposit_coin |
| fasten_screws |
| fill_pen_holder |
| fold_clothes |
| imitate_sorting_sequence |
| insert_key |
| insert_tubes |
| make_kong |
| match_and_pick_from_conveyor |
| organize_table |
| play_Xylophone |
| play_stacking_toy |
| play_tic_tac_toe |
| plug_in_charger |
| pour_balls_into_vase |
| pour_by_language |
| pour_liquid_into_cup |
| press_by_number |
| push_T |
| put_bottles_into_dustbin |
| solve_equation |
| sort_nesting_dolls_by_size |
| stack_blocks |
| stack_blocks_by_language |
| stack_bowls |
| store_tools_in_toolbox |
| swap_T |
| swap_blocks |

The table is not full coverage, and is not meant to read as such:

- **Some task configurations have no program to evidence.** A program may hand its episode to the
  learned branch on the first action; such a task has no code plan and therefore no values to
  reproduce. Its program says so in one line.
- **At least one task with a program has no folder.** Where this method could not reduce a value to
  recorded data, there is an open gap rather than a padded row.
