"""Hold frames (OURS_HOLD_FRAMES): the top images of a long hold, every 50 steps.

Why: without it, a `still` hold (agp_core/runner.py hold_still, OURS_WAIT_STILL) that waits through another robot's
demonstration returns only the observation after its last chunk, so the order in which the demonstration moved the
objects is not visible (in a recorded imitate_sorting_sequence episode: 5 objects placed in 480 steps without a pause,
one hold of up to 1000 steps, the order guessed wrong). Every chunk of such a hold already ends with an observation of
its own (agp_core/rgb_observe.py under official conditions: the official top camera image, captures/<id>/top.png, and the
step count state.low_level_actions); this module only picks those top images. No simulator state, no scene-change
detection: fixed spacing.

What: for a hold step of the run whose arms all hold (the only hold that can run more than EVERY steps: the runtime
allows one hold action 1..25 samples, so a hold through the `step` tool or one action in exec never gets there) and that
held more than EVERY steps in all, the top image at hold steps EVERY, 2 EVERY, ... <= the steps held (counted from the
hold's start; a mark at the hold's last step takes its last image) is attached as an image of its own: the captured
top.png itself, the same 640x480 PNG as every other top image, not re-encoded, labelled "top at step N of the hold
(steps_used M)" (N its step within the hold, M the episode's steps_used after it). The hold's end is shown as for any
step (the run's show / on_stop images). A mark without an observation of its own gets the captured one nearest to it
(ties: the earlier one), labelled with that observation's real step; an observation is never attached twice. At most
1000 / EVERY = 20 images (the run's STILL_MAX). The spacing: in the official demonstration trajectories of
imitate_sorting_sequence (all 195 layouts) two objects leave the table at least 68.7 steps apart, so no two of them move
between two images; the other watched tasks (make_kong, play_tic_tac_toe, the two conveyor tasks) show one lasting
change per wait or keep their order in every view (checked offline on recorded sessions).

Used by agp_core/runner.py only with the switch on: the images go into the run's result right after the hold step, the
step's view lists their steps (`frames`) and the run's full result records them (`hold_frames`). With the switch off
this module is not imported.
"""
EVERY = 50              # hold steps between two images
LABEL = "top at step {step} of the hold"            # an image's label, followed by " (steps_used M)" when known


def frames(chunks):
    """[(hold step, steps_used, top image path)] from the chunk results of one hold, in order: the hold step is the sum
    of the chunks' low_level_actions up to and including that chunk; a chunk without an ok observation with a top image
    adds its steps but no frame."""
    out, held = [], 0
    for result in chunks:
        held += int(((result or {}).get("action") or {}).get("low_level_actions") or 0)
        obs = (result or {}).get("observation") or {}
        if not obs.get("ok"):
            continue
        path = next((image.get("path") for image in obs.get("images") or [] if image.get("label") == "top"), None)
        if path:
            out.append((held, (obs.get("state") or {}).get("low_level_actions"), path))
    return out


def pick(found, held, every=EVERY):
    """The frames to attach, [(hold step, steps_used, path)] in order: for each mark every, 2 every, ... <= held the
    frame nearest to it (ties: the earlier one), never one frame twice. Empty when held <= every or there is no frame."""
    if held <= every or not found:
        return []
    chosen = []
    for mark in range(every, held + 1, every):
        frame = min(found, key=lambda f: (abs(f[0] - mark), f[0]))
        if frame not in chosen:
            chosen.append(frame)
    return sorted(chosen, key=lambda f: f[0])


def label(step, used):
    """The label of one image (the result's text line reads "Image <label> (640x480)")."""
    text = LABEL.format(step=step)
    return text if used is None else f"{text} (steps_used {used})"


def build(chunks, k, every=EVERY):
    """The images of one hold step (run step k) whose chunk results are `chunks`: (image references for the run's
    images, record) or ([], None) when the hold held <= every steps or has no frame."""
    held = sum(int(((r or {}).get("action") or {}).get("low_level_actions") or 0) for r in chunks)
    chosen = pick(frames(chunks), held, every)
    if not chosen:
        return [], None
    record = {"k": k, "held": held, "every": every,
              "frames": [{"step": step, "steps_used": used, "top": str(path)} for step, used, path in chosen]}
    return [{"label": label(step, used), "path": str(path)} for step, used, path in chosen], record
