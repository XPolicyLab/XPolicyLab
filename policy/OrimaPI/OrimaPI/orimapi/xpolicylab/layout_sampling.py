"""Opt-in research scheduling of original native layouts, without reading poses."""

import json


def filter_layout_queue(manager, selection):
    if set(selection) == {"task", "seed", "layout_id"}:
        layout_ids = [selection["layout_id"]]
    elif (
        set(selection) == {"task", "seed", "layout_ids"}
        and isinstance(selection["layout_ids"], list)
        and selection["layout_ids"]
    ):
        layout_ids = selection["layout_ids"]
    else:
        raise ValueError("Research selection requires task, seed and layout_id(s)")
    if selection["task"] != manager.task_name or selection["seed"] != manager.eval_seed:
        raise ValueError("Research layout selection does not match the native task/seed")
    for layout_id in layout_ids:
        if type(layout_id) is not int or layout_id not in manager.seed_info:
            raise ValueError("Research layout ID is not in the original native pool")
    # A resumed, already-completed selection stays excluded by upstream.
    existing = set(manager.seed_list)
    manager.seed_list = [lid for lid in layout_ids if lid in existing]
    manager.st_idx = manager.idx = 0
    manager.ed_idx = len(manager.seed_list)
    manager._current_batch_seeds = None


def install(selection_json):
    from env.seed_manager.seed_manager import SeedManager

    selection = json.loads(selection_json)
    original = SeedManager.init_eval

    def selected_init(manager, *args, **kwargs):
        original(manager, *args, **kwargs)
        filter_layout_queue(manager, selection)
        print("ORIMAPI_RESEARCH_LAYOUT " + json.dumps(selection), flush=True)

    SeedManager.init_eval = selected_init
