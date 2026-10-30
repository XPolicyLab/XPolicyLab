"""Every source string a task program is allowed to yield with.

A program that gives up the rest of its plan yields a request whose ``source``
must appear here for that task. The table is data, not logic: the runtime only
checks membership, so nothing in it can select a branch or a checkpoint.

Each entry names the task whose program it belongs to; ``_random`` siblings use
the base task's entry.
"""

SWITCH_SOURCES = {
    'align_blocks': {
        'align_blocks.perception_abandoned',
        'align_blocks.terminal_plan_error',
    },
    'arrange_largest_number': {
        'arrange_largest_number.count_mismatch',
        'arrange_largest_number.identity_abandoned',
        'arrange_largest_number.no_digits',
        'arrange_largest_number.transfers_abandoned',
    },
    'build_tower': {
        'build_tower.perception_abandoned',
        'build_tower.task_aborted',
    },
    'classify_objects': {
        'classify_objects.initial_scan_plan_error',
    },
    'classify_objects_by_language': {
        'classify_objects_by_language.initial_scan_plan_error',
    },
    'cover_blocks': {
        'cover_blocks.aborted',
        'cover_blocks.cubes_unavailable',
        'cover_blocks.cups_unavailable',
    },
    'deposit_coin': {
        'deposit_coin.coin_flat_abandoned',
        'deposit_coin.coin_unavailable',
        'deposit_coin.executor_abandoned',
        'deposit_coin.lift_retries_exhausted',
        'deposit_coin.pinch_abandoned',
        'deposit_coin.plan_exhausted',
        'deposit_coin.relay_coin_flat',
        'deposit_coin.relay_lift_abandoned',
        'deposit_coin.relay_pinch_abandoned',
        'deposit_coin.relay_release_abandoned',
        'deposit_coin.relay_side_exhausted',
        'deposit_coin.relay_take_abandoned',
    },
    'fasten_screws': {
        'fasten_screws.pairs_unavailable',
    },
    'fill_egg_holder': {
        'fill_egg_holder.no_code_plan',
        'fill_egg_holder.early_handoff',
    },
    'fill_pen_holder': {
        'fill_pen_holder.holder_unavailable',
    },
    'fold_clothes': {
        'fold_clothes.garment_unavailable',
    },
    'general_pickup': {
        'general_pickup.instruction_no_target',
        'general_pickup.perception_no_blobs',
        'general_pickup.perception_no_candidates',
        'general_pickup.perception_no_match',
        'general_pickup.reach_no_candidate',
    },
    'hang_mugs': {
        'hang_mugs.no_code_plan',
        'hang_mugs.early_handoff',
    },
    'imitate_sorting_sequence': {
        'imitate_sorting_sequence.demo_end_not_seen',
        'imitate_sorting_sequence.no_demonstration_objects',
        'imitate_sorting_sequence.pairing_exhausted',
        'imitate_sorting_sequence.placement_loop_exhausted',
    },
    'insert_key': {
        'insert_key.initial_plan_exhausted',
        'insert_key.key_not_found',
        'insert_key.keyhole_search_exhausted',
        'insert_key.lock_search_exhausted',
    },
    'insert_tubes': {
        'insert_tubes.budget_abandoned_remaining_tubes',
        'insert_tubes.rack_search_exhausted',
    },
    'make_kong': {
        'make_kong.main_plan_error',
    },
    'make_toast': {
        'make_toast.no_code_plan',
        'make_toast.early_handoff',
    },
    'match_and_pick_from_conveyor': {
        'match_and_pick_from_conveyor.catch_loop_exhausted',
        'match_and_pick_from_conveyor.sample_read_exhausted',
    },
    'organize_table': {
        'organize_table.keyboard_recovery_exhausted',
    },
    'pack_objects_into_box': {
        'pack_objects_into_box.no_code_plan',
        'pack_objects_into_box.early_handoff',
    },
    'pick_from_conveyor_by_image': {
        'pick_from_conveyor_by_image.no_code_plan',
        'pick_from_conveyor_by_image.early_handoff',
    },
    'play_Xylophone': {
        'play_Xylophone.perception_abandoned',
        'play_Xylophone.terminal_plan_error',
        'play_Xylophone.test_lift_abandoned',
        'play_Xylophone.visual_retries_exhausted',
    },
    'play_stacking_toy': {
        'play_stacking_toy.terminal_791',
    },
    'play_tic_tac_toe': {
        'play_tic_tac_toe.terminal_76',
    },
    'plug_in_charger': {
        'plug_in_charger.final_plan_exhausted',
        'plug_in_charger.terminal_126',
        'plug_in_charger.terminal_132',
        'plug_in_charger.terminal_139',
        'plug_in_charger.terminal_209',
        'plug_in_charger.terminal_226',
        'plug_in_charger.terminal_243',
        'plug_in_charger.terminal_65',
        'plug_in_charger.terminal_69',
        'plug_in_charger.terminal_78',
    },
    'pour_balls_into_vase': {
        'pour_balls_into_vase.grasp_chain_abandoned',
        'pour_balls_into_vase.terminal_454',
        'pour_balls_into_vase.terminal_465',
        'pour_balls_into_vase.terminal_498',
        'pour_balls_into_vase.terminal_510',
        'pour_balls_into_vase.terminal_532',
    },
    'pour_by_language': {
        'pour_by_language.terminal_137',
        'pour_by_language.terminal_84',
    },
    'pour_liquid_into_cup': {
        'pour_liquid_into_cup.perception_exhausted',
        'pour_liquid_into_cup.terminal_241',
        'pour_liquid_into_cup.terminal_252',
        'pour_liquid_into_cup.terminal_255',
        'pour_liquid_into_cup.terminal_358',
        'pour_liquid_into_cup.terminal_363',
        'pour_liquid_into_cup.terminal_375',
        'pour_liquid_into_cup.terminal_378',
        'pour_liquid_into_cup.terminal_385',
    },
    'press_by_number': {
        'press_by_number.required_stroke_plan_error',
        'press_by_number.terminal_213',
        'press_by_number.terminal_225',
    },
    'push_T': {
        'push_T.terminal_163',
        'push_T.terminal_174',
        'push_T.terminal_181',
        'push_T.terminal_213',
        'push_T.terminal_218',
    },
    'put_bottles_into_dustbin': {
        'put_bottles_into_dustbin.remaining_bottles_after_recovery',
    },
    'solve_equation': {
        'solve_equation.terminal_343',
        'solve_equation.terminal_373',
        'solve_equation.terminal_413',
    },
    'sort_nesting_dolls_by_size': {
        'sort_nesting_dolls_by_size.initial_perception_exhausted',
    },
    'stack_blocks': {
        'stack_blocks.final_grasp_not_held',
        'stack_blocks.native_transfer_budget_exhausted',
        'stack_blocks.perception_exhausted',
        'stack_blocks.pick_exhausted',
        'stack_blocks.place_exhausted',
        'stack_blocks.target_plan_exhausted',
    },
    'stack_blocks_by_language': {
        'stack_blocks_by_language.base_destination_exhausted',
        'stack_blocks_by_language.base_grasp_exhausted',
        'stack_blocks_by_language.instruction_exhausted',
        'stack_blocks_by_language.layer_transfer_exhausted',
        'stack_blocks_by_language.perception_exhausted',
    },
    'stack_bowls': {
        'stack_bowls.dual_regrasp_exhausted',
        'stack_bowls.perception_exhausted',
        'stack_bowls.rim_grasp_exhausted',
    },
    'store_laptop_and_headphones': {
        'store_laptop_and_headphones.no_code_plan',
        'store_laptop_and_headphones.early_handoff',
    },
    'store_tools_in_toolbox': {
        'store_tools_in_toolbox.perception_exhausted',
    },
    'swap_T': {
        'swap_T.perception_exhausted',
        'swap_T.stage_transfer_exhausted',
    },
    'swap_blocks': {
        'swap_blocks.button_grasp_exhausted',
        'swap_blocks.button_lift_exhausted',
        'swap_blocks.mat_perception_exhausted',
        'swap_blocks.perception_exhausted',
        'swap_blocks.stage_transfer_exhausted',
    },
    'sweep_blocks': {
        'sweep_blocks.no_code_plan',
        'sweep_blocks.early_handoff',
    },
}
