# Task recipes: the ten RoboDojo tasks of the dev30 and panel60 cases

[proposal: integration content for the OpenWAM-α/RoboDojo integration; not a harness requirement]

Written 2026-10-18 from python/oeh-policy/oeh_policy/teacher/task_context.json "notes" (one section per notes key):
each note expanded into numbered steps; no step comes from a run. Under review.context task.note `recipe` the LLM
component receives only the section for its task, as `task_context.recipe` in the first packet
(oeh_policy.review_context); this header is never sent. Whether any of this text changes behavior is unmeasured.

## organize_table

1. Read the instruction and list each named desk item with its instructed destination.
2. Take the named items one at a time: grasp, lift clear of the other items, place at its destination, release, then
   lift straight up before moving sideways.
3. Check in the views that every named item rests at its destination before handling the remaining items.
4. If the drawer is not open far enough to take an item, open it first.
5. Put each remaining item inside the drawer, one at a time, releasing it fully inside before withdrawing upward.
6. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## classify_objects_by_language

1. Read this episode's instruction and note the category-to-basket mapping it gives; the mapping is specific to the
   episode, so do not carry one over from another episode.
2. Identify the category of every item on the table from the views.
3. For each item: grasp, lift above the basket rims, move over the basket the mapping names, lower, release and
   withdraw upward.
4. After each placement, check that the item is inside the basket the mapping names before taking the next.
5. Continue until every item is in its mapped basket.
6. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## imitate_sorting_sequence

1. Observe the demonstration first; do not start placing until it has finished.
2. Remember the order in which the demonstration places the items and where each one goes.
3. Place the corresponding items in that same order, one at a time.
4. After each placement, check that it matches the demonstrated step before starting the next.
5. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## arrange_largest_number

1. Read the digits on the table.
2. Order them from largest to smallest digit; that sequence, left to right, is the largest number.
3. Place each digit on its pad in that left-to-right order.
4. Orient every digit correctly (upright and readable) before releasing it; rotate in the air, not on the pad.
5. Check that every digit sits on its pad, in order and correctly oriented.
6. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## pack_objects_into_box

1. Identify the front of each item.
2. Grasp each item so it can be set down with its front facing left, as instructed; rotate it in the air before
   lowering.
3. Clear the box rim with both the item and the fingers, lower it inside, release and withdraw upward.
4. Check that each placed item is inside the box with its front facing left before taking the next.
5. Continue until every item is inside the box.
6. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## classify_objects

1. Identify the categories among the items; there are three baskets, one per category.
2. Fix which basket holds which category; once a basket holds an item, that item's category is the basket's.
3. For each item: grasp, lift above the basket rims, move over its category's basket, lower, release and withdraw
   upward.
4. After each placement, check that the item is inside its category's basket.
5. Continue until all items are grouped by category in the three baskets.
6. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## build_tower

1. Identify the wooden blocks and the boards.
2. Build from the bottom up: set the supporting blocks of a layer first, then lay a board across them.
3. Start the next layer only when the layer below is supported and steady in the views.
4. Lower each piece gently onto its support, release, and withdraw upward without touching the tower.
5. Continue until the supported tower layers are built.
6. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## make_kong

1. Wait for the discard; do not handle tiles before it.
2. Identify the tiles that match the discard.
3. Expose the matching tiles, touching only those tiles.
4. Finish the tile-handling sequence.
5. Keep every other tile undisturbed; withdraw clear of the tiles when done.
6. This task's end check is its tile-handling sequence; returning the arms to their start poses is not required.

## fold_clothes

1. Locate the garment's sleeves and its lower part.
2. Fold one sleeve inward across the garment, release, then fold the other sleeve inward.
3. Grasp the lower edge and fold the lower part upward.
4. Check in the views that the garment lies in a neat folded shape; adjust an edge if it does not.
5. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.

## put_bottles_into_dustbin

1. Find every bottle in the views.
2. For each bottle: grasp, lift above the dustbin rim, move over the opening, lower, release inside and withdraw
   upward.
3. After each placement, check that the bottle is inside the dustbin before taking the next.
4. Continue until all bottles are inside the dustbin.
5. Release, retract upward and bring both arms back near their episode-start poses with both grippers open.
