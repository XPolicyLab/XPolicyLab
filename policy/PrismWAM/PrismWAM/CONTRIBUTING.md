# Contributing

Keep changes focused. New recipes go in `configs/<benchmark>/`, inherit a stage file through
`defaults:` and set only what differs; add a launch script in `scripts/<benchmark>/`.

Released checkpoints must keep loading. If a change touches checkpoint keys, action units,
camera layout, history frames or normalization, say so in the pull request.

Do not commit credentials, weights, datasets, local absolute paths, W&B runs or videos.
Report benchmark results with the checkpoint, seed, number of episodes and the benchmark's
official aggregation. Keep third-party notices when moving code.
