# Bagel-Robotics-1-Preview inference runtime

The VPP2 inference modules are vendored from the local research source snapshot
and the compact checkpoint loader lives in the policy adapter, so it can load
the pruned dense architecture without changing the existing `policy/VPP2`
adapter. Training, pruning search, datasets, and model weights are not bundled.

The original MIT and Wan-derived Apache-2.0 notices are retained in `LICENSE`
and `Apache-2.0.txt`.
