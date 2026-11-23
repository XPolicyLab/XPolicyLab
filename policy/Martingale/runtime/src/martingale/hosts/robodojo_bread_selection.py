# Copyright © 2026 Martingale. All Rights Reserved.
import statistics


def choose(fit, minimum_score=0.55):
    """Keep the existing per-view gate; rank by supporting cameras only."""
    baseline = fit.get('baseline_per_view_scores', [])
    slab = fit.get('slab_per_view_scores', [])
    if len(baseline) < 2 or len(slab) != len(baseline):
        return 'baseline'
    if min(slab) < minimum_score:
        return 'baseline'
    return ('bread_slab' if statistics.mean(slab[1:]) > statistics.mean(baseline[1:])
            else 'baseline')
