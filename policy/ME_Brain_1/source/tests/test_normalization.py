import json

import numpy as np

from focus_vlwa.data.normalization import load_norm_stats, normalize_quantile, unnormalize_quantile


def test_quantile_normalization_round_trip(tmp_path) -> None:
    directory = tmp_path / "assets" / "arx_x5_sim"
    directory.mkdir(parents=True)
    stats = {
        "norm_stats": {
            "state": {"mean": [1.0], "std": [2.0], "q01": [-1.0], "q99": [3.0]},
        }
    }
    (directory / "norm_stats.json").write_text(json.dumps(stats))
    loaded = load_norm_stats(tmp_path)
    values = np.asarray([0.0], dtype=np.float32)
    normalized = normalize_quantile(values, loaded["state"])
    assert loaded["state"].q01.dtype == np.float64
    np.testing.assert_allclose(unnormalize_quantile(normalized, loaded["state"]), values, atol=1e-14)
