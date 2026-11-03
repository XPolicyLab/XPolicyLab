"""Synthetic inference smoke checks; never produce task qualification receipts."""

from pathlib import Path

import numpy as np

from PhysicalRSI_core.infra.storage import atomic_json, file_digest


def validate_actions(actions, dimensions):
    if not isinstance(actions, list) or not actions:
        raise ValueError("Expert must return a nonempty action chunk")
    normalized = []
    for action in actions:
        if not isinstance(action, dict) or set(action) != set(dimensions):
            raise ValueError("Expert action fields differ from robot contract")
        clean = {}
        for key, size in dimensions.items():
            values = np.asarray(action[key])
            if (
                values.shape != (size,)
                or not (
                    np.issubdtype(values.dtype, np.integer)
                    or np.issubdtype(values.dtype, np.floating)
                )
                or not np.isfinite(values).all()
            ):
                raise ValueError("Expert action has invalid shape or nonfinite values")
            clean[key] = values.tolist()
        normalized.append(clean)
    return normalized


def probe(client, *, identity, state_template, output):
    """Test batched inference before and after reset with noncontiguous env IDs.

    The caller derives state_template from XPolicyLab robot helpers. Inputs
    are synthetic RGB and zero robot state, never benchmark observations.
    """
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    dimensions = {key: len(value) for key, value in state_template.items()}
    frames = {
        key: np.full((224, 224, 3), index * 40, dtype=np.uint8)
        for index, key in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist"))
    }
    np.savez_compressed(
        root / "synthetic-input.npz",
        **frames,
        **{key: np.asarray(value) for key, value in state_template.items()},
    )
    record = dict(
        schema="physicalrsi.robodojo.synthetic-probe/v1",
        state="running",
        identity=identity,
        physical_qualification=False,
        dimensions=dimensions,
        input_sha256=file_digest(root / "synthetic-input.npz"),
        observation_recipe="supplied state template; each base RGB frame plus env_idx, clipped to uint8; Stack the bowls instruction",
        probe_source_sha256=file_digest(Path(__file__)),
        calls=[],
    )
    atomic_json(root / "probe.json", record)
    try:
        actual = client.call(func_name="physicalrsi_identity")
        if actual.get("identity") != identity:
            raise ValueError("Probe connected to a different expert")
        for indices in ([3, 7], [7, 3]):
            client.call(func_name="reset")
            observations = [
                dict(
                    env_idx=index,
                    instruction="Stack the bowls.",
                    state={
                        key: np.asarray(value).copy()
                        for key, value in state_template.items()
                    },
                    vision={
                        key: dict(
                            color=np.clip(
                                value.astype(np.int16) + index, 0, 255
                            ).astype(np.uint8)
                        )
                        for key, value in frames.items()
                    },
                )
                for index in indices
            ]
            client.call(func_name="update_obs_batch", obs=observations)
            batches = client.call(func_name="get_action_batch", obs=indices)
            if not isinstance(batches, list) or len(batches) != len(indices):
                raise ValueError("Expert omitted batch environments")
            actions = [validate_actions(batch, dimensions) for batch in batches]
            path = root / f"actions-{len(record['calls'])}.json"
            atomic_json(path, dict(env_indices=indices, actions=actions))
            record["calls"].append(
                dict(
                    env_indices=indices,
                    horizons=list(map(len, actions)),
                    file=path.name,
                    sha256=file_digest(path),
                )
            )
            atomic_json(root / "probe.json", record)
        record["state"] = "completed"
    except BaseException as error:
        record.update(state="failed", error=repr(error))
        raise
    finally:
        atomic_json(root / "probe.json", record)
    return record
