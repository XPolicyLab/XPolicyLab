"""Map requested environment IDs to rows of the most recent observation batch."""


def validate_observed_ids(ids):
    if len(set(ids)) != len(ids):
        raise ValueError("Observation batch contains duplicate environment IDs")


def observation_env_ids(obs_list):
    ids = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
    validate_observed_ids(ids)
    return ids


def observation_rows(observed_ids, requested_ids):
    positions = {env_id: row for row, env_id in enumerate(observed_ids)}
    if len(positions) != len(observed_ids):
        raise ValueError("Observation batch contains duplicate environment IDs")
    ids = observed_ids if requested_ids is None else list(requested_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Requested batch contains duplicate environment IDs")
    try:
        return [positions[env_id] for env_id in ids]
    except KeyError as exc:
        raise ValueError(f"No observation for environment {exc.args[0]}") from exc


def selected_env_ids(observed_ids, requested_ids):
    """Select currently observed IDs before touching a policy's per-environment history."""
    return [observed_ids[row] for row in observation_rows(observed_ids, requested_ids)]
