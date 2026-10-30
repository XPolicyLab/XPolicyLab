"""Record auditable visual departure events without inferring grasp state."""


def observe_departures(memory, measurements, observation_step, minimum_frames=32):
    """Confirm persistent exposed-table regions and sort by their observed onset."""
    if observation_step <= memory.last_departure_observation_step:
        return not memory.demonstration_order_evidence.get('ambiguous_onset_steps', [])
    memory.last_departure_observation_step = int(observation_step)
    for object_id, values in measurements.items():
        if any(event['demonstration_id'] == object_id
               for event in memory.demonstration_events):
            continue
        departed = values['persistence_score'] < .35 and values['exposed_table_score'] > .70
        if departed:
            if object_id not in memory.first_missing:
                memory.first_missing[object_id] = int(observation_step)
                memory.demonstration_candidates.append({
                    'demonstration_id': object_id,
                    'first_missing_step': int(observation_step),
                    'status': 'pending', 'holding_verified': False})
            memory.missing[object_id] = memory.missing.get(object_id, 0) + 1
            if memory.missing[object_id] >= minimum_frames:
                event = dict(values, demonstration_id=object_id,
                    first_missing_step=memory.first_missing[object_id],
                    confirmed_step=int(observation_step),
                    persistent_frames=memory.missing[object_id],
                    event_kind='rgb_source_departure',
                    persistence_threshold=.35, exposed_table_threshold=.70,
                    holding_verified=False, private_grasp_state_used=False)
                memory.demonstration_events.append(event)
                for candidate in reversed(memory.demonstration_candidates):
                    if candidate['demonstration_id'] == object_id and candidate['status'] == 'pending':
                        candidate.update(status='confirmed', confirmed_step=int(observation_step))
                        break
        else:
            if values['persistence_score'] >= .35:
                if object_id in memory.first_missing:
                    for candidate in reversed(memory.demonstration_candidates):
                        if candidate['demonstration_id'] == object_id and candidate['status'] == 'pending':
                            candidate.update(status='visibility_recovered', reset_step=int(observation_step))
                            break
                memory.first_missing.pop(object_id, None)
            memory.missing[object_id] = 0
    confirmed = sorted(memory.demonstration_events,
                       key=lambda event: (event['first_missing_step'], event['demonstration_id']))
    memory.order = [event['demonstration_id'] for event in confirmed]
    ties = [left['first_missing_step'] for left, right in zip(confirmed, confirmed[1:])
            if left['first_missing_step'] == right['first_missing_step']]
    memory.demonstration_order_evidence = {
        'method': 'persistent_rgb_source_departure',
        'minimum_persistent_frames': minimum_frames,
        'event_count': len(confirmed), 'ambiguous_onset_steps': sorted(set(ties)),
        'order_verified_from_visual_events': not ties,
        'grasp_order_directly_observed': False,
        'holding_verified': False,
        'private_task_order_used': False}
    return not ties
