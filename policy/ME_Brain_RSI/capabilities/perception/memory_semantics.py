"""Keep paired RGB identity evidence consistent with ordinary grasp height."""


def annotate_source_grasps(pairs, evidence):
    """Use the matched counterpart and paired keypads without rejecting a source."""
    for assignment in evidence.get('assignments',[]):
        source = pairs[assignment['demonstration_id']]
        paired_keypads = bool(assignment.get('source_keypad') and assignment.get('demonstration_keypad'))
        phone = bool(source.get('appearance_family') == 'phone'
                     or (assignment.get('demonstration_family') == 'phone'
                         and assignment.get('demonstration_keypad')) or paired_keypads)
        source['sorting_phone_like'] = phone
        source['sorting_toy_like'] = bool(assignment.get('source_toy_face')
                                         and assignment.get('demonstration_toy_face'))
        families=(assignment.get('source_family'),assignment.get('demonstration_family'))
        unbanded=(not phone and not assignment.get('source_hollow_band')
                  and not assignment.get('demonstration_hollow_band')
                  and all(family in ('watch','action_camera') for family in families))
        source['sorting_unbanded_dark_body']=unbanded
        source['sorting_camera_like']=bool(unbanded and 'action_camera' in families)
        assignment['unbanded_dark_body_grasp']=unbanded
        assignment['grasp_phone_like'] = phone
        assignment['paired_keypad_grasp_evidence'] = paired_keypads


def pinch_height_for_source(source, proposed_height):
    """Prevent bright phone bodies from taking the tall-object height override."""
    if source.get('sorting_phone_like') or source.get('appearance_family') == 'phone':
        return .020
    if source.get('appearance_family') == 'toy_car':
        return .021
    return proposed_height
