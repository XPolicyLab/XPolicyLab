DEFAULT=dict(mode="episode",missing="episode",invalid="exclude",gaps="exclude")
VERSION="crwm.episode_language.v1"
def language_policy(source,override=None):
    if source.get("language",{}) or override:raise ValueError("This recipe uses episode instructions only")
    return dict(DEFAULT)
def episode_instruction(ep):
    tasks=ep.get("tasks") or []
    return tasks[0] if tasks else ""
def training_intervals(ep,policy):
    for start,end in ep["intervals"]:
        yield dict(ep,instruction=episode_instruction(ep),language_segment_id=-1,language_status="episode",supervision_end=end),start,end
