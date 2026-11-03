"""Changed entity current states only; SpatialMemory owns all history and metadata."""
import copy
from .input_context import rounded


class SpatialEvents:
    """Send only current states of entities whose public information changed.

    Frame ticks, worker status and model-selected working sets are local metadata.
    The stream never replays history or sends a full scene on an idle frame.
    """
    def __init__(self):
        self.states={}

    @staticmethod
    def state(e):
        fields=('entity_id','semantic_annotation','label','description','bbox_2d','image_size','frame_id',
                'position_uncertainty_m','orientation_ambiguous','visibility','motion',
                'velocity_m_s','angular_velocity_rad_s','lifecycle','pose_status','bbox_status','dimensions_observed_at',
                'instruction_mentioned','interactable','avoid_interaction')
        state={k:copy.deepcopy(e[k]) for k in fields if k in e}
        box=e.get('box')
        if isinstance(box,dict):
            state['box']={k:copy.deepcopy(box[k]) for k in ('center','R','dimensions') if k in box}
        elif box is not None:
            state['box']=copy.deepcopy(box) # Preserve a literal deleted tombstone.
        return rounded(state)

    @staticmethod
    def tombstones(state,previous):
        for key,old in previous.items():
            if key not in state:state[key]='deleted'
            elif isinstance(state[key],dict) and isinstance(old,dict):
                SpatialEvents.tombstones(state[key],old)
        return state

    def notify(self,snapshot,working_set=None,frame_id=None):
        updated=[]
        for entity in snapshot.get('entities',[]):
            eid=entity['entity_id'];previous=self.states.get(eid)
            state=self.tombstones(self.state(entity),previous or {})
            if previous!=state:
                # Current timestamp accompanies an actual update, but a clock tick
                # alone never triggers an update or replays older measurements.
                current=copy.deepcopy(state)
                if entity.get("evidence") is not None:current["evidence"]=entity["evidence"]
                if entity.get('time') is not None:current['observed_at']=entity['time']
                updated.append(current)
            self.states[eid]=state
        # Partial/stale responses do not imply that omitted entities were deleted.
        return {'entities':updated} if updated else None


def validate_working_set(value,snapshot):
    if not isinstance(value,dict) or set(value)!={'phase','entity_ids','reason'}:
        raise ValueError('stmem_working_set requires phase, entity_ids, reason on every response')
    if not isinstance(value['phase'],str) or not value['phase'].strip() or len(value['phase'])>240 or not isinstance(value['reason'],str) or len(value['reason'])>800:
        raise ValueError('invalid working-set phase/reason')
    ids=value['entity_ids'];known={e['entity_id'] for e in snapshot.get('entities',[])}
    if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(ids)>32 or len(set(ids))!=len(ids) or any(i not in known for i in ids):
        raise ValueError('working set must contain unique known entity IDs; use [] and query unresolved objects')
    return copy.deepcopy(value)
