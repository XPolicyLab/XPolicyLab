"""VLM silhouette judgments, reused only for identical visual evidence."""
import copy
import hashlib
from pathlib import Path

COMPLETENESS_PROMPT=(
    'Assess visible object completeness using exactly one category: '
    'complete = the whole visible silhouette is present, with no material occlusion or crop truncation; '
    'partial = mostly complete, with some occlusion, sufficient for tracking but not a new full-size template; '
    'occluded = only fragments or a heavily hidden/truncated object. '
    'Hidden back surfaces do not make an otherwise complete silhouette partial. '
    'Use unknown if evidence is insufficient. Return completeness and a short description of visible shape, '
    'occlusion and truncation; do not infer physical pose, task progress or success. ')


def visual_key(obs):
    path=Path(obs.get('bbox_crop_path') or '__missing__')
    if not path.is_file():return None
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    mask=Path(obs.get('mask_path') or '__missing__')
    bbox=obs.get('bbox_2d') or [0,0,0,0]
    return (obs.get('camera'),obs.get('label','').casefold(),bbox[2]-bbox[0],bbox[3]-bbox[1],digest,hashlib.sha256(mask.read_bytes()).hexdigest() if mask.is_file() else None,
            bool(obs.get('boundary_truncated',False)))


def record(memory,obs,item,source,now):
    category=item.get('completeness')
    description=item.get('description')
    if category not in {'complete','partial','occluded','unknown'} or not isinstance(description,str) or not description.strip():
        category='unknown'
    if category=='complete' and obs.get('boundary_truncated'):category='occluded'
    row={'completeness':category,'description':description[:240] if isinstance(description,str) else '',
         'observation_id':obs.get('observation_id'),'observed_at':obs['time'],'known_at':now,'source':copy.deepcopy(source)}
    key=visual_key(obs)
    if key is not None:memory.completeness_reviews[key]=row
    memory.log('visual_completeness_review',row)
    return row


def apply(memory,obs):
    key=visual_key(obs);row=memory.completeness_reviews.get(key) if key else None
    if row and obs.get('completeness','unknown')=='unknown':
        obs['completeness']=row['completeness']
        obs['completeness_evidence']=copy.deepcopy(row)
        obs['completeness_source']='vlm_identical_visual_evidence'
    return row
