"""Pending observations are not entities. Confirmation is a separate, auditable gate."""
import copy
import uuid
from pathlib import Path
import shutil
import json
import numpy as np


def overlap(a,b):
    if a is None or b is None:return 0.
    a,b=np.asarray(a,float),np.asarray(b,float)
    area_a=float(np.prod(a[2:]-a[:2]));area_b=float(np.prod(b[2:]-b[:2]))
    if min(area_a,area_b)<=0:return 0.
    inter=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
    # IoU avoids treating a sliver contained in a card as a complete new footprint.
    return inter/max(1.,area_a+area_b-inter)


class IdentityLifecycle:
    def __init__(self,memory,threshold=.6,pending_limit=128):
        self.memory,self.threshold,self.pending_limit=memory,threshold,pending_limit
        self.pending={}
        self.tentative={}

    def candidates(self,obs):
        result=[]
        for eid,entity in self.memory.entities.items():
            old=entity['best_observation'];state=entity['state']
            if old.get('camera')!=obs.get('camera') or state.get('image_size')!=obs.get('image_size') or state['frame_id']!=obs['frame_id']:continue
            score=overlap(obs.get('bbox_2d'),state.get('bbox_2d'))
            continuity=self.memory.image_continuity(obs,entity)
            from .stmem import similarity
            appearance=max((similarity(obs.get('feature',[]),p.get('feature',[])) for p in entity['patterns']),default=0.)
            if score>=.35 or continuity is not None and continuity>=.6 or entity.get('lifecycle')=='dormant' and appearance>=.85:
                result.append({'entity_id':eid,'bbox_similarity':score,
                    'label_agrees':obs['label'].casefold() in entity['aliases'],
                    'tier':entity.get('lifecycle','active')})
        return sorted(result,key=lambda x:-x['bbox_similarity'])

    def stage(self,obs,reason,now):
        compatible=[p for p in self.pending.values() if p['status'] not in {'consumed','expired'} and
            p['latest'].get('camera')==obs.get('camera') and p['latest'].get('image_size')==obs.get('image_size') and
            p['latest']['frame_id']==obs['frame_id'] and overlap(p['latest'].get('bbox_2d'),obs.get('bbox_2d'))>=.6]
        compatible.sort(key=lambda p:-overlap(p['latest']['bbox_2d'],obs['bbox_2d']))
        if len(compatible)>1 and overlap(compatible[0]['latest']['bbox_2d'],obs['bbox_2d'])-overlap(compatible[1]['latest']['bbox_2d'],obs['bbox_2d'])<.12:
            compatible=[] # Keep ambiguity explicit, do not transfer confirmation.
        if compatible:p=compatible[0]
        else:
            if len([p for p in self.pending.values() if p['status'] not in {'consumed','expired'}])>=self.pending_limit:
                self.memory.log('pending_capacity',{'observation':obs,'reason':reason});return None
            key='proposal_'+uuid.uuid4().hex[:12]
            p={'candidate_id':key,'first':copy.deepcopy(obs),'latest':copy.deepcopy(obs),
               'status':'proposed','observations':0,'review':None}
            self.pending[key]=p
        p['last_match_iou']=overlap(p['latest'].get('bbox_2d'),obs.get('bbox_2d'))
        review=p.get('review')
        from .stmem_completeness import visual_key
        if review and not review['accepted'] and review['item'].get('completeness')!='complete' and len(p.get('review_history',[]))<2:
            if visual_key(obs)!=visual_key(p['first']) and not obs.get('boundary_truncated'):
                p.setdefault('review_history',[]).append(review)
                p['review']=None;p['review_attempted']=False;p['status']='proposed'
                self.memory.log('confirmation_new_evidence',{'candidate_id':p['candidate_id'],'reason':'changed_visual_evidence','time':now})
        p['latest']=copy.deepcopy(obs);p['observations']+=1;p['last_time']=now
        p['reason']=reason;p['existing_candidates']=self.candidates(obs)
        self.memory.log('tentative_observation',{'candidate_id':p['candidate_id'],'observation':obs,'reason':reason,'existing_candidates':p['existing_candidates']})
        return p

    def confirm(self,candidate_id,item,source,now):
        p=self.pending.get(candidate_id)
        if p is None or p.get('review') is not None:return False
        item=copy.deepcopy(item)
        # Legacy responses may contain uncalibrated GPT geometry scores. Ignore them.
        item.pop('geometry_confidence',None)
        for key in ('instruction_mentioned','interactable','avoid_interaction'):
            if key not in item or item[key] is not None and type(item[key]) is not bool:item[key]=None
        description=item.get('description')
        semantic_valid=isinstance(description,str) and bool(description.strip()) and (item.get('label_valid') is None or type(item.get('label_valid')) is bool)
        accepted=item.get('accepted') is True and item.get('completeness')=='complete' and semantic_valid
        from .stmem_completeness import record
        record(self.memory,p['first'],item,source,now)
        p['review']={'item':item,'source':source,'available_at':now,'accepted':accepted}
        p['status']='confirmed_observation' if accepted else 'rejected_or_incomplete'
        self.memory.log('observation_confirmation',{'candidate_id':candidate_id,**p['review']})
        return accepted

    def admit(self,p,obs):
        if not p or p['status']!='confirmed_observation' or obs.get('completeness')=='partial' or obs['confidence']<self.threshold:return False
        reviewed=p['first']
        # Confirmation belongs to the actual reviewed crop; no transfer across label changes or partial returns.
        if obs['label'].casefold()!=reviewed['label'].casefold():return False
        old,new=reviewed.get('bbox_2d'),obs.get('bbox_2d')
        if old is None or new is None:return False
        ratio=((new[2]-new[0])*(new[3]-new[1]))/max(1.,(old[2]-old[0])*(old[3]-old[1]))
        continuous=(p['observations']>=3 and p.get('last_match_iou',0)>=.85 and .85<=ratio<=1.15)
        if overlap(new,old)<.85 and not continuous:return False
        from .stmem import similarity
        if similarity(obs.get('feature',[]),reviewed.get('feature',[])) < .9:return False
        if self.candidates(obs):return False
        obs['completeness']='complete'
        return True

    def consume(self,p,entity,now):
        p['status']='consumed';p['entity_id']=entity['entity_id']
        entity['identity_status']='confirmed';entity['confirmation']=copy.deepcopy(p['review'])
        item=p['review']['item'];source=p['review']['source']
        entity['description_request']={'candidate_id':p['candidate_id'],'source':source}
        root=self.memory.directory/'entities'/entity['entity_id'];directory=root/'confirmation'
        directory.mkdir(parents=True,exist_ok=True);assets={}
        for key,name in [('bbox_crop_path','first-bbox.png'),('crop_path','first-segmented.png'),('mask_path','first-mask.png'),('geometry_path','first-geometry.npz'),('calibration_path','first-calibration.json')]:
            path=Path(p['first'].get(key) or '__missing__')
            if path.is_file():
                shutil.copy2(path,directory/name);assets[key]=str((directory/name).relative_to(root))
        entity['first_complete_observation']={'observation_id':p['first'].get('observation_id'),
            'sample_index':p['first'].get('sample_index'),'observed_time':p['first']['time'],
            'confirmed_at':p['review']['available_at'],'entity_created_at':now,'assets':assets}
        from .stmem_anchors import make_anchor
        entity['initial_anchor']=make_anchor(p['first'],known_at=p['review']['available_at'],assets=assets)
        (directory/'metadata.json').write_text(json.dumps(entity['first_complete_observation'],indent=2)+'\n', encoding="utf-8")
        entity['description_status']='not_requested'
        self.memory.apply_description(entity['entity_id'],item,source,now)
        self.memory.archive_summary(entity)

    def expire(self,now):
        for p in self.pending.values():
            if p['status'] not in {'consumed','expired'} and now-p['last_time']>30:
                p['status']='expired';self.memory.log('tentative_expired',{'candidate_id':p['candidate_id'],'time':now})

    def nearby(self,obs,exclude=()):
        choices=[]
        for eid,e in self.memory.entities.items():
            if eid in exclude:continue
            old=e.get('tracking_current') or e['state'];ref=e['best_observation']
            if old['frame_id']!=obs['frame_id'] or ref.get('camera')!=obs.get('camera') or old.get('image_size')!=obs.get('image_size'):continue
            prediction=e.get('tracking_prediction',{})
            anchor=prediction['bbox_2d'] if prediction.get('time')==obs['time'] and prediction.get('camera')==obs.get('camera') else old.get('bbox_2d')
            a,b=obs.get('bbox_2d'),anchor
            if a is None or b is None:continue
            a,b=np.asarray(a,float),np.asarray(b,float)
            intersection=float(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod())
            containment=intersection/max(1.,float(np.prod(a[2:]-a[:2])))
            distance=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2))
            scale=max(1.,float(np.linalg.norm(b[2:]-b[:2])))
            if intersection>0 or distance<scale:
                choices.append({'entity_id':eid,'score':max(overlap(a,b),containment)*.8+.2*max(0,1-distance/scale),
                    'label_agrees':obs['label'].casefold() in e['aliases']})
        return sorted(choices,key=lambda c:-c['score'])

    def full_for(self,obs,eid):
        if eid is None:return False
        if obs.get('completeness')=='complete':return True
        if obs.get('completeness') in {'partial','occluded'}:return False
        from .stmem_completeness import apply
        apply(self.memory,obs)
        return obs.get('completeness')=='complete'

    def attach_tentative(self,obs,complete_ids,now,forced_owner=None):
        choices=[c for c in self.nearby(obs,complete_ids) if c['label_agrees']]
        if forced_owner is not None:choices=[{'entity_id':forced_owner,'score':1.,'label_agrees':True,'source':'vlm_reidentification'}]
        if not choices or len(choices)>1 and choices[0]['score']-choices[1]['score']<.12:
            self.memory.log('invalid_tentative',{'observation':obs,'reason':'no_unique_unobserved_owner','candidates':choices})
            return None
        eid=choices[0]['entity_id']
        row={'entity_id':eid,'observation_id':obs['observation_id'],'time':now,'sample_index':self.memory.sample_index,
             'bbox_2d':copy.deepcopy(obs.get('bbox_2d')),'label':obs['label'],'visibility':'occluded',
             'camera':obs.get('camera'),'frame_id':obs['frame_id'],'image_size':copy.deepcopy(obs.get('image_size')),'completeness':'partial',
             'status':'active','association_candidates':choices}
        root=self.memory.directory/'entities'/eid/'tentative'/obs['observation_id'];root.mkdir(parents=True,exist_ok=True)
        row['assets']={}
        for key,name in [('bbox_crop_path','bbox.png'),('crop_path','segmented.png'),('mask_path','mask.png')]:
            path=Path(obs.get(key) or '__missing__')
            if path.is_file():
                shutil.copy2(path,root/name);row['assets'][key]=str((root/name).relative_to(self.memory.directory/'entities'/eid))
        row['metadata_path']=str(root/'metadata.json')
        (root/'metadata.json').write_text(json.dumps(row,indent=2)+'\n', encoding="utf-8")
        self.tentative.setdefault(eid,[]).append(row)
        self.memory.log('tentative_tracking',row)
        entity=self.memory.entities[eid]
        previous=entity.get('tracking_current')
        support=[row]
        if previous and previous['time']==now:
            support=[r for r in self.tentative[eid] if r['time']==now and r['camera']==row['camera'] and r['status']=='active']
        boxes=[r['bbox_2d'] for r in support if r.get('bbox_2d') is not None]
        bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
        # A partial point-cloud centroid is not an estimate of the rigid object's center.
        # Until a pose estimator supplies explicit evidence, withdraw current pose.
        entity['tracking_current']={'time':now,'frame_id':obs['frame_id'],'camera':obs.get('camera'),
            'image_size':copy.deepcopy(obs.get('image_size')),'bbox_2d':bbox,
            'box':{'center':None,'R':None,'dimensions':copy.deepcopy(entity.get('last_complete',{}).get('box',{}).get('dimensions'))},
            'pose_status':{'center':'unknown','R':'unknown','dimensions':'last_complete' if entity.get('last_complete') else 'unknown'},
            'visibility':'occluded','observed_or_predicted':'partial_observation','motion':'unknown',
            'velocity_m_s':None,'angular_velocity_rad_s':None,'yaw_rad':None,
            'orientation_ambiguous':True,'evidence':obs['observation_id']}
        entity['tracking_support']={'time':now,'kind':'tentative_bbox','observation_ids':[r['observation_id'] for r in support],
            'camera':obs.get('camera'),'association_candidates':copy.deepcopy(choices)}
        history=entity.setdefault('tracking_history',[])
        if history and history[-1]['time']==now and history[-1].get('camera')==obs.get('camera'):history[-1]=copy.deepcopy(entity['tracking_current'])
        else:history.append(copy.deepcopy(entity['tracking_current']))
        entity['lifecycle']='active'
        self.memory.log('tracking_anchor_updated',{'entity_id':eid,'current':entity['tracking_current'],'support':entity['tracking_support']})
        return eid

    def has_tentative(self,eid):
        entity=self.memory.entities[eid]
        latest=entity['state']['time']
        return any(r['status']=='active' and r['time']>=latest for r in self.tentative.get(eid,[]))

    def merge_tentatives(self,eid,now):
        from .stmem_images import capture_key,combine_capture
        entity=self.memory.entities[eid];groups={}
        from .stmem_images import capture_key
        active=[r for r in self.tentative.get(eid,[]) if r['status']=='active' and r['time']<=now]
        captures={}
        for row in active:captures.setdefault(capture_key(row),[]).append(row)
        anchor=entity['state'];expected_camera=entity['best_observation'].get('camera')
        accepted=[]
        for key,rows in sorted(captures.items(),key=lambda item:item[0][0],reverse=True):
            sample=rows[0]
            if sample['camera']!=expected_camera or sample['frame_id']!=anchor['frame_id'] or sample.get('image_size')!=anchor.get('image_size'):continue
            boxes=[r['bbox_2d'] for r in rows]
            bounds=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
            a,b=np.asarray(bounds,float),np.asarray(anchor['bbox_2d'],float)
            distance=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
            gate=max(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2]))
            if anchor['time']-sample['time']>self.memory.max_gap_s or distance>gate:break
            accepted.extend(rows);anchor={**sample,'bbox_2d':bounds}
        for row in accepted:
            # Only the evidence-connected suffix is confirmed; older/conflicting segments stay pending.
            row['status']='deleted';row['merged_at']=now
            row['merge_evidence']={'kind':'continuous_same_camera_suffix','confirmed_observation_id':entity['state']['evidence']}
            Path(row['metadata_path']).write_text(json.dumps(row,indent=2)+'\n', encoding="utf-8")
            groups.setdefault(capture_key(row),[]).append(row)
            self.memory.log('tentative_merged',row)
        root=self.memory.directory/'entities'/eid
        for rows in groups.values():
            merged=combine_capture(root,rows)
            entity.setdefault('tracking_observations',[]).append(merged)
            if len(rows)>1:self.memory.log('tentative_capture_bbox_union',merged)
