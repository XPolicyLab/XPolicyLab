"""Episode-local original scene RGB and bounded, explicit historical reads."""
from __future__ import annotations
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image

CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')


class SceneFrames:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.index=self.directory/'frames.jsonl'
        self.frames=[json.loads(line) for line in self.index.read_text(encoding="utf-8").splitlines()] if self.index.exists() else []
        links=self.directory/'entity-links.jsonl'
        self.entity_links={r['frame_id']:r['entity_state_refs'] for r in (json.loads(line) for line in links.read_text(encoding="utf-8").splitlines())} if links.exists() else {}

    def capture(self,vision,*,native_step,sim_time,calibrations=None,entity_ids=()):
        frame_id='scene_'+uuid.uuid4().hex
        root=self.directory/frame_id;root.mkdir()
        row={'frame_id':frame_id,'native_step':native_step,'sim_time':sim_time,'capture_time':time.time(),
             'entity_ids':list(entity_ids),'cameras':{}}
        for camera in CAMERAS:
            value=vision.get(camera,{}).get('color')
            if value is None:
                row['cameras'][camera]={'status':'missing'};continue
            image=Image.fromarray(np.asarray(value,dtype=np.uint8)).convert('RGB')
            path=root/(camera+'.png');image.save(path)
            row['cameras'][camera]={'status':'ok','artifact':str(path.relative_to(self.directory)),
                'size':list(image.size),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'calibration':copy.deepcopy((calibrations or {}).get(camera))}
        self.frames.append(row)
        with self.index.open('a', encoding="utf-8") as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        return copy.deepcopy(row)

    def query(self,*,native_steps=None,cameras=None,k=4,offset=0,limit=6,max_bytes=8*1024*1024,frame_ids=None,now=None):
        if type(k) is not int or k<1:raise ValueError('k must be a positive integer')
        if type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=32:raise ValueError('invalid frame pagination')
        if type(max_bytes) is not int or not 1024<=max_bytes<=16*1024*1024:raise ValueError('max_bytes must be 1024..16777216')
        cameras=CAMERAS if not cameras else cameras
        if any(c not in CAMERAS for c in cameras) or len(cameras)!=len(set(cameras)):raise ValueError('invalid cameras')
        rows=[r for r in self.frames if (not native_steps or r['native_step'] in native_steps) and
              (not frame_ids or r['frame_id'] in frame_ids) and (now is None or r['sim_time']<=now)]
        selected=[(r,c) for r in rows for c in cameras]
        missing=[{'native_step':s,'status':'missing'} for s in native_steps or [] if not any(r['native_step']==s for r in rows)]
        missing += [{'frame_id':f,'status':'missing'} for f in frame_ids or [] if not any(r['frame_id']==f for r in rows)]
        result={'operation':'frames','query_schema':'oeh.stmem.frames.v1','images':[], 'missing_images':missing,
                'total_images':len(selected),'offset_unit':'images','next_offset':None,'encoded_image_bytes':0}
        cursor=offset
        for row,camera in selected[offset:offset+limit]:
            source=row['cameras'][camera];metadata={key:row[key] for key in ('frame_id','native_step','sim_time','capture_time')}
            metadata.update(camera=camera,k=k,calibration=source.get('calibration'),entity_state_refs=copy.deepcopy(self.entity_links.get(row['frame_id'],[])))
            if source['status']!='ok' or not (self.directory/source['artifact']).is_file():
                result['missing_images'].append({**metadata,'status':'missing'});cursor+=1;continue
            with Image.open(self.directory/source['artifact']) as original:
                image=original.convert('RGB')
                if k!=1:image=image.resize(tuple(max(1,n//k) for n in image.size),Image.Resampling.LANCZOS)
                buffer=io.BytesIO();image.save(buffer,format='PNG');data=buffer.getvalue()
            encoded=base64.b64encode(data).decode('ascii')
            if result['encoded_image_bytes']+len(encoded)>max_bytes:
                result['budget_status']='image_exceeds_budget' if not result['images'] else 'page_byte_limit'
                result['required_next_image_bytes']=len(encoded);break
            sha=hashlib.sha256(data).hexdigest();path=self.directory/'query-images'/(sha+'.png');path.parent.mkdir(exist_ok=True);path.write_bytes(data)
            metadata.update(size=list(image.size),original_size=source['size'],sha256=sha,original_sha256=source['sha256'],
                            artifact=str(path.relative_to(self.directory)),status='ok')
            result['images'].append(metadata)
            result.setdefault('_image_items',[]).append({'type':'input_image','image_url':'data:image/png;base64,'+encoded})
            result['encoded_image_bytes']+=len(encoded);cursor+=1
        result['next_offset']=cursor if cursor<len(selected) else None
        result['status']='ok' if result['images'] else 'missing' if not selected else result.get('budget_status','missing')
        return result

    def calibration(self, *, frame_id, cameras=None, now=None):
        cameras = CAMERAS if not cameras else cameras
        if any(c not in CAMERAS for c in cameras) or len(cameras) != len(set(cameras)):
            raise ValueError('invalid cameras')
        row = next((r for r in self.frames if r['frame_id'] == frame_id and (now is None or r['sim_time'] <= now)), None)
        result = {'operation': 'frame_calibration', 'frame_id': frame_id, 'status': 'missing', 'cameras': []}
        if row is None:
            return result
        result.update(status='ok', native_step=row['native_step'], sim_time=row['sim_time'], image_space='original_pixels')
        for camera in cameras:
            source = row['cameras'][camera]
            calibration = source.get('calibration')
            result['cameras'].append({'camera': camera, 'original_size': copy.deepcopy(source.get('size')),
                'status': 'ok' if calibration is not None else 'unknown', 'calibration': copy.deepcopy(calibration)})
        return result

    def link_entities(self,frame_id,context):
        refs=[{'entity_id':e['entity_id'],'state_time':e.get('time'),'evidence':e.get('evidence'),
            'support_observation_ids':e.get('tracking_support',{}).get('observation_ids',[])} for e in context.get('entities',[])]
        self.entity_links[frame_id]=copy.deepcopy(refs)
        with (self.directory/'entity-links.jsonl').open('a', encoding="utf-8") as f:
            f.write(json.dumps({'frame_id':frame_id,'entity_state_refs':refs,'linked_at':time.time()})+'\n')

    def delivered(self,frame_id,call_id):
        with (self.directory/'deliveries.jsonl').open('a', encoding="utf-8") as f:
            f.write(json.dumps({'frame_id':frame_id,'call_id':call_id,'recorded_at':time.time()})+'\n')
