"""One asynchronous GPT bbox-description attempt per new entity.

Image input is the immutable first crop, never periodic segmentation frames or
historical ReID exemplars. Geometry/visibility continue updating independently.
"""
from __future__ import annotations
import copy
import hashlib
import io
from PIL import Image
import json
from pathlib import Path
import queue
import threading
import time


class EntityDescriptions:
    def __init__(self, memory, cfg, directory, client_factory):
        self.memory, self.cfg, self.directory = memory, cfg, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.client_factory, self.client = client_factory, None
        self.results = queue.Queue(maxsize=1)
        self.inflight = False
        self.closed = False
        self.pending = []
        self.image_owners = {}
        self.followers = {}
        self.cache = {}

    def register(self, entity_id, observation):
        entity = self.memory.entities[entity_id]
        if entity.get('description_request') is not None:
            return
        path = Path(observation['bbox_crop_path'])
        from .stmem_images import semantic_object_input
        input_path,input_kind=semantic_object_input(observation,self.directory,'entity-'+entity_id)
        entity['description_request'] = {
            'entity_id':entity_id, 'detector_label':observation['label'],
            'observation_id':observation['observation_id'], 'time':observation['time'],
            'sample_index':observation['sample_index'], 'bbox_2d':observation['bbox_2d'],
            'camera':observation.get('camera'), 'crop_path':str(input_path),'image_kind':input_kind,
            'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'prior_description':entity.get('description_record',{}).get('text',observation.get('description','')),
            'prior_description_source':entity.get('description_record',{}).get('source',observation.get('description_source')),
            'bbox_crop_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        entity['description_status']='not_requested'
        request = entity['description_request']
        with Image.open(input_path) as source:
            sent = source.convert('RGB');sent.thumbnail((32,32))
            buf = io.BytesIO();sent.save(buf,format='PNG')
        request['sent_bbox_sha256'] = hashlib.sha256(buf.getvalue()).hexdigest()
        key = self.cache_key(request)
        if key in self.cache:
            item, source, error = self.cache[key]
            self._cached(request,item,source,observation['time'],error)
            return
        if key in self.image_owners:
            entity['description_status'] = 'waiting_same_image'
            self.followers.setdefault(key,[]).append(copy.deepcopy(request))
            self.memory.archive_summary(entity)
            return
        self.image_owners[key] = entity_id
        if len(self.pending) >= self.cfg.get('description_queue_limit',32):
            entity['description_status'] = 'deferred_capacity'
        else:
            entity['description_status'] = 'queued'
            self.pending.append(entity_id)
        self.memory.archive_summary(entity)

    def cache_key(self, request):
        return (request['sent_bbox_sha256'], self.memory.instruction,
                request['detector_label'], request.get('prior_description',''))

    def _cached(self, request, item, source, now, error):
        item = copy.deepcopy(item)
        if item:
            item['entity_id'] = request['entity_id']
            item['instruction_label'] = None  # Referential phrases do not transfer between entities.
            if source.get('detector_label')!=request['detector_label']:
                item['label_valid'] = None  # The cached call did not validate this new detector label.
        self.memory.apply_description(request['entity_id'],item,
            {**source, **request, 'kind':'cached_identical_bbox',
             'cached_from_entity':source.get('entity_id')},now,error=error)

    def poll(self, now):
        try:
            batch, response, source, error = self.results.get_nowait()
        except queue.Empty:
            return
        self.inflight = False
        items = (response or {}).get('objects', [])
        if not isinstance(items, list):items = []
        for request in batch:
            eid = request['entity_id']
            matches = [i for i in items if isinstance(i, dict) and i.get('entity_id') == eid]
            item = matches[0] if len(matches) == 1 else None
            failure = error or ('missing_or_duplicate_entity_result' if item is None else None)
            provenance = {**source,**request}
            ok = self.memory.apply_description(eid,item,provenance,now,error=failure)
            if not ok:failure = failure or self.memory.entities[eid].get('description_error')
            key = self.cache_key(request)
            self.cache[key] = (copy.deepcopy(item),copy.deepcopy(provenance),failure)
            for follower in self.followers.pop(key,[]):
                self._cached(follower,item,provenance,now,failure)

    def pump(self):
        if self.closed or self.inflight:return
        for eid,entity in self.memory.entities.items():
            if len(self.pending)>=self.cfg.get('description_queue_limit',32):break
            if entity.get('description_status')=='deferred_capacity':
                entity['description_status']='queued';self.pending.append(eid)
        if not self.pending:return
        n = self.cfg.get('description_batch_size', 8)
        ids, self.pending = self.pending[:n], self.pending[n:]
        batch = [copy.deepcopy(self.memory.entities[e]['description_request']) for e in ids]
        for eid in ids:
            self.memory.entities[eid]['description_status'] = 'inflight'
            self.memory.archive_summary(self.memory.entities[eid])
        instruction = self.memory.instruction
        timeout = self.cfg.get('description_timeout_s', 90.)
        self.inflight = True
        def run():
            response, source, error = None, {}, None
            try:
                if self.closed:raise RuntimeError('description_worker_closed')
                if self.client is None:self.client = self.client_factory()
                if self.closed:raise RuntimeError('description_worker_closed')
                prompt = ('Preserve the prior short description of the object inside each indexed RGB bbox crop, including readable numbers, colors, shapes and markings. Correct only an explicitly instruction-mentioned attribute when the RGB supports the correction; preserve unrelated observed attributes. If no prior description exists, describe visible appearance briefly. If the instruction contradicts the RGB, retain the observation and mark uncertainty instead of making it fit. Do not invent hidden surfaces, digits, task progress or object roles. Validate its detector_label; '
                    'For segmented_object previews describe only foreground appearance; uniform mask background is not object material or table colour. '
                    'if incorrect give a short corrected label; if uncertain label_valid=null and keep it. '
                    'Give a short visible appearance/shape description, at most 180 characters. '
                    'Include a clearly readable printed digit or number literally in description; if unreadable say unknown, never guess. '
                    'If a prior description exists, return description_corrections as [] or scoped patches {attribute:number/color/shape/marking,old_text,new_text,instruction_phrase,supported_by_rgb:true}. Only instruction-backed RGB-supported patches change prior appearance; never replace a physical attribute merely to match an instruction. '
                    'Do not infer task progress, identity from prior images, or hidden geometry. '
                    'If instruction_label is unambiguous, copy an exact phrase from the instruction, else null. '
                    'Each entity image is sent once: return every supplied entity_id exactly once. '
                    'Return {"objects":[{"entity_id":"...","label_valid":true,"label":"...",'
                    '"description":"...","instruction_label":null}]}. '
                    'Instruction: '+json.dumps(instruction)+' Indexed objects: '+json.dumps([
                        {k:r[k] for k in ('entity_id','detector_label','prior_description','prior_description_source','bbox_2d','camera','image_kind')} for r in batch]))
                from .stmem_llm_contract import ask
                response = ask(self.client,'first_entity_description',','.join(ids),prompt, [r['crop_path'] for r in batch],
                    time.monotonic()+timeout+5, downsample_k=1, image_max_edge=32)
                source = {'kind':'vlm_first_entity_bbox', 'model':self.client.cfg.get('model'),
                    'call_id':getattr(self.client,'last_status',{}).get('call_id'),
                    'image_max_edge':32, 'image_downsample_k':1}
            except Exception as exc:
                error = type(exc).__name__
                source = {'kind':'vlm_first_entity_bbox','diagnostic':getattr(self.client,'last_status',{})}
            self.results.put((batch, response, source, error))
        threading.Thread(target=run,name='oeh-entity-description',daemon=True).start()

    def close(self, now):
        self.poll(now)
        self.closed = True
        for eid, entity in self.memory.entities.items():
            if entity.get('description_status') in {'queued','inflight','waiting_same_image','deferred_capacity'}:
                entity['description_status'] = 'interrupted'
                self.memory.archive_summary(entity)
        client = getattr(self.client,'client',None)
        if client is not None and hasattr(client,'close'):client.close()
