"""Propose instruction-named image regions with a fixed public GroundingDINO model."""
from functools import lru_cache
import inspect
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image


MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
MODEL_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
MODEL_SHA256 = '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
DEFAULT_MODEL_PATH = str(Path(__policy_root__) / 'model_assets/classification/grounding-dino-tiny')
ALIASES = {'watch':'wristwatch', 'car':'toy car', 'wooden_toy':'wooden animal figurine',
           'pepper':'bell pepper', 'chocolate_bar':'wrapped chocolate bar'}
NEGATIVES = ('basket', 'robot_arm', 'robot_gripper')


def instruction_categories(instruction):
    """Read public category names without consulting the scene or task state."""
    matches = re.findall(r'(?:Put\s+|,\s*(?:and\s+)?)(.+?) objects into the (left|middle|right) basket',
                         str(instruction), re.I)
    return tuple(name.strip().replace(' ', '_') for name, side in matches)


def _iou(first, second):
    left = np.maximum(first[:2], second[:2])
    right = np.minimum(first[2:], second[2:])
    overlap = float(np.prod(np.maximum(0., right-left)))
    area = float(np.prod(first[2:]-first[:2])+np.prod(second[2:]-second[:2]))
    return overlap/max(area-overlap, 1e-8)


def _merge_query_duplicates(proposals):
    """Keep one proposal per overlapping query region while retaining raw labels."""
    selected=[]
    for proposal in sorted(proposals,key=lambda row:-row['dino_score']):
        box=np.asarray(proposal['box_xyxy'])
        if any(proposal['category']==other['category'] and
               _iou(box,np.asarray(other['box_xyxy']))>.55 for other in selected):
            continue
        selected.append(proposal)
    return selected


class GroundingProposer:
    """Load resident CPU weights and expose RGB proposals independently of action execution."""

    def __init__(self, model_dir=None, threads=4, threshold=.13, text_threshold=.12):
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch=torch
        self.threads=int(threads)
        torch.set_num_threads(self.threads)
        self.folder=Path(model_dir or os.environ.get('CLASSIFY_DINO_PATH',DEFAULT_MODEL_PATH))
        start=time.perf_counter()
        self.processor=AutoProcessor.from_pretrained(str(self.folder),local_files_only=True,
                                                     trust_remote_code=False)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(
            str(self.folder),local_files_only=True,trust_remote_code=False,use_safetensors=True,
            disable_custom_kernels=True).to('cpu').eval()
        self.load_seconds=time.perf_counter()-start
        self.threshold=float(threshold)
        self.text_threshold=float(text_threshold)
        method=self.processor.post_process_grounded_object_detection
        self.threshold_key='box_threshold' if 'box_threshold' in inspect.signature(method).parameters else 'threshold'
        self.environment=dict(torch=torch.__version__,transformers=transformers.__version__,
            device='cpu',threads=self.threads,local_files_only=True,trust_remote_code=False)

    def detect(self, image_rgb, instruction, classifier=None):
        """Use independent category queries plus one basket/robot query on observed pixels."""
        start=time.perf_counter()
        image=np.asarray(image_rgb,np.uint8)[...,:3]
        picture=Image.fromarray(image)
        categories=instruction_categories(instruction)
        sources=list(categories)+['__negative__']
        prompts=[ALIASES.get(category,category.replace('_',' '))+' .' for category in categories]
        prompts.append('basket. robot arm. robot gripper.')
        self.torch.set_num_threads(self.threads)
        preprocess=time.perf_counter()
        inputs=self.processor(images=[picture]*len(prompts),text=prompts,padding=True,return_tensors='pt').to('cpu')
        preprocess_seconds=time.perf_counter()-preprocess
        inference=time.perf_counter()
        with self.torch.inference_mode():
            output=self.model(**inputs)
        inference_seconds=time.perf_counter()-inference
        results=self.processor.post_process_grounded_object_detection(
            output,inputs.input_ids,**{self.threshold_key:self.threshold,
            'text_threshold':self.text_threshold,'target_sizes':[image.shape[:2]]*len(prompts)})
        targets=[]
        negatives=[]
        for source,prompt,result in zip(sources,prompts,results):
            labels=result.get('text_labels',result.get('labels'))
            for box,score,label in zip(result['boxes'],result['scores'],labels):
                row=dict(category=source,prompt=prompt,dino_label=str(label),dino_score=float(score),
                         box_xyxy=box.detach().cpu().numpy().tolist())
                (negatives if source=='__negative__' else targets).append(row)
        raw_count=len(targets)
        targets=_merge_query_duplicates(targets)
        clip_seconds=0.
        if classifier is not None and targets:
            clip_start=time.perf_counter()
            crops=[]
            for target in targets:
                x0,y0,x1,y1=np.asarray(target['box_xyxy']).round().astype(int)
                crops.append(image[max(0,y0-3):min(image.shape[0],y1+3),
                                   max(0,x0-3):min(image.shape[1],x1+3)].copy())
            names=categories+NEGATIVES
            scores=np.asarray(classifier(crops,names))
            for row,values in zip(targets,scores):
                index=int(np.argmax(values[:len(categories)]))
                negative=int(np.argmax(values[len(categories):]))
                row.update(clip_label=categories[index],clip_scores=dict(zip(names,map(float,values))),
                    clip_negative_label=NEGATIVES[negative],
                    clip_negative_margin=float(values[index]-values[len(categories)+negative]))
            clip_seconds=time.perf_counter()-clip_start
        return dict(model_id=MODEL_ID,model_revision=MODEL_REVISION,model_sha256=MODEL_SHA256,
            categories=categories,prompts=prompts,proposals=targets,negative_proposals=negatives,
            raw_target_count=raw_count,mask_backend=None,
            timing=dict(preprocess_seconds=preprocess_seconds,inference_seconds=inference_seconds,
                        clip_seconds=clip_seconds,total_seconds=time.perf_counter()-start),
            evidence='normal_RGB_and_instruction_only')


@lru_cache(maxsize=1)
def proposer():
    return GroundingProposer()
