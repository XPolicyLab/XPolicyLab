"""Propose normal-taxonomy entities with grounded RGB and retained public prototypes."""
from copy import deepcopy
import math
import time

import cv2
import numpy as np
from PIL import Image

from coap.tasks import classify_r02b2_perception as base
from coap.tasks import classify_r05_grounded_perception as grounded
from coap.tasks.classify_grounding_dino import proposer, _merge_query_duplicates
from coap.tasks.classify_r05_grounded_clip import encoder
from coap.tasks.classify_r06_robot_projection import robot_mask
from coap.tasks.sorting_paired_car_semantics import PROTOTYPES, PROTOTYPE_LABELS
from coap.tasks.sorting_r03_dino import _encode


CATEGORIES=('toy_car','action_camera','pen','watch','garage')
NEGATIVES=('basket','robot_arm','robot_gripper','table','table_line')
CLIP_NAMES=CATEGORIES+('phone',)+NEGATIVES
PROMPTS=('toy car. wooden toy car. toy tank. toy taxi.',
         'action camera. GoPro camera.', 'pen. mechanical pencil.',
         'wristwatch. wrist watch.',
         'plush toy animal. stuffed rabbit toy. cute animal figurine.',
         'basket. robot arm. robot gripper.')


def _detect(image):
    """Query the five fixed public families independently, with robot and basket negatives."""
    model=proposer()
    model.torch.set_num_threads(model.threads)
    picture=Image.fromarray(image)
    started=time.perf_counter()
    inputs=model.processor(images=[picture]*len(PROMPTS),text=list(PROMPTS),padding=True,
                           return_tensors='pt').to('cpu')
    with model.torch.inference_mode():
        output=model.model(**inputs)
    results=model.processor.post_process_grounded_object_detection(output,inputs.input_ids,
        **{model.threshold_key:model.threshold,'text_threshold':model.text_threshold,
           'target_sizes':[image.shape[:2]]*len(PROMPTS)})
    targets=[];negatives=[]
    for source,prompt,result in zip(CATEGORIES+('__negative__',),PROMPTS,results):
        labels=result.get('text_labels',result.get('labels'))
        for box,score,label in zip(result['boxes'],result['scores'],labels):
            item=dict(category=source,prompt=prompt,dino_label=str(label),dino_score=float(score),
                      box_xyxy=box.detach().cpu().numpy().tolist())
            (negatives if source=='__negative__' else targets).append(item)
    return dict(proposals=_merge_query_duplicates(targets),negative_proposals=negatives,
                timing=dict(total_seconds=time.perf_counter()-started),prompts=PROMPTS)


def _region(image,box,workspace,complete=False):
    """Keep a complete pen's visible components while leaving other entities separate."""
    cv2.setRNGSeed(0)
    if not complete:
        return grounded._grabcut_region(image,box,workspace)
    x0,y0,x1,y1=np.asarray(box,float).round().astype(int)
    x0,y0=max(0,x0),max(0,y0)
    x1,y1=min(image.shape[1],x1),min(image.shape[0],y1)
    if x1-x0<3 or y1-y0<3:
        return None
    left,top=max(0,x0-6),max(0,y0-6)
    right,bottom=min(image.shape[1],x1+6),min(image.shape[0],y1+6)
    crop=image[top:bottom,left:right]
    labels=np.zeros(crop.shape[:2],np.uint8)
    labels[y0-top:y1-top,x0-left:x1-left]=cv2.GC_PR_FGD
    labels[workspace[top:bottom,left:right]==0]=cv2.GC_BGD
    labels[[0,-1],:]=cv2.GC_BGD;labels[:,[0,-1]]=cv2.GC_BGD
    if not np.any(labels==cv2.GC_PR_FGD):
        return None
    cv2.grabCut(crop,labels,None,np.zeros((1,65)),np.zeros((1,65)),3,cv2.GC_INIT_WITH_MASK)
    foreground=((labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)).astype(np.uint8)
    count,connected,statistics,_=cv2.connectedComponentsWithStats(foreground,8)
    if count<2:
        return None
    areas=statistics[1:,cv2.CC_STAT_AREA]
    chosen=np.flatnonzero(areas>=max(3.,.02*float(areas.max())))+1
    mask=np.isin(connected,chosen).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour=np.concatenate(contours)
    x,y,width,height=cv2.boundingRect(contour)
    return contour[:,0,:]+[left,top],mask[y:y+height,x:x+width].copy(),(x+left,y+top,width,height)


def _material_geometry(ctx,image,camera,contour,mask,bbox):
    """Project contour extent with an area centroid from actual binary material."""
    points=np.asarray(contour,np.float32).reshape(-1,2)
    yy,xx=np.nonzero(mask)
    pixel=np.array([float(xx.mean())+bbox[0],float(yy.mean())+bbox[1]])
    world=base._project(ctx,points,ctx.table_z+.030,camera,image.shape)
    rectangle=cv2.minAreaRect(world[:,:2].astype(np.float32))
    corners=cv2.boxPoints(rectangle)
    edges=np.roll(corners,-1,axis=0)-corners
    axis=edges[np.argmax(np.linalg.norm(edges,axis=1))]
    axis/=max(np.linalg.norm(axis),1e-8)
    cross=np.array([-axis[1],axis[0]])
    length=np.ptp(np.einsum('ni,i->n',world[:,:2],axis))
    short=np.ptp(np.einsum('ni,i->n',world[:,:2],cross))
    position=base._project(ctx,[pixel],ctx.table_z+.030,camera,image.shape)[0]
    return pixel,position,float(math.atan2(axis[1],axis[0])%math.pi),(float(length),float(short)),world


def _item(ctx,image,camera,region,proposal):
    contour,mask,bbox=region
    pixel,position,yaw,size,world=_material_geometry(ctx,image,camera,contour,mask,bbox)
    if not (.025<size[0]<.23 and .003<size[1]<.15):
        return None
    if abs(position[0])>.56 or not -.285<position[1]<.40:
        return None
    x,y,width,height=map(int,bbox)
    crop=image[y:y+height,x:x+width].copy()
    selected=mask>0
    if int(selected.sum())<12:
        return None
    item=dict(name='object',kind='object',position=position,yaw=yaw,size=(*size,.030),
        pixel=tuple(pixel),bbox=tuple(bbox),head_bbox=tuple(bbox),camera=camera,contour=contour,
        crop=crop,entity_mask=mask,area=float(selected.sum()),mean_rgb=crop[selected].mean(axis=0),
        confidence=float(proposal.get('dino_score',.65)),surface_z=ctx.table_z+.030,
        dino_proposal=proposal,endpoints=np.array([
            world[np.argmin(world[:,:2]@[np.cos(yaw),np.sin(yaw)])],
            world[np.argmax(world[:,:2]@[np.cos(yaw),np.sin(yaw)])]]))
    item['views']={camera:dict(bbox=item['bbox'],pixel=item['pixel'],crop=crop,confidence=item['confidence'])}
    item['rgb_features']=base._features(image,contour,item)
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    yy,xx=np.nonzero(mask)
    centroid=np.array([xx.mean(),yy.mean()])
    rows,columns=np.where(distance>=.95*float(distance.max()))
    choices=np.column_stack((columns,rows))
    local=choices[np.argmin(np.linalg.norm(choices-centroid,axis=1))]
    solid=local+np.array([x,y])
    item['pixel']=tuple(solid)
    item['rgb_features']['solid_pixel']=solid
    item['solid_grasp_evidence']=dict(method='grounded_entity_distance_95pct_nearest_centroid',
        mask_pixels=int(mask.sum()),thickness_pixels=float(distance.max()),pixel=solid.tolist())
    return item


def _semantics(ctx,image,camera,items):
    """Use retained instance prototypes for families and open-set CLIP for pens and negatives."""
    if not items:
        return []
    crops=[]
    for item in items:
        x,y,width,height=item['bbox']
        margin=max(6,min(14,int(round(.20*max(width,height)))))
        context=image[max(0,y-margin):min(image.shape[0],y+height+margin),
                      max(0,x-margin):min(image.shape[1],x+width+margin)]
        crops.extend((item['crop'],context))
    values=np.asarray(encoder()(crops,CLIP_NAMES)).reshape(len(items),2,len(CLIP_NAMES))
    scores=.65*values[:,0]+.35*values[:,1]
    prototype_crops=[]
    for item in items:
        box=np.asarray(item['dino_proposal']['box_xyxy'],float)
        x0,y0,x1,y1=box.round().astype(int)
        pixels=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)
        _,_,_,size,_=base._geometry(ctx,pixels,ctx.table_z+.030,camera,image)
        crop=item['crop']
        if .025<size[0]<.23 and .003<size[1]<.15:
            x0,y0=max(0,x0),max(0,y0)
            x1,y1=min(image.shape[1],x1),min(image.shape[0],y1)
            if x1>x0 and y1>y0:
                crop=image[y0:y1,x0:x1]
                item['semantic_proposal_bbox']=(int(x0),int(y0),int(x1-x0),int(y1-y0))
        prototype_crops.extend((item['crop'],crop))
    vectors=[]
    for begin in range(0,len(prototype_crops),8):
        vectors.extend(_encode(prototype_crops[begin:begin+8]))
    similarities=np.einsum('nj,ij->ni',np.asarray(vectors),PROTOTYPES)
    similarities=similarities.reshape(len(items),2,-1).max(axis=1)
    accepted=[]
    for item,similar,values in zip(items,similarities,scores):
        clip=dict(zip(CLIP_NAMES,map(float,values)))
        prototypes={key:float(max(value for value,label in zip(similar,PROTOTYPE_LABELS) if label==key))
                    for key in ('car','camera','watch','toy','phone')}
        canonical=dict(toy_car=max(prototypes['car'],prototypes['phone']),
            action_camera=prototypes['camera'],watch=prototypes['watch'],garage=prototypes['toy'])
        best=max(canonical,key=canonical.get)
        cosine=canonical[best]
        positive=max(clip['toy_car'],clip['phone']) if best=='toy_car' else clip[best]
        negative=max(NEGATIVES,key=lambda key:clip[key])
        clip_labels={key:clip[key] for key in CATEGORIES}
        clip_labels['toy_car']=max(clip_labels['toy_car'],clip['phone'])
        clip_label=max(clip_labels,key=clip_labels.get)
        pen=item['rgb_features']['aspect']>3. and clip_label=='pen' and clip['pen']>=positive-.015
        family=(cosine>.55 and positive>=clip[negative]-.025) or (cosine>.70 and positive>=clip[negative]-.10)
        if not (pen and clip['pen']>=clip[negative]-.015) and not family:
            continue
        category='pen' if pen else best
        if item.get('pen_region_method') and category!='pen':
            continue
        item.update(category=category,category_scores=canonical,dino_class_cosines=canonical,
            normal_prototype_scores=prototypes,normal_prototype_confidence=cosine,
            clip_semantics=dict(category=category,positive_score=clip[category],
                negative_score=clip[negative],negative_category=negative,
                negative_margin=clip[category]-clip[negative],scores=clip,
                method='public_dinov2_family_prototypes_and_open_set_clip'))
        appearance=category
        if category=='toy_car':
            hsv=cv2.cvtColor(item['crop'],cv2.COLOR_RGB2HSV)
            selected=item['entity_mask']>0
            cyan=(hsv[...,0]>75)&(hsv[...,0]<110)&(hsv[...,1]>55)&(hsv[...,2]>65)
            pale=(hsv[...,1]<65)&(hsv[...,2]>110)
            red=((hsv[...,0]<9)|(hsv[...,0]>170))&(hsv[...,1]>90)
            if float(cyan[selected].mean())>.12 and float(pale[selected].mean())>.15 and float(red[selected].mean())<.10:
                appearance='phone'
        item['appearance_family']=appearance
        item['category_scores']['pen']=float(clip['pen'])
        accepted.append(item)
    return accepted


def _overlap(first,second):
    a,b,w,h=map(float,first);x,y,width,height=map(float,second)
    shared=max(0.,min(a+w,x+width)-max(a,x))*max(0.,min(b+h,y+height)-max(b,y))
    return shared/max(1.,min(w*h,width*height))


def _merge(items):
    groups=[]
    for item in sorted(items,key=lambda value:-value['area']):
        group=next((group for group in groups if any(_overlap(item['bbox'],member['bbox'])>.55 for member in group)),None)
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    objects=[]
    for group in groups:
        item=max(group,key=lambda value:value['area'])
        semantic=max(group,key=lambda value:value['normal_prototype_confidence']+.10*(value['category']=='pen'))
        for key in ('category','appearance_family','category_scores','clip_semantics',
                    'normal_prototype_scores','normal_prototype_confidence'):
            item[key]=semantic[key]
        item['normal_proposal_count']=len(group)
        objects.append(item)
    return objects


def _mapping(ctx,objects,baskets):
    """Keep observed category identities in their first assigned baskets across refreshes."""
    mapping=getattr(ctx,'_classify_free_mapping',None)
    if mapping is None:
        mapping=base._free_mapping(objects,baskets)
    else:
        mapping=dict(mapping)
        available=[index for index in range(len(baskets)) if index not in mapping.values()]
        for category in sorted({item['category'] for item in objects}-mapping.keys()):
            if not available:
                break
            average=float(np.mean([item['position'][0] for item in objects if item['category']==category]))
            index=min(available,key=lambda slot:abs(float(baskets[slot]['position'][0])-average))
            mapping[category]=index;available.remove(index)
    ctx._classify_free_mapping=dict(mapping)
    return mapping


def observe(ctx):
    """Measure normal sources from public RGB, calibration and proprioception alone."""
    if ctx.task_name!='classify_objects':
        return grounded.observe(ctx)
    started=time.perf_counter()
    image,camera=base._head(ctx)
    cached=getattr(ctx,'_classify_basket_cache',None)
    baskets=deepcopy(cached) if cached is not None else base.measure_baskets(ctx,image,camera)
    if len(baskets)==3:
        ctx._classify_basket_cache=deepcopy(baskets)
    workspace=base._object_mask(ctx,image,baskets)
    workspace[robot_mask(ctx,image,camera)>0]=0
    inference=_detect(image)
    proposals=list(inference['proposals'])
    for contour in base._contours(ctx,image,base._object_mask(ctx,image,baskets),False):
        x,y,width,height=cv2.boundingRect(contour)
        proposals.append(dict(category='rgb_region',dino_score=.65,
            box_xyxy=[x-2,y-2,x+width+2,y+height+2],source='rgb_material_region'))
    items=[]
    for proposal in proposals:
        region=_region(image,proposal['box_xyxy'],workspace)
        original=None
        if region is not None:
            item=_item(ctx,image,camera,region,proposal)
            if item is not None:
                items.append(item)
                original=item
        region=_region(image,proposal['box_xyxy'],workspace,complete=True)
        if region is None:
            continue
        complete=_item(ctx,image,camera,region,proposal)
        if (complete is not None and complete['rgb_features']['aspect']>3.
                and (original is None or complete['area']>original['area']+2.)):
            complete['pen_region_method']='all_visible_collinear_material_in_semantic_proposal'
            items.append(complete)
    items=_semantics(ctx,image,camera,items)
    materials=[item for item in items if item['category']!='pen'
               and item['normal_prototype_confidence']>.60]
    items=[item for item in items if not item.get('pen_region_method') or not any(
        _overlap(item['bbox'],other['bbox'])>.20 for other in materials)]
    objects=_merge(items)
    source_y=min((basket['position'][1] for basket in baskets),default=.445)-.045
    objects=[item for item in objects if item['position'][1]<source_y]
    for item in objects:
        category=item['category'];appearance=item['appearance_family']
        if category=='watch':
            grounded._watch_solid_case(item)
        if appearance=='phone':
            item['category']='phone'
        base._decorate_grasp(ctx,image,camera,item)
        item['category']=category
        item['appearance_family']=appearance
        if appearance=='phone':
            item['appearance_height']=.032
            item['appearance_height_evidence']='public_toy_car_00004_static_AABB'
    objects.sort(key=lambda item:(item['category'],item['position'][0],item['position'][1]))
    mapping=_mapping(ctx,objects,baskets)
    ctx.classify_grounded_normal_proposals=inference
    ctx._classify_skip_count_prior=True;ctx._classify_observed_initial=True
    return dict(objects=objects,baskets=baskets,instruction_mapping=mapping,mapping=mapping,
        evidence=dict(method='normal_grounding_dino_rgb_regions_public_prototypes_and_projected_robot',
            language=False,camera=camera,basket_count=len(baskets),object_count=len(objects),
            dino_proposals=len(inference['proposals']),rgb_proposals=len(proposals)-len(inference['proposals']),
            timing=dict(dino=inference['timing'],total_seconds=time.perf_counter()-started)))


classify_scene=observe
observe_scene=observe
