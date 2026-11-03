"""Initial visual evidence. Semantic orientation waits for an explicit global front reference."""
import copy



def make_anchor(observation,*,known_at,assets=None):
    direction={'status':'unknown','reason':'semantic_direction_deferred_until_global_front_is_defined',
        'coordinate_frame':'image_xy','direction_stem_to_crossbar':None,'is_3d_orientation':False}
    return {'observed_at':observation['time'],'known_at':known_at,'observation_id':observation.get('observation_id'),
        'scene_frame_id':observation.get('scene_frame_id'),'native_step':observation.get('native_step'),
        'camera':observation.get('camera'),'frame_id':observation['frame_id'],'image_size':copy.deepcopy(observation.get('image_size')),
        'bbox_2d':copy.deepcopy(observation.get('bbox_2d')),'box':copy.deepcopy(observation['box']),
        'geometry_source':copy.deepcopy(observation.get('geometry_provenance')),'semantic_direction_2d':direction,
        'front_reference':None,'assets':copy.deepcopy(assets or {}),'anchor_semantics':'initial_observed_pose_not_task_goal',
        'metric_accuracy':'unverified','revision':0}
