"""Keep identity crops aligned with their owning sorting geometry camera."""
import types

import numpy as np

from coap.tasks import sorting_r01_d as parent_D


def align_primary_views(objects, return_review=False):
    """Copy primary view metadata from the owning geometry without dropping aliases."""
    aligned = []
    changes = []
    for index, original in enumerate(objects):
        item = dict(original)
        views = {camera:dict(view) for camera,view in original.get('views',{}).items()}
        camera = original.get('camera','cam_head')
        previous = views.get(camera,{})
        different = tuple(previous.get('bbox',()))!=tuple(original.get('bbox',()))
        if not different and 'crop' in previous and 'crop' in original:
            different = not np.array_equal(previous['crop'],original['crop'])
        view = dict(previous)
        for key in ('bbox','pixel','crop','confidence'):
            if key in original:
                view[key] = original[key]
        views[camera] = view
        item['views'] = views
        aligned.append(item)
        if different:
            changes.append(dict(index=index,camera=camera,
                geometry_bbox=tuple(original.get('bbox',())),
                previous_view_bbox=tuple(previous.get('bbox',()))))
    review = dict(object_count=len(objects),changed_items=changes,
                  source='geometry_owned_camera_view_metadata',execution_exit_added=False)
    return (aligned,review) if return_review else aligned


def run_with_core(ctx, core):
    """Align local observation dictionaries after all parent alias recovery is done."""
    replay = core.__globals__['memory'].run
    replay_namespace = dict(replay.__globals__)
    original_pair = replay_namespace['pair_objects']
    def pair(images, targets, demonstrations, *args, **kwargs):
        reviews = []
        for role,objects in (('sources',targets),('demonstrations',demonstrations)):
            aligned,review = align_primary_views(objects,return_review=True)
            # These are task-local observations, also retained by departure tracking.
            for original,normalized in zip(objects,aligned):
                original['views'] = normalized['views']
            review.update(step=int(ctx.steps),role=role)
            reviews.append(review)
        ctx.sorting_primary_view_alignment = reviews
        return original_pair(images,targets,demonstrations,*args,**kwargs)
    replay_namespace['pair_objects'] = pair
    private_replay = types.FunctionType(replay.__code__,replay_namespace,replay.__name__,
                                       replay.__defaults__,replay.__closure__)
    private_replay.__kwdefaults__ = replay.__kwdefaults__
    namespace = dict(core.__globals__)
    namespace['memory'] = types.SimpleNamespace(run=private_replay)
    private_core = types.FunctionType(core.__code__,namespace,core.__name__,
                                     core.__defaults__,core.__closure__)
    private_core.__kwdefaults__ = core.__kwdefaults__
    execution = private_core(ctx)
    try:
        for action in execution:
            reviews = list(getattr(ctx,'sorting_primary_view_alignment',[]))
            if isinstance(ctx.memory,dict):
                ctx.memory['sorting_primary_view_alignment'] = reviews
            elif isinstance(getattr(ctx.memory,'initial_partition',None),dict):
                ctx.memory.initial_partition['primary_view_alignment'] = reviews
            yield action
    finally:
        execution.close()


def run(ctx):
    """Retain all parent D initialization and replay behavior around aligned views."""
    yield from run_with_core(ctx,parent_D.run)
