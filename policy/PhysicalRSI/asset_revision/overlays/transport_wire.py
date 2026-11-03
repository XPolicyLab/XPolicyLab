"""Adapt server batching to the XPolicyLab action-chunk contract."""
from numbers import Integral


class Batch10Wire:
    def __init__(self, backend):
        self.backend=backend
        self.ids=None

    def __getattr__(self,name):
        return getattr(self.backend,name)

    def reset(self):
        self.ids=None
        return self.backend.reset()

    def update_obs_batch(self, observations):
        if not isinstance(observations,(list,tuple)) or not 1<=len(observations)<=20:
            raise ValueError('Expected one to twenty active observations')
        ids=[o.get('env_idx') if isinstance(o,dict) else None for o in observations]
        if any(isinstance(i,bool) or not isinstance(i,Integral) or i<0 for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('Unique explicit environment IDs required')
        self.ids=None
        public=[]
        for obs in observations:
            clean={key:obs[key] for key in ('env_idx','state','instruction') if key in obs}
            if 'vision' in obs:
                clean['vision']={name:{key:camera[key] for key in ('color','intrinsic_matrix','extrinsic_matrix') if key in camera} for name,camera in obs['vision'].items()}
            public.append(clean)
        set_active_indices=getattr(self.backend,'set_active_indices',None)
        if callable(set_active_indices):
            set_active_indices(list(ids))
        result=self.backend.update_obs_batch(public)
        self.ids=list(ids)
        return result

    def get_action_batch(self, indices):
        if (self.ids is None or not indices or len(set(indices))!=len(indices)
                or any(isinstance(i,bool) or not isinstance(i,Integral) for i in indices)
                or not set(indices)<=set(self.ids)):
            raise ValueError('Action IDs must be distinct members of the latest observations')
        result=self.backend.get_action_batch(indices)
        if not isinstance(result,(list,tuple)) or len(result)!=len(indices):
            raise ValueError('Batch action count differs from active environment count')
        chunks=[]
        for item in result:
            chunk=[item] if isinstance(item,dict) else item
            if not isinstance(chunk,(list,tuple)) or not chunk or not all(isinstance(a,dict) for a in chunk):
                raise ValueError('Expected action dictionaries or nonempty action chunks')
            chunks.append(list(chunk))
        if len({len(c) for c in chunks})!=1:
            raise ValueError('Unequal action horizons cannot be consumed by the unchanged client')
        # Preserve all actions, including multi-action chunks. Never truncate,
        # duplicate, pad, or alter controller action values.
        return chunks


def adapt(backend,cfg):
    return Batch10Wire(backend) if cfg.get('batch_wire',False) else backend
