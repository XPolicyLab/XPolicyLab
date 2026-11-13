from collections import OrderedDict
from fractions import Fraction
import av,numpy as np

class _Array:
    def __init__(self,value):self.value=value
    def asnumpy(self):return self.value

class IndexedVideoReader:
    def __init__(self,path,num_threads=1,**kwargs):
        self.path=str(path);self.num_threads=1;self._cache=OrderedDict()
        with av.open(self.path,mode='r') as container:
            stream=container.streams.video[0]
            self.frames=int(stream.frames or 0);self.rate=stream.average_rate
            if self.frames<=0 or self.rate is None:raise ValueError('An admitted video must contain exact frame count and rate')
            self.width,self.height=stream.width,stream.height
    def __len__(self):return self.frames
    def get_avg_fps(self):return float(self.rate)
    def get_batch(self,indices):return _Array(self.get(indices))
    def get(self,indices):
        requested=[int(i) for i in indices]
        if not requested:return np.empty((0,self.height,self.width,3),np.uint8)
        if min(requested)<0 or max(requested)>=self.frames:raise IndexError('Video ordinal outside admitted media')
        wanted=set(requested);found={i:self._cache[i] for i in wanted if i in self._cache}
        missing=wanted-set(found)
        if missing:
            # Ordinals, not approximated timestamps, define RGB/depth alignment.
            with av.open(self.path,mode='r') as container:
                stream=container.streams.video[0];stream.thread_count=1
                for number,frame in enumerate(container.decode(stream)):
                    if number in missing:found[number]=frame.to_ndarray(format='rgb24')
                    if number>=max(missing):break
            if wanted-set(found):raise ValueError('Video ends before its declared frame count')
        for i in requested:
            self._cache.pop(i,None);self._cache[i]=found[i]
        while len(self._cache)>16:self._cache.popitem(last=False)
        return np.stack([found[i] for i in requested])

def decode_depth_frames(video_path,timestamps,tolerance_s,vcodec,log_loaded_timestamps=False):
    import torch
    if vcodec!='libx264rgb':raise ValueError('AS2 admits original lossless RGB-packed depth only')
    reader=IndexedVideoReader(video_path)
    indices=[round(float(t)*reader.get_avg_fps()) for t in timestamps]
    if timestamps and max(abs(i/reader.get_avg_fps()-float(t)) for i,t in zip(indices,timestamps))>=tolerance_s:raise ValueError('Depth timestamp alignment differs')
    rgb=reader.get(indices)
    value=(rgb[...,0].astype(np.uint16)<<8)|rgb[...,1].astype(np.uint16)
    return torch.from_numpy(value)
