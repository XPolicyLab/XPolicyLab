"""Host-local shared DA3/SAM3 inference; no entity memory or credentials in this service."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import uuid
import numpy as np


def identity(cfg):
    return {k:cfg.get(k) for k in ('da3_checkpoint','sam3_checkpoint','device','max_edge','mask_threshold')}


class RemoteModels:
    def __init__(self, cfg):
        self.cfg=cfg
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(5.)
            client.connect(cfg['inference_socket'])
            client.sendall((json.dumps({'operation':'ping','identity':identity(cfg)})+'\n').encode())
            reply=json.loads(client.makefile('rb').readline())
            if reply!={'ready':True}:raise RuntimeError('shared inference is not ready: '+str(reply))
    def infer(self, rgb, K, labels):
        directory=Path(self.cfg['inference_socket']).parent
        source=directory/(uuid.uuid4().hex+'.npz')
        result=source.with_suffix('.result.npz')
        np.savez(source,rgb=rgb,K=K)
        source.chmod(0o600)
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
                client.settimeout(self.cfg['perception_seconds'])
                client.connect(self.cfg['inference_socket'])
                client.sendall((json.dumps({'input':str(source),'labels':labels,'identity':identity(self.cfg)})+'\n').encode())
                response=json.loads(client.makefile('rb').readline())
            if response.get('error'):raise RuntimeError(response['error'])
            with np.load(result,allow_pickle=False) as data:
                detections=[{'label':str(label),'confidence':float(score),'mask':mask}
                    for label,score,mask in zip(data['labels'],data['scores'],data['masks'])]
                return data['image'],data['depth'],data['K'],detections
        finally:
            source.unlink(missing_ok=True)
            result.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"));cfg=cfg.get('stmem',cfg)
    sock=Path(cfg['inference_socket'])
    sock.parent.mkdir(parents=True,exist_ok=True);sock.parent.chmod(0o700)
    if sock.exists():raise RuntimeError('refuse to replace an existing inference socket')
    from .stmem_perception import DA3SAM3
    models=DA3SAM3(cfg)
    models.infer(np.zeros((378,504,3),np.uint8),np.array([[300,0,252],[0,300,189],[0,0,1]]),['object'])
    lock=threading.Lock()
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request=json.loads(self.rfile.readline(1024*1024))
                if request.get('identity')!=identity(cfg):raise ValueError('model configuration differs from service')
                if request.get('operation')=='ping':
                    self.wfile.write(b'{"ready":true}\n');return
                source=Path(request['input']).resolve()
                if source.parent!=sock.parent.resolve() or source.suffix!='.npz':
                    raise ValueError('request must reference a private inference input')
                if request['identity']!=identity(cfg):raise ValueError('model configuration differs from service')
                with np.load(source,allow_pickle=False) as data:rgb,K=data['rgb'],data['K']
                with lock:image,depth,k,detections=models.infer(rgb,K,request['labels'])
                # A timed-out caller removes its input; skip unnecessary late output when possible.
                if source.exists():
                    np.savez(source.with_suffix('.result.npz'),image=image,depth=depth,K=k,
                        labels=np.array([d['label'] for d in detections],dtype=str),
                        scores=np.array([d['confidence'] for d in detections]),
                        masks=np.array([d['mask'] for d in detections],dtype=bool))
                response={'ok':True}
            except Exception as exc:response={'error':type(exc).__name__+': '+str(exc)[:300]}
            try:self.wfile.write((json.dumps(response)+'\n').encode())
            except BrokenPipeError:pass
    class Server(socketserver.ThreadingUnixStreamServer):daemon_threads=True
    with Server(str(sock),Handler) as server:
        sock.chmod(0o600)
        print(json.dumps({'ready':True,'identity':identity(cfg)}),flush=True)
        try:server.serve_forever()
        finally:sock.unlink(missing_ok=True)

if __name__=='__main__':main()
