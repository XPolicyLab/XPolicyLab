import ast
import base64
import contextlib
import io
import json
import math
import sys
import time
import traceback
from types import SimpleNamespace


class Record(SimpleNamespace):
    def arm(self,name):
        if name not in ('left','right'):raise ValueError('Unknown arm')
        return getattr(self,name)


def pack(value):
    import dataclasses
    if type(value).__module__.startswith('PIL.'):
        import numpy as np
        value=np.asarray(value)
    if type(value).__module__.startswith('numpy') and hasattr(value,'item') and not getattr(value,'shape',None):
        return pack(value.item())
    if dataclasses.is_dataclass(value):
        return {'$record':{f.name:pack(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value,bytes):return {'$bytes':base64.b64encode(value).decode()}
    if hasattr(value,'dtype') and hasattr(value,'shape'):
        return {'$array':value.tolist(),'dtype':str(value.dtype)}
    if isinstance(value,dict):return {k:pack(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [pack(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Unsupported RPC value: '+type(value).__name__)


def unpack(value):
    if isinstance(value,list):return [unpack(v) for v in value]
    if not isinstance(value,dict):return value
    if '$record' in value:return Record(**{k:unpack(v) for k,v in value['$record'].items()})
    if '$bytes' in value:return base64.b64decode(value['$bytes'],validate=True)
    if '$array' in value:
        import numpy as np
        dtype=np.dtype(value['dtype'])
        if dtype.hasobject:raise ValueError('Object arrays forbidden')
        return np.asarray(value['$array'],dtype=dtype)
    return {k:unpack(v) for k,v in value.items()}


def main():
    channel=sys.stdout
    class CellTerminal(BaseException):
        pass
    def send(value):channel.write(json.dumps(value)+'\n');channel.flush()
    trace_enabled=False
    def rpc(name,*args,**kwargs):
        frames=[]
        if trace_enabled:
            frame=sys._getframe(1)
            while frame:
                filename=frame.f_code.co_filename
                if filename.startswith('/workspace/'):
                    frames.append({'path':filename[len('/workspace/'):],'line':frame.f_lineno,'function':frame.f_code.co_name})
                frame=frame.f_back
        send({'rpc':name,'args':pack(args),'kwargs':pack(kwargs),'stack':list(reversed(frames))} if trace_enabled else {'rpc':name,'args':pack(args),'kwargs':pack(kwargs)})
        response=json.loads(sys.stdin.readline())
        if 'terminal' in response:raise CellTerminal(response['terminal'])
        if 'error' in response:raise RuntimeError(response['error'])
        return unpack(response['value'])
    class Robot:
        LEFT='left';RIGHT='right';OPEN=-1;CLOSE=1;KEEP=0
        CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
        @property
        def geometry(self):
            from geometry import Geometry
            return Geometry()
        @property
        def task(self):return rpc('task')
        def __getattr__(self,name):return lambda *a,**kw:rpc(name,*a,**kw)
    import numpy as np
    from efficiency_helpers import bounded_sequence
    namespace={'__name__':'__policy__','robo':Robot(),'math':math,'time':time,
               'bounded_sequence':bounded_sequence,'np':np,'numpy':np,'log':print,'display':lambda value:rpc('display',value)}
    send({'ready':True})
    for line in sys.stdin:
        request=json.loads(line);trace_enabled=bool(request.get('code_trace'));stdout=io.StringIO();stderr=io.StringIO();result={}
        try:
            filename=request.get('filename','<cell>')
            tree=ast.parse(request['code']);tail=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                exec(compile(tree,filename,'exec'),namespace)
                if tail:
                    value=eval(compile(ast.Expression(tail.value),filename,'eval'),namespace)
                    if value is not None:result['value']=repr(value)
        except BaseException:result['exception']=traceback.format_exc()
        result.update(stdout=stdout.getvalue(),stderr=stderr.getvalue())
        send({'result':result})


if __name__=='__main__':main()
