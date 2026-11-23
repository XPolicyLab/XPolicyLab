# Copyright © 2026 Martingale. All Rights Reserved.
import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import numpy as np
from PIL import Image
from . import recipe_handlers
from .rlm_context import RLMContext
from .episode_protocol import python_tool
from .cell_worker import pack
from .isolation import CellWorker
from .motion_preview import preview, preview_candidates
from .dual_arm import move_both
from .paired_gripper import set_grippers
from .observation_views import observation_bundle
from .temporal_geometry_bridge import GeometryHistory
from .._backend import EpisodeFinished

PUBLIC = frozenset(('task','state','image','show','segment','back_project','locate',
                    'camera_meta','move_to','approach_to','set_gripper','home','fit_tube_openings','check_tube_alignment',
                    'fit_blocks','fit_slots','fit_pan','fit_peg_landmarks','fit_peg','fit_mugs',
                    'measure_disks','measure_key','measure_wedge','fit_basket','track_basket'))
PUBLIC = PUBLIC | frozenset(recipe_handlers.PUBLIC)
CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')

EXTENSIONS={'preview_move':preview,'preview_candidates':preview_candidates,'move_both':move_both,'set_grippers':set_grippers,
            'observation_bundle':observation_bundle,'geometry_capture':GeometryHistory.capture,'geometry_measure':GeometryHistory.measure}
LOCAL={'action_count':'robo.action_count()\n    Number of actions executed. No actions.',
       'record_progress':'robo.record_progress(checklist: dict)\n    Keep a compact agent-authored checklist (under 8000 characters); echoed in the result of the cell that changes it. A claim, not a verdict.',
       'skill_catalog':'robo.skill_catalog(capability=None)\n    Module and function signatures of one library namespace, or of all.',
       'skill_doc':'robo.skill_doc("capability/module.py", "function")\n    Signature and complete docstring of one library function.',
       'skill_read':'robo.skill_read("relative/path")\n    Text or image bytes of one library file.',
       'tool_doc':'robo.tool_doc(*names)\n    Signature and complete contract of robo tools. No actions.'}

def _render_extension(name,function):
    import inspect,textwrap
    parameters=list(inspect.signature(function).parameters.values())
    while parameters and parameters[0].name in ('self','robo'):parameters.pop(0)
    head=f'robo.{name}{inspect.Signature(parameters)}'
    doc=inspect.getdoc(function) or ''
    return head+('\n'+textwrap.indent(doc,'    ') if doc else '')

def tool_doc(robot_cls,*names):
    """Signature and complete contract of robo tools: the robot's own methods (including those the always-shown
    reference only lists by name) and the shared extensions."""
    from ..prompt import _public_members,_render_callable,_render_property
    members=_public_members(robot_cls)
    if not names:return 'robo.tool_doc(name, ...) prints the full contract of: '+', '.join(sorted({*members,*EXTENSIONS,*LOCAL}))
    known=sorted({*members,*EXTENSIONS,*LOCAL})
    def render(n):
        if not isinstance(n,str) or n not in known:
            import difflib
            close=difflib.get_close_matches(str(n),known,n=3,cutoff=0.5)
            return f'robo.{n}: no such robo tool'+(' (did you mean '+', '.join(close)+'?)' if close else '')+'; robo.tool_doc() lists the names.'
        if n in EXTENSIONS:return _render_extension(n,EXTENSIONS[n])
        if n in LOCAL:return LOCAL[n]
        return _render_property('robo',n,members[n]) if isinstance(members[n],property) else _render_callable('robo',n,members[n])
    return '\n\n'.join(render(n) for n in names)

class Result:
    def __init__(self,result,images=()): self.result,self.images=result,list(images)
    @property
    def content_blocks(self):
        blocks=[{'type':'text','text':json.dumps(self.result,default=str)}]
        for raw in self.images:
            blocks.append({'type':'image','source':{'type':'base64','media_type':'image/png',
                          'data':base64.b64encode(raw).decode()}})
        return blocks

class Toolkit:
    def __init__(self,robo,workspace,*,worker=None,context=None):
        self.robo=robo;self.workspace=Path(workspace);self.workspace.mkdir(parents=True,exist_ok=True)
        self.worker=worker or CellWorker(self.workspace)
        self.context=context if context is not None else RLMContext()
        self.lock=threading.RLock();self.replies=self.context.repl_results;self.images=[];self.progress={}
        self.geometry=GeometryHistory(self.workspace/'geometry')
        self.suspended=False;self.progress_shown={}
    def get_tools_spec(self):
        return [python_tool()]
    def action_count(self):
        status=self.robo._backend._toolkit._status()
        return status.get('step')
    def dispatch(self,name,args,kwargs):
        if name=='display':
            value=args[0]
            if isinstance(value,bytes):raw=value
            else:
                im=value if isinstance(value,Image.Image) else Image.fromarray(np.asarray(value,dtype=np.uint8))
                stream=io.BytesIO();im.save(stream,format='PNG');raw=stream.getvalue()
            self.images.append(raw);return None
        if name=='show':
            camera=args[0] if args else kwargs.get('camera','cam_head')
            return self.dispatch('display',[self.robo.image(camera)],{})
        if name=='action_count':return self.action_count()
        if name=='task':return self.robo.task
        if name=='skill_catalog':return self.context.list_memory(*args,**kwargs)
        if name=='skill_doc':return self.context.inspect_memory(*args,**kwargs)
        if name=='skill_read':return self.context.read_memory(*args,**kwargs)
        if name=='record_progress':
            value=args[0] if args else kwargs
            if len(json.dumps(value))>8000:raise ValueError('Keep checklist under 8000 characters')
            self.progress=value;return {'source':'agent claim, not the task outcome','progress':value}
        if name=='tool_doc':return tool_doc(type(self.robo),*args,**kwargs)
        if name=='preview_move':return preview(self.robo,*args,**kwargs)
        if name=='preview_candidates':return preview_candidates(self.robo,*args,**kwargs)
        if name=='move_both':return move_both(self.robo,*args,**kwargs)
        if name=='set_grippers':return set_grippers(self.robo,*args,**kwargs)
        if name=='observation_bundle':return observation_bundle(self.robo,*args,**kwargs)
        if name=='geometry_capture':return self.geometry.capture(self.robo,*args,**kwargs)
        if name=='geometry_measure':return self.geometry.measure(*args,**kwargs)
        if name=='artifact':
            if not args or args[0] not in tuple(c+'.png' for c in CAMERAS):raise PermissionError('Only camera artifacts')
            return self.robo.artifact(*args,**kwargs)
        if name not in PUBLIC:raise PermissionError('No such public robot tool: '+name)
        return getattr(self.robo,name)(*args,**kwargs)
    def execute_tool(self,name,arguments):
        with self.lock:
            if name!='python':return Result({'error':'Only python cells are supported; the episode ends by itself.'})
            ident=arguments.get('request_id');code=arguments.get('code')
            if not isinstance(ident,str) or not ident or len(ident)>128 or not isinstance(code,str):return Result({'error':'request_id and code are required strings'})
            digest=hashlib.sha256(code.encode()).hexdigest()
            if ident in self.replies:
                old=self.replies[ident]
                if old['sha256']!=digest:return Result({'error':'request_id reused with different code'})
                return old.get('result') or Result({'error':'Cell outcome uncertain; preserve session, never replay it','interrupted':True})
            if self.suspended:return Result({'error':'Session suspended; operator resume required','interrupted':True})
            index=len(self.replies)+1;stem=self.workspace/f'cell_{index:04d}'
            try:
                stem.with_suffix('.py').write_text(code)
                stem.with_suffix('.json').write_text(json.dumps({'request_id':ident,'sha256':digest,'status':'executing'}))
            except OSError as error:
                self.suspended=True
                return Result({'interrupted':True,'error':'Cannot persist pre-execution reply: '+str(error),'actions_executed':False})
            reply=self.replies[ident]={'sha256':digest};self.images=[];started=time.monotonic()
            try: feedback=self.worker.execute(code,self.dispatch,filename=f'/workspace/{stem.name}.py')
            except EpisodeFinished as error:feedback={'episode_active':False,'reason':error.reason,**getattr(error,'cell_feedback',{})}
            except BaseException as error:
                self.suspended=True
                feedback={'interrupted':True,'error':f'{type(error).__name__}: {error}',
                          'continuation':'Live worker state retained; uncertain cell must not be replayed'}
            feedback.update(cell=index,actions_executed=self.action_count(),elapsed_s=time.monotonic()-started)
            if self.progress!=self.progress_shown:feedback['progress']=self.progress;self.progress_shown=self.progress
            result=Result(feedback,self.images)
            reply['result']=result
            try:
                image_names=[]
                for i,raw in enumerate(self.images):
                    p=self.workspace/f'{stem.name}_image_{i}.png';p.write_bytes(raw);image_names.append(p.name)
                stem.with_suffix('.json').write_text(json.dumps({'request_id':ident,'sha256':digest,'status':'returned','feedback':feedback,'images':image_names},default=str))
            except OSError as error:
                self.suspended=True
                feedback.update(interrupted=True,evidence_error=str(error),continuation='Completed cell retained in memory; never replay')
            return result
    def close(self):self.worker.close()
