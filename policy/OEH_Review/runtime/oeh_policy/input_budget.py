"""Cross-process input reservation ledger, including unreported/failed requests."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import uuid


class InputBudgetExceeded(RuntimeError):pass


class InputBudget:
    def __init__(self,path,cfg):self.path,self.cfg=Path(path),cfg

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a+', encoding="utf-8") as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            raw=f.read();state=json.loads(raw) if raw else {'schema':'oeh.input-budget.v1','requests':{},'blocked':False}
            try:yield state
            finally:
                f.seek(0);f.truncate();json.dump(state,f);f.flush();fcntl.flock(f,fcntl.LOCK_UN)

    def reserve(self,channel,estimate):
        cap=self.cfg[channel+'_call_input_limit'];call_limit=self.cfg[channel+'_call_limit']
        with self.transaction() as state:
            rows=list(state['requests'].values())
            if state['blocked'] or estimate>cap or sum(x['channel']==channel for x in rows)>=call_limit or sum(x['charged'] for x in rows)+cap>self.cfg['episode_input_limit']:
                raise InputBudgetExceeded('input budget exhausted or per-call estimate exceeds cap')
            rid=uuid.uuid4().hex
            state['requests'][rid]={'channel':channel,'reserved':cap,'charged':cap,'estimate':estimate,'status':'reserved_or_usage_unreported'}
        return rid

    def settle(self,rid,usage):
        reported=usage.get('input_tokens') if isinstance(usage,dict) else None
        if type(reported) is not int or reported<0:return
        with self.transaction() as state:
            row=state['requests'][rid];row.update(charged=max(reported,row['reserved']),reported_input=reported,reported_output=usage.get('output_tokens') or 0,status='reported')
            # Keep the whole reservation; never spend expected savings on extra calls.
            if (reported>row['reserved'] or sum(r['charged'] for r in state['requests'].values())>self.cfg['episode_input_limit']
                    or sum(r['charged']+r.get('reported_output',0) for r in state['requests'].values())>self.cfg['episode_input_limit']+self.cfg.get('output_allowance',0)):state['blocked']=True
