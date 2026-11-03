"""One episode's LLM input+output cutoff, scaled by its configured native-step allowance."""
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import threading
import uuid


class LLMTokenBudgetExceeded(Exception):
    pass


def validate(value):
    allowed={'limit','reference_steps','on_exceed','output_reserve','unknown_input_reserve','scaling'}
    if not isinstance(value,dict) or set(value)-allowed:
        raise ValueError('unknown review.llm_token_budget settings')
    result={'limit':7000000,'reference_steps':700,'on_exceed':'stop',
            'output_reserve':8192,'unknown_input_reserve':25000,**copy.deepcopy(value)}
    if result['on_exceed']!='stop':raise ValueError('LLM token cutoff currently supports explicit episode stop')
    for key in ('limit','reference_steps','output_reserve','unknown_input_reserve'):
        if type(result[key]) is not int or result[key]<1:
            raise ValueError('positive integer llm_token_budget.'+key+' required')
    scaling=result.get('scaling')
    if 'scaling' in result:
        if (not isinstance(scaling,dict) or set(scaling)!={'base_limit','reference_steps','native_step_limit'}
                or any(type(v) is not int or v<1 for v in scaling.values())
                or scaling['reference_steps']!=result['reference_steps']
                or result['limit']!=scaling['base_limit']*scaling['native_step_limit']//scaling['reference_steps']):
            raise ValueError('inconsistent LLM token budget scaling provenance')
    return result


def resolve(value,native_step_limit):
    """Freeze a total episode limit proportional to its configured native-step allowance.

    Revalidation/refreezing uses the original base, never scales the resolved limit twice.
    The allowance is fixed before the episode, not unlocked by executing more steps.
    """
    cfg=validate(value)
    if type(native_step_limit) is not int or native_step_limit<1:
        raise ValueError('positive integer native_step_limit required for LLM token scaling')
    base=cfg.get('scaling',{}).get('base_limit',cfg['limit'])
    cfg['limit']=base*native_step_limit//cfg['reference_steps']
    cfg['scaling']={'base_limit':base,'reference_steps':cfg['reference_steps'],'native_step_limit':native_step_limit}
    return validate(cfg)


class EpisodeLLMBudget:
    def __init__(self,path,cfg):self.path,self.cfg=Path(path),validate(cfg)

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a+', encoding="utf-8") as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            state=json.loads(f.read() or '{}') or {'schema':'oeh.llm-token-budget.v1','limit':self.cfg['limit'],'blocked':False,'requests':{}}
            if state['limit']!=self.cfg['limit']:raise ValueError('episode LLM token limit mismatch')
            basis=self.cfg.get('scaling')
            if state.get('scaling') is not None and state['scaling']!=basis:
                raise ValueError('episode LLM token scaling basis mismatch')
            if basis is not None:state['scaling']=copy.deepcopy(basis)
            try:yield state
            finally:
                f.seek(0);f.truncate();json.dump(state,f);f.flush();fcntl.flock(f,fcntl.LOCK_UN)

    @staticmethod
    def totals(state):
        rows=state['requests'].values()
        actual=sum(r.get('input_tokens',0)+r.get('output_tokens',0) for r in rows)
        charged=sum(r['charged'] for r in rows)
        return {'reported_input_output':actual,'charged_including_unreported_and_inflight':charged}

    def status(self):
        with self.transaction() as state:
            return {k:copy.deepcopy(v) for k,v in state.items() if k!='requests'} | self.totals(state)

    def check(self):
        state=self.status()
        if state['blocked']:raise LLMTokenBudgetExceeded('episode LLM token budget stopped: '+state.get('reason','limit'))

    def reserve(self,channel,estimated_input):
        reserve=max(int(estimated_input),self.cfg['unknown_input_reserve'])+self.cfg['output_reserve']
        with self.transaction() as state:
            if state['blocked']:raise LLMTokenBudgetExceeded('episode LLM token budget already stopped')
            if self.totals(state)['charged_including_unreported_and_inflight']+reserve>state['limit']:
                state.update(blocked=True,reason='insufficient_request_reserve')
                raise LLMTokenBudgetExceeded('episode LLM token budget lacks request reserve')
            rid=uuid.uuid4().hex
            state['requests'][rid]={'channel':channel,'reserved':reserve,'charged':reserve,'status':'inflight'}
            return rid

    def update(self,rid,usage,*,finished=False):
        with self.transaction() as state:
            row=state['requests'][rid]
            if isinstance(usage,dict):
                for key in ('input_tokens','output_tokens'):
                    if type(usage.get(key)) is int and usage[key]>=0:row[key]=max(row.get(key,0),usage[key])
                cached=(usage.get('input_tokens_details') or {}).get('cached_tokens')
                if type(cached) is int:row['cached_input_subset']=max(row.get('cached_input_subset',0),cached)
            actual=row.get('input_tokens',0)+row.get('output_tokens',0)
            known='input_tokens' in row and 'output_tokens' in row
            row['charged']=actual if finished and known else max(row['reserved'],actual)
            row['status']='reported' if finished and known else 'usage_unreported' if finished else 'inflight'
            if self.totals(state)['reported_input_output']>=state['limit']:
                state.update(blocked=True,reason='reported_input_output_limit')
            elif self.totals(state)['charged_including_unreported_and_inflight']>=state['limit']:
                state.update(blocked=True,reason='charged_input_output_limit')
            return state['blocked']


class BudgetedClient:
    """Wrap the authenticated transport; Codex additionally checks every retry/usage event.

    Failed operations without usage retain their reserve. A shared stop signals all
    in-flight clients and prohibits every new model request. No policy fallback runs.
    """
    def __init__(self,client,budget,channel,prompt,tool=None):
        self.client,self.budget,self.channel=client,budget,channel
        self.prompt,self.tool=prompt,tool
        self.history_proxy=0

    def __getattr__(self,key):return getattr(self.client,key)

    def reset_context(self):
        self.client.reset_context();self.history_proxy=0

    def create(self,body):
        from .input_context import input_estimate,validate as input_cfg,text_tokens
        cfg=input_cfg({});schema=self.tool or (body.get('tools') or [{'parameters':body.get('output_schema',{})}])[0]
        estimate=input_estimate(body,self.prompt,schema,cfg)
        if body.get('transport')=='codex_app_server':estimate['estimated_input']+=int(self.history_proxy*cfg['tokenizer_multiplier'])
        current=[None];last_usage=[None];attempts=[0];native_turns=[0]
        stop=threading.Event()
        def begin():
            if current[0] is not None:self.budget.update(current[0],last_usage[0],finished=True)
            current[0]=self.budget.reserve(self.channel,estimate['estimated_input']);last_usage[0]=None;attempts[0]+=1
        def progress(usage):
            last_usage[0]=usage
            if self.budget.update(current[0],usage):raise LLMTokenBudgetExceeded('episode LLM input+output cutoff reached')
        def monitor():
            while not stop.wait(.1):
                if self.budget.status()['blocked']:
                    closed=getattr(self.client,'closed',None)
                    if closed is not None:closed.set()
                    return
        def before_turn():
            if native_turns[0]:begin()
            else:self.budget.check()
            native_turns[0]+=1
        codex=body.get('transport')=='codex_app_server'
        self.budget.check();begin()
        if codex:
            self.client.before_model_turn=before_turn
            self.client.model_usage_observer=progress
        watcher=threading.Thread(target=monitor,name='oeh-llm-token-cutoff',daemon=True);watcher.start()
        try:
            data,errors=self.client.create(body)
            # Stand-in clients can implement Codex's body without native hooks.
            if current[0] is None:begin()
            usage=data.get('usage') or {};last_usage[0]=usage
            blocked=self.budget.update(current[0],usage,finished=True);current[0]=None
            output=data.get('output',[])
            raw='\n'.join(i.get('arguments','') for i in output if i.get('type')=='function_call') if codex else output
            self.history_proxy+=estimate['parts']['current_input']+estimate['parts']['images_estimate']+text_tokens(raw)
            if blocked:raise LLMTokenBudgetExceeded('episode LLM input+output cutoff reached')
            return data,errors
        except BaseException:
            if current[0] is not None:self.budget.update(current[0],last_usage[0],finished=True)
            raise
        finally:
            stop.set();watcher.join(.2)
            if codex:
                self.client.before_model_turn=None;self.client.model_usage_observer=None
