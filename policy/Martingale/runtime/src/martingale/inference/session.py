# Copyright © 2026 Martingale. All Rights Reserved.
"""One persistent robot session and one fixed Codex model for every instruction."""
from dataclasses import asdict
import json
import hashlib
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from . import task_text
from .rlm_context import RLMContext
from .accounts import AccountRuntime, select_account
from .control import (PROVIDER_INTERRUPTS, atomic_json, classify_provider_failure,
                      process_identity)
from .config import InferenceConfig
from .codex_cli import DISABLED_FEATURES, exec_prefix
from .toolkit import Toolkit
from .._backend import Ledger
from ..hosts.robodojo_native import NativeToolkit, NativeBackend
from ..hosts.xpolicylab_bridge import BridgeEnv
from ..robodojo_xpl_rgb import RobodojoOfficialColourRobot
from ..mcp_server import ArmMcpServer
from ..prompt import api_reference, knowledge_text

RETRY_INITIAL_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
AUTO_RETRY_KINDS = frozenset({"provider_network", "provider_error", "account_busy"})

CONTINUE = '''The episode remains active. Continue from the current state and existing Python variables. Do not replay any acknowledged or partially executed cell. Read the last reply and reobserve if uncertain. Use a NEW request_id for a new cell. Recover and pursue the instruction until episode termination; no voluntary early finish.'''

INDEXED_TOOLS=('fit_key','measure_keyhole','calibrate_keyhole','key_alignment_view_pose','key_insertion_alignment',
               'begin_key_insertion','key_alignment_views','measure_visual_key','visual_key_step','key_alignment_overlay',
               'key_alignment_suggestion')
UNLISTED_TOOLS=frozenset({'approach_to',*INDEXED_TOOLS})

def reference_robot():
    """The robot class as shown in the tool reference: the same for every instruction.

    Tools in UNLISTED_TOOLS stay callable in every episode; their contract lives with the
    procedure that needs them, so agents that do not need them are not steered toward them.
    """
    return type(RobodojoOfficialColourRobot.__name__,(RobodojoOfficialColourRobot,),
                {'ABSENT_MEMBERS':RobodojoOfficialColourRobot.ABSENT_MEMBERS|UNLISTED_TOOLS,
                 '__doc__':RobodojoOfficialColourRobot.__doc__,'__module__':RobodojoOfficialColourRobot.__module__})

def tool_index():
    """The same short index for every instruction: name and first docstring line of each indexed tool."""
    import inspect
    rows=['    robo.'+name+': '+(inspect.getdoc(getattr(RobodojoOfficialColourRobot,name)) or '').split('\n')[0] for name in INDEXED_TOOLS]
    return ('More `robo` tools (callable in every episode; their full contracts are not reproduced above). Before the first\n'
            'use of one, print its signature and complete contract in a cell: print(robo.tool_doc("name", "other_name")).\n'
            +'\n'.join(rows))

class Session:
    def __init__(self,bridge,out,config,*,gpu=None,run_geometry_fitting=False,geometry_task=None,self_play=False):
        if self_play and run_geometry_fitting:
            raise ValueError('Self-play subcalls and geometry fitting are separate modes')
        self.bridge=bridge;self.out=Path(out).resolve();self.config=config;self.gpu=gpu
        self.self_play=bool(self_play);self._provider_home=None
        from .self_play_directives import load_directives
        directive_path=os.environ.get('MARTINGALE_SELF_PLAY_DIRECTIVES')
        if directive_path and not self.self_play:
            raise ValueError('Self-play directives require --self-play')
        self.self_play_directives=load_directives(directive_path)
        self.run_geometry_fitting=bool(run_geometry_fitting);self.geometry_task=geometry_task or 'unknown'
        self.geometry_ready=False
        self.out.mkdir(mode=0o700,parents=True,exist_ok=False)
        self.resume_event=threading.Event();self.status='initializing';self.toolkit=None
        self.thread_id=None;self.error=None;self.attempt=0;self.server=None
        self._status_lock=threading.RLock()
        self._suspend_lock=threading.Lock()
        self.identity=process_identity()
        self.provider_active=False;self.interruption_kind=None;self.pause_id=None
        self.last_provider_stderr=''
        self.retry_streak=0;self.retry_progress=None;self.retry_at=None
        self.automatic_retries=0
        self.last_resume_request=None;self.account_history=[];self.account_runtime=None
        self.account_registry=os.environ.get('MARTINGALE_ACCOUNT_REGISTRY')
        self._forbidden_roots=(self.out.parent,Path(__file__).resolve().parents[3])
        self.account=None;self._account_error=None
        self.remote_episode = None
        if 'MARTINGALE_CODEX_TOKEN' in os.environ and not (self_play or run_geometry_fitting):
            from .episode_client import RemoteEpisode
            token = os.environ['MARTINGALE_CODEX_TOKEN'].strip()
            if not token:
                raise ValueError('MARTINGALE_CODEX_TOKEN is empty')
            self.remote_episode = RemoteEpisode(token, config.model, config.effort)
        else:
            try:self.account=select_account(self.account_registry,forbidden_roots=self._forbidden_roots)
            except (ValueError,RuntimeError,OSError) as exc:self._account_error=type(exc).__name__
        self.context=RLMContext()
        self.library_hash=self.context.memory_fingerprint()
        digest=hashlib.sha256()
        for file in sorted(Path(__file__).parents[1].rglob('*.py')):
            digest.update(file.relative_to(Path(__file__).parents[1]).as_posix().encode()+b'\0'+file.read_bytes())
        self.runtime_hash=digest.hexdigest()
        self.write_status()
    def write_status(self):
        with self._status_lock:
            value={'status':self.status,'model':asdict(self.config),'system_sha256':self.config.fingerprint(),
                   'library_sha256':self.library_hash,'runtime_sha256':self.runtime_hash,'thread_id':self.thread_id,'provider_attempts':self.attempt,
                   'run_geometry_fitting':self.run_geometry_fitting,
                   'self_play':self.self_play,
                   'pid':os.getpid(),'process_identity':self.identity,'error':self.error,
                   'resume':'martingale resume EPISODE',
                   'automatic_retries':self.automatic_retries,'retry_at':self.retry_at,
                   'automatic_retry_eligible':self._can_auto_retry(),
                   'node_loss_restore_supported':False,'provider_active':self.provider_active,
                   'interruption_kind':self.interruption_kind,'pause_id':self.pause_id,
                   'resume_allowed':self._can_resume(),
                   'account_handoff_allowed':self._can_handoff(),
                   'account':self.account.alias if self.account else None,
                   'account_registry':self.account_registry,'account_history':self.account_history,
                   'credential_recovery_file':str(self.account_runtime.pending_refresh)
                       if self.account_runtime and self.account_runtime.pending_refresh else None,
                   'last_resume_request':self.last_resume_request,
                   'task_text_source':getattr(self,'task_text_source',None),
                   'updated_at':time.time()}
            if self.self_play:
                value['library_updates']=getattr(self,'library_updates',None)
                value['self_play_directives']=getattr(self,'self_play_directives',None)
            atomic_json(self.out/'status.json',value)
    def _uncertain_cell(self):
        if self.toolkit is None:return False
        return bool(getattr(self.toolkit,'suspended',False) or any(
            'result' not in reply for reply in getattr(self.toolkit,'replies',{}).copy().values()))
    def _can_resume(self):
        if self.bridge.over is not None:return False
        if self.interruption_kind=='provider_protocol':return False
        if self.provider_active and self.interruption_kind!='worker':return False
        if self.interruption_kind=='worker':
            return not getattr(self.toolkit,'suspended',False)
        return not self._uncertain_cell()
    def _can_handoff(self):
        return False
    def _can_auto_retry(self):
        return bool(self.status=='interrupted'
                    and (self.thread_id or (self.toolkit is not None and not self.toolkit.replies))
                    and self.interruption_kind in AUTO_RETRY_KINDS
                    and not self.provider_active and self._can_resume()
                    and not (self.account_runtime and self.account_runtime.pending_refresh))
    def _retry_delay(self):
        progress=sum('result' in r for r in getattr(self.toolkit,'replies',{}).copy().values())
        if progress!=self.retry_progress:
            self.retry_streak=0;self.retry_progress=progress
        delay=min(RETRY_MAX_SECONDS,RETRY_INITIAL_SECONDS*2**min(self.retry_streak,10))
        self.retry_streak+=1
        return delay
    def _record_recovery(self,source):
        with (self.out/'recovery.jsonl').open('a') as stream:
            stream.write(json.dumps({'source':source,'pause_id':self.pause_id,
                'kind':self.interruption_kind,'thread_id':self.thread_id,
                'attempt':self.attempt,'at':time.time()})+'\n')
    def _consume_resume_request(self):
        request=self.out/'resume.request'
        if not request.exists():return False
        value={}
        try:
            if request.is_symlink() or request.stat().st_size>16384:
                raise ValueError('Invalid operator request file')
            value=json.loads(request.read_text())
            if not isinstance(value,dict):raise ValueError('Invalid operator request')
            if (value.get('version')!=1 or value.get('pause_id')!=self.pause_id
                    or value.get('thread_id')!=self.thread_id
                    or value.get('process_identity')!=self.identity
                    or self.status!='interrupted'):
                raise ValueError('Stale request: pause, thread or process changed')
            if not self._can_resume():raise RuntimeError('Uncertain cell requires operator reconciliation')
            if value.get('account') is not None:
                raise RuntimeError('Mid-episode account handoff is disabled')
            if set(value)-{'version','request_id','pause_id','thread_id','process_identity','account','requested_at'}:
                raise ValueError('Resume accepts infrastructure fields only')
            self.last_resume_request={'request_id':value.get('request_id'),'status':'accepted'}
            return True
        except (OSError,ValueError,RuntimeError,TypeError) as exc:
            self.last_resume_request={'request_id':value.get('request_id') if isinstance(value,dict) else None,
                                      'status':'rejected','error':str(exc)}
            return False
        finally:
            request.unlink(missing_ok=True)
            self.write_status()
    def suspend(self,reason,kind='provider_error'):
        with self._suspend_lock:
            if self.bridge.over is None:self._suspend(reason,kind)
    def _suspend(self,reason,kind):
        if self.provider_active and kind=='provider_error':kind='worker'
        self.status='interrupted';self.error=reason;self.interruption_kind=kind
        self.pause_id=uuid.uuid4().hex
        deadline=None;self.retry_at=None
        readiness=(self._can_resume(),self._can_handoff())
        self.write_status()
        while self.bridge.over is None:
            eligible=self._can_auto_retry()
            if eligible and deadline is None:
                delay=self._retry_delay();deadline=time.monotonic()+delay
                self.retry_at=time.time()+delay;self.write_status()
            elif not eligible and deadline is not None:
                deadline=None;self.retry_at=None;self.write_status()
            requested=self.resume_event.wait(.2)
            self.resume_event.clear()
            current=(self._can_resume(),self._can_handoff())
            if current!=readiness:
                readiness=current;self.write_status()
            requested=self._consume_resume_request() or requested
            automatic=(deadline is not None and time.monotonic()>=deadline
                       and self._can_auto_retry())
            if requested or automatic:
                if not self._can_resume():
                    self.error='Uncertain cell execution requires operator diagnosis; do not replay';self.write_status();continue
                try:self._record_recovery('automatic' if automatic else 'operator-infrastructure-resume')
                except OSError:
                    self.error='Cannot persist recovery receipt; live state retained'
                    deadline=None;self.retry_at=None
                    continue
                if automatic:self.automatic_retries+=1
                self.retry_at=None
                self.status='running';self.error=None;self.pause_id=None;self.interruption_kind=None
                self.write_status();return
    def prompt(self,robo):
        stated,self.task_text_source=task_text.statement(self.bridge.get_obs(),robo.task)
        if self.run_geometry_fitting:
            return self.toolkit.prompt(stated)
        parts=[self.context.read_prompt_source('SYSTEM.md'),
            api_reference(reference_robot()),
            tool_index(),
            knowledge_text('robodojo_xpl_rgb'),
            self.context.read_prompt_source('LESSONS.md',library=True),
            self.context.read_prompt_source('CAPABILITIES.md',library=True),
            'Available capability namespaces (ALL are available in every episode):\n'+
            '\n'.join(self.context.memory_namespaces()),
            stated,
            'do not waste actions']
        if self.self_play:
            parts.append(self.context.read_prompt_source('SELF_PLAY.md'))
            if getattr(self,'self_play_directives',None) is not None:
                from .self_play_directives import prompt
                parts.append(prompt(self.self_play_directives))
        return self.context.build_root_prompt(parts)
    def argv(self,workspace):
        argv=exec_prefix(os.environ.get('MARTINGALE_CODEX_BIN','codex'))
        if self.thread_id:argv+=['resume']
        argv+=['--ignore-user-config','--ignore-rules','--skip-git-repo-check','--json',
               '-m',self.config.model,'-c','model_reasoning_effort='+json.dumps(self.config.effort),
               '-c','approval_policy="never"','-c','forced_login_method="chatgpt"',
               '-c','cli_auth_credentials_store="file"',
               '-c','project_doc_max_bytes=0','-c','web_search="disabled"',
               '--enable','skip_host_skill_discovery',
               '-c','mcp_servers.martingale.url='+json.dumps(self.server.url),
               '-c','mcp_servers.martingale.required=true',
               '-c','mcp_servers.martingale.tool_timeout_sec=3300',
               '-c','service_tier="default"','--enable','code_mode_host']
        for feature in DISABLED_FEATURES:
            argv+=['--disable',feature]
        if self.thread_id:argv+=[self.thread_id]
        else:argv+=['-C',str(workspace),'-s','read-only']
        return argv+['-']
    def run_agent(self,workspace,prompt):
        if self.remote_episode is not None:
            return self._run_remote_agent(prompt)
        if self.account is None:
            self.account=select_account(self.account_registry,forbidden_roots=self._forbidden_roots)
            self._account_error=None
        if self.account_runtime is None:
            self.account_runtime=AccountRuntime(self.account.resolved_runtime_root,
                hashlib.sha256(str(self.out).encode()).hexdigest(),forbidden_roots=self._forbidden_roots)
        self.attempt+=1;events=self.out/f'provider_{self.attempt:03d}.jsonl'
        self.last_provider_stderr=''
        keep=('PATH','HOME','USER','LANG','TERM','TMPDIR','SSL_CERT_FILE','CODEX_CA_CERTIFICATE',
              'HTTP_PROXY','HTTPS_PROXY','NO_PROXY')
        env={k:os.environ[k] for k in keep if k in os.environ};env['NO_PROXY']='127.0.0.1,localhost'
        faults=[];proc=None;wrong_thread=False
        def collect(stream,final=False):
            nonlocal wrong_thread
            while True:
                before=stream.tell();line=stream.readline()
                if not line:break
                if not line.endswith('\n') and not final:stream.seek(before);break
                try:event=json.loads(line)
                except ValueError:
                    faults.append({'type':'error','message':'Malformed provider JSON event'});continue
                if not isinstance(event,dict):
                    faults.append({'type':'error','message':'Malformed provider event object'});continue
                if event.get('type')=='thread.started':
                    ident=event.get('thread_id')
                    if not isinstance(ident,str) or not ident:
                        faults.append({'type':'error','message':'Missing provider thread identifier'})
                    elif self.thread_id and ident!=self.thread_id:
                        faults.append({'type':'error','message':'Provider resumed a different thread'})
                        wrong_thread=True
                    else:self.thread_id=ident;self.write_status()
                if event.get('type') in ('error','turn.failed'):faults.append(event)
        try:
            with self.account_runtime.lease(self.account) as home:
                self._provider_home=home
                if self.self_play:self.toolkit.subcalls.begin_turn()
                from .codex_isolation import runtime_command
                argv=runtime_command(self.argv(workspace),workspace,home,include_library=not self.run_geometry_fitting)
                with events.open('wb') as output,(self.out/f'provider_{self.attempt:03d}.stderr').open('wb') as errors:
                    with events.open('r') as reader:
                        self.provider_active=True;self.write_status()
                        try:
                            proc=subprocess.Popen(argv,cwd=workspace,env=env,stdin=subprocess.PIPE,stdout=output,stderr=errors)
                            try:proc.stdin.write(prompt.encode());proc.stdin.close()
                            except BrokenPipeError:pass
                            while proc.poll() is None:
                                collect(reader)
                                if (self.bridge.over is not None or wrong_thread
                                        or getattr(self.toolkit,'calibration_complete',False)):break
                                time.sleep(.2)
                        finally:
                            if self.self_play:self.toolkit.subcalls.end_turn()
                            if proc is not None and proc.poll() is None:
                                proc.terminate()
                                try:proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                            collect(reader,final=True)
                            self.provider_active=False
        finally:
            self._provider_home=None
            self.provider_active=False;self.write_status()
            stderr=self.out/f'provider_{self.attempt:03d}.stderr'
            if stderr.is_file():
                with stderr.open('rb') as stream:
                    stream.seek(max(0,stderr.stat().st_size-65536))
                    self.last_provider_stderr=stream.read().decode('utf-8',errors='replace')
        if wrong_thread:
            faults.append({'type':'error','code':'thread_identity_mismatch'})
        return proc.returncode,faults

    def _run_remote_agent(self, prompt):
        self.attempt += 1
        self.last_provider_stderr = ''
        events = self.out / f'provider_{self.attempt:03d}.jsonl'
        with events.open('w') as stream:
            def report(event):
                stream.write(json.dumps(event) + '\n')
                stream.flush()
                if event['type'] == 'thread.started':
                    self.thread_id = event['thread_id']
                    self.write_status()

            self.provider_active = True
            try:
                self.write_status()
                self.remote_episode.turn(prompt, self.toolkit, report, lambda: self.bridge.over is not None)
                return 0, []
            except Exception as exc:
                if self.bridge.over is not None:
                    return 0, []
                fault = {'type': 'error', 'message': str(exc)}
                report(fault)
                return 1, [fault]
            finally:
                self.provider_active = False
                self.write_status()

    def _self_play_query(self,prompt,call_dir):
        from .self_play import query_codex
        if not self.self_play or self._provider_home is None:
            raise RuntimeError('Self-play subcalls require an active provider lease')
        return query_codex(prompt,call_dir,self.config,self._provider_home,
                           cancelled=lambda:self.bridge.over is not None or self.toolkit.subcalls.cancelled)
    def initialize(self):
        from ..hosts import sam3_local
        self.bridge.first_obs(60)
        if self.config.sam3_sha256 is not None and not hasattr(self,'weights_verified'):
            checkpoint=Path(os.environ['SAM3_CHECKPOINT_PATH'])
            digest=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=self.config.sam3_sha256:raise ValueError('Segmentation checkpoint hash mismatch')
            self.weights_verified=True
        if not hasattr(self,'robot_kit'):
            self.robot_kit=NativeToolkit(BridgeEnv(self.bridge),sam3_local.shared(cuda_device=self.gpu),self.out,colour_only=True)
        if not hasattr(self,'robo'):
            self.robo=RobodojoOfficialColourRobot(NativeBackend(self.robot_kit),ledger=Ledger())
        if self.toolkit is None:
            if self.run_geometry_fitting:
                from .geometry_fitting_toolkit import FittingToolkit
                self.toolkit=FittingToolkit(self.robo,self.out/'workspace',task=self.geometry_task)
            else:
                if self.self_play:
                    from .self_play import SelfPlayToolkit, RLMSubcalls
                    subcalls=RLMSubcalls(self.out/'workspace'/'rlm',self._self_play_query)
                    self.toolkit=SelfPlayToolkit(self.robo,self.out/'workspace',subcalls=subcalls,context=self.context,
                                                directives=self.self_play_directives)
                else:
                    self.toolkit=Toolkit(self.robo,self.out/'workspace',context=self.context)
            self.toolkit.worker._hold_timeout=lambda reason:self.suspend(reason,kind='worker')
        if self.server is None and self.remote_episode is None:
            server=ArmMcpServer(self.toolkit,annotations={'readOnlyHint':False,'destructiveHint':False,'openWorldHint':False})
            try:server.start()
            except BaseException:
                server.stop();raise
            self.server=server
        if not hasattr(self,'initial_prompt'):
            self.initial_prompt=self.prompt(self.robo)
            (self.out/'prompt.txt').write_text(self.initial_prompt)
        self.status='running';self.error=None;self.write_status()

    def run(self):
        geometry_report_complete=False
        while self.bridge.over is None:
            try:
                self.initialize()
            except Exception as exc:
                self.suspend(f'Host initialization interrupted: {type(exc).__name__}: {exc}',kind='initialization')
                continue
            try:
                continuation=('Continue the blind RGB calibration using existing variables and saved evidence. '
                              'Call fit_progress() and keep investigating every remaining parameter and component. '
                              'Do not voluntarily stop until each has an RGB estimate with uncertainty or a documented '
                              'fit_impossible claim supported by actual attempts, evidence, and why further permitted '
                              'observations cannot identify it. Try alternate views and methods; unresolved notes, '
                              'failed fits, missing definitions, and interrupted sessions do not establish impossibility; do not waste actions. '
                              'Only call fit_finish() when the completion contract is satisfied. '
                              'Never invent a value. Never replay completed cells.'
                              if self.run_geometry_fitting else CONTINUE)
                rc,faults=self.run_agent(self.out/'workspace',self.initial_prompt if not self.thread_id else continuation)
            except Exception as exc:
                from .accounts import AccountBusy, CredentialConflict, AccountError
                kind=('account_busy' if isinstance(exc,AccountBusy) else
                      'credential_conflict' if isinstance(exc,CredentialConflict)
                          or (self.account_runtime and self.account_runtime.pending_refresh) else 'provider_auth' if isinstance(exc,AccountError) else
                          classify_provider_failure([str(exc)]))
                self.suspend(f'Provider invocation interrupted: {type(exc).__name__}: {exc}',kind=kind)
                continue
            if getattr(self.toolkit,'calibration_complete',False) and not faults:
                # Provider shutdown and the credential lease must finish before
                # the client can close the episode as a completed calibration.
                geometry_report_complete=True
                break
            if self.bridge.over is not None:break
            if rc or faults:
                kind=classify_provider_failure(faults,self.last_provider_stderr)
                self.suspend('Provider interrupted; retaining live state for recovery',kind=kind)
            elif not self.thread_id:
                self.suspend('Provider returned no resumable thread ID; preserve state for diagnosis',kind='provider_protocol')
            elif self.toolkit.suspended:
                self.suspend('Cell infrastructure interrupted; worker preserved for diagnosis',kind='worker')
            else:self.retry_streak=0
        verdict=self.bridge.over or {}
        self.pause_id=None;self.interruption_kind=None;self.retry_at=None
        if self.run_geometry_fitting and geometry_report_complete:
            self.status='geometry_finalizing';self.error=None
        elif self.run_geometry_fitting:
            self.status='geometry_incomplete';self.error='Calibration ended before the report was frozen'
        elif verdict.get('abandoned'):
            self.status='abandoned'
            self.error='Ended on a client-side fault: the evaluation client started a new episode without reporting this one\'s end'
        elif verdict.get('debug_completed') is True and 'success' not in verdict:
            self.status='debug_completed';self.error=None
        elif verdict.get('interrupted') or not isinstance(verdict.get('success'),bool):
            self.status='interrupted'
            self.error='Evaluation client ended without a success/failure verdict; simulator continuation unavailable'
        else:
            self.status='success' if verdict['success'] else 'failure'
            self.error=None
        self.write_status()
        if self.server:self.server.stop()
        if self.remote_episode is not None:
            try:
                self.remote_episode.release()
            except Exception as exc:
                # The gateway expires disconnected sessions. Preserve the simulator verdict.
                (self.out / 'provider_cleanup_error.txt').write_text(str(exc))
        if self.toolkit:self.toolkit.close()
        if self.self_play and self.toolkit:
            try:
                self.library_updates=self.toolkit.finish_updates(verdict)
            except Exception as exc:
                self.library_updates={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            self.write_status()
        if self.run_geometry_fitting and geometry_report_complete:
            self.status='geometry_fitted';self.write_status()
            self.geometry_ready=True
            self.bridge.finish()
