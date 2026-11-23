# Copyright © 2026 Martingale. All Rights Reserved.
import json, os, selectors, subprocess, sys, time
from pathlib import Path
from .cell_worker import pack, unpack

def command(workspace, *, include_library=True):
    """No host root/home, host PID namespace, sockets or network in either worker."""
    executable=Path(sys.executable).resolve();prefix=executable.parent.parent
    argv=[os.environ.get('MARTINGALE_BWRAP','bwrap'),'--die-with-parent','--unshare-all','--new-session',
          '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
          '--proc','/proc','--dev','/dev','--tmpfs','/tmp']
    mounts=[(prefix,prefix)]
    for root in (Path(sys.base_prefix),Path(sys.prefix)):
        mounts.extend(((root.resolve(),root.resolve()),(root.resolve(),root.absolute())))
    for source,target in dict.fromkeys(mounts):
        if target!=Path('/usr') and not str(target).startswith('/usr/'):
            argv+=['--ro-bind',str(source),str(target)]
    worker=Path(__file__).with_name('cell_worker.py')
    helper=Path(__file__).with_name('efficiency_helpers.py')
    argv+=['--ro-bind',str(helper),'/efficiency_helpers.py']
    argv+=['--ro-bind',str(worker),'/cell_worker.py','--bind',str(Path(workspace).resolve()),'/workspace']
    geometry=Path(__file__).parents[1]/'geometry.py'
    argv+=['--ro-bind',str(geometry),'/geometry.py']
    library=Path(__file__).with_name('library')
    if include_library:
        dimensions = Path(__file__).parents[1] / 'robot_dimensions.py'
        argv+=['--ro-bind',str(dimensions),'/robot_dimensions.py',
               '--ro-bind',str(library),'/workspace/library']
    else:
        # A binary install may put the package inside a mounted Python prefix.
        # Mask it there too: fitting cells must not read the installed package.
        import sysconfig
        for directory in {sysconfig.get_path('purelib'), sysconfig.get_path('platlib')}:
            package=Path(directory)/'martingale'
            if package.is_dir():argv+=['--tmpfs',str(package)]
    argv+=['--chdir','/workspace','--setenv','HOME','/tmp','--setenv','OPENBLAS_NUM_THREADS','1',
           '--setenv','PYTHONPATH','/workspace',str(Path(sys.executable).absolute()),'-u','/cell_worker.py']
    return argv


class CellWorker:
    MAX_IPC_BYTES=64*1024*1024
    def __init__(self,workspace,*,timeout=600,include_library=True):
        self.timeout=timeout
        self._ipc_buffer=bytearray()
        self.argv=command(workspace,include_library=include_library)
        self.proc=subprocess.Popen(self.argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                   text=True,bufsize=1,env={'PATH':'/usr/bin:/bin'})
        try:
            if self.read().get('ready') is not True:raise RuntimeError('Cell worker failed to initialize')
        except BaseException as exc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            detail=self.proc.stderr.read(4000).strip()
            raise RuntimeError('Cell worker failed to initialize: '+(detail or str(exc))) from exc

    def read(self):
        deadline=time.monotonic()+self.timeout
        scan_from=0
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout,selectors.EVENT_READ)
            while True:
                end=self._ipc_buffer.find(b'\n',scan_from)
                if end>=0:
                    if end>self.MAX_IPC_BYTES:
                        raise RuntimeError('Isolated IPC payload exceeds limit; output incomplete')
                    line=bytes(self._ipc_buffer[:end]);del self._ipc_buffer[:end+1]
                    message=json.loads(line)
                    if not isinstance(message,dict):
                        raise RuntimeError('Isolated IPC message must be an object')
                    return message
                if len(self._ipc_buffer)>self.MAX_IPC_BYTES:
                    raise RuntimeError('Isolated IPC payload exceeds limit; output incomplete')
                remaining=deadline-time.monotonic()
                if remaining<=0 or not selector.select(remaining):
                    hold=getattr(self,'_hold_timeout',None)
                    if hold is not None:
                        hold('isolated_ipc_timeout')
                        deadline=time.monotonic()+self.timeout
                        continue
                    raise TimeoutError('Isolated cell timed out')
                scan_from=len(self._ipc_buffer)
                chunk=os.read(self.proc.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('Isolated worker exited before complete IPC result')
                self._ipc_buffer.extend(chunk)

    def send(self,value):
        self.proc.stdin.write(json.dumps(value)+'\n');self.proc.stdin.flush()

    def execute(self,code,dispatch,*,filename='<cell>',trace=None):
        self.send({'code':code,'filename':filename,'code_trace':trace is not None})
        deadline=time.monotonic()+self.timeout
        terminal=None
        trace_errors=[]
        def trace_call(phase,message,value):
            try:return trace(phase,message,value) if trace else None
            except Exception as error:
                trace_errors.append({'operation':'trace_'+phase,'error':f'{type(error).__name__}: {error}'})
                return None
        try:
            while True:
                if time.monotonic()>deadline:
                    hold=getattr(self,'_hold_timeout',None)
                    if hold is None:
                        raise TimeoutError('Cell time budget exceeded')
                    hold('cell_wall_timeout')
                    deadline=time.monotonic()+self.timeout
                message=self.read()
                if 'result' in message:
                    result=message['result']
                    if (set(message)!={'result'} or not isinstance(result,dict)
                            or not {'stdout','stderr'}<=set(result)
                            or set(result)-{'stdout','stderr','value','exception'}
                            or any(not isinstance(v,str) for v in result.values())):
                        raise RuntimeError('Invalid isolated result schema; output incomplete')
                    if trace_errors:
                        message['result']['host_infrastructure_error']=True
                        message['result'].setdefault('evidence_errors',[]).extend(trace_errors)
                    if terminal is not None:
                        terminal.cell_feedback=message['result']
                        raise terminal
                    return message['result']
                if terminal is not None or trace_errors:
                    self.send({'terminal':terminal.reason if terminal else 'host_trace_failure'})
                    continue
                marker=trace_call('start',message,None)
                if trace_errors:
                    self.send({'terminal':'host_trace_failure'})
                    continue
                try:
                    value=dispatch(message['rpc'],unpack(message['args']),unpack(message['kwargs']))
                    response={'value':pack(value)}
                except Exception as error:response={'error':f'{type(error).__name__}: {error}'}
                except BaseException as error:
                    from martingale._backend import EpisodeFinished
                    if not isinstance(error,EpisodeFinished):raise
                    terminal=error
                    response={'terminal':error.reason}
                trace_call('end',message,(marker,response.get('error') or response.get('terminal')))
                if trace_errors and terminal is None:response={'terminal':'host_trace_failure'}
                self.send(response)
        except BaseException as error:
            if terminal is None or error is terminal:
                raise
            feedback=dict(getattr(terminal,'cell_feedback',{}))
            feedback.update(host_infrastructure_error=True,output_complete=False,
                output_unavailable_reason='Worker failed before returning complete cell output')
            feedback.setdefault('evidence_errors',[]).extend(trace_errors)
            feedback['evidence_errors'].append({'operation':'terminal_worker_drain',
                'error':f'{type(error).__name__}: {error}'})
            try:self.close()
            except Exception as cleanup_error:
                feedback['evidence_errors'].append({'operation':'terminal_worker_reap',
                    'error':f'{type(cleanup_error).__name__}: {cleanup_error}'})
            terminal.cell_feedback=feedback
            raise terminal from error

    def close(self):
        if self.proc.poll() is None:self.proc.kill()
        self.proc.wait(timeout=10)
        for stream in (self.proc.stdin,self.proc.stdout,self.proc.stderr):stream.close()
