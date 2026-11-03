"""Persistent offline Python process; fail closed if OS isolation is unavailable."""
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import time
import uuid
from ..tools.docker_control import remove_container, startup_limited

MAX_VIEW_IMAGE_EDGE_PX = 1000

KERNEL = '''import contextlib, ctypes, io, json, os, sys, traceback, resource, importlib.metadata
memory_limit = int(os.environ.get("Q0_PREVIEW_PYTHON_MEMORY_MIB", "16384"))*1024**2
resource.setrlimit(resource.RLIMIT_AS, (memory_limit,memory_limit))
resource.setrlimit(resource.RLIMIT_FSIZE, (64*1024**2,64*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (256,256))
# OpenCV otherwise expands to the host CPU count (many threads on a large host).
import cv2, numpy, scipy
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
cv2.setNumThreads(1)
runtime = {"opencv_threads": cv2.getNumThreads(), "python": sys.version.split()[0], "memory_limit_mib": memory_limit//1024**2}
for package in ("numpy", "scipy", "Pillow", "matplotlib", "opencv-python"):
    try: runtime[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: runtime[package] = None
def memory_status():
    try:
        with open("/proc/self/status") as f:
            return {k: int(v.split()[0]) for k,v in (line.split(":",1) for line in f)
                    if k in ("VmSize", "VmRSS", "VmPeak")}
    except (OSError, ValueError): return {}
# Leave room to report an allocation failure without losing the control reply.
emergency_reserve = bytearray(256*1024)
ns = {"__name__": "__main__"}
# Private, non-inheritable descriptors carry control traffic. FD 0/1/2 belong
# exclusively to executed code, including native libraries and subprocesses.
control_in = os.fdopen(os.dup(0), "r")
control_out = os.fdopen(os.dup(1), "w", buffering=1)
diagnostic_fd = os.dup(2)
null = open(os.devnull, "r+")
os.dup2(null.fileno(), 0)
# Preserve out-of-band/late output in _kernel.stderr, never in the control pipe.
os.dup2(diagnostic_fd, 1)
flush_native = ctypes.CDLL(None).fflush
flush_native.argtypes = [ctypes.c_void_p]
flush_native.restype = ctypes.c_int
control_out.write(json.dumps({"kernel_ready": True, "runtime": runtime})+"\\n")
control_out.flush()
for line in control_in:
    request = json.loads(line)
    rc = 0
    error = {}
    before = memory_status()
    with open(request["output"], "w", buffering=1) as out:
        os.dup2(out.fileno(), 1)
        os.dup2(out.fileno(), 2)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            sys.stdin = io.StringIO("")
            try:
                exec(compile(open(request["script"]).read(), request["script"], "exec"), ns)
            except BaseException as exc:
                if isinstance(exc, MemoryError): emergency_reserve = None
                error = {"error_type": type(exc).__name__, "error": str(exc)[:500],
                         "memory_error": isinstance(exc, MemoryError)}
                traceback.print_exc(); rc = 1
            finally:
                # Flush C/Fortran stdio while FD 1/2 still target this request's log.
                flush_native(None)
                out.flush()
                sys.stdin = io.StringIO("")
                os.dup2(diagnostic_fd, 1)
                os.dup2(diagnostic_fd, 2)
    control_out.write(json.dumps({"request_id":request["request_id"],"returncode":rc,
        "memory_before_kib":before,"memory_after_kib":memory_status(),"runtime":runtime,**error})+"\\n")
    control_out.flush()
'''


class Sandbox:
    def __init__(self, work, python_prefix, mode='bwrap', image='ubuntu:24.04', bwrap_executable='bwrap',
                 memory_limit_mib=16384):
        self.work = Path(work).resolve(); self.work.mkdir(parents=True, exist_ok=True)
        self.observations = self.work/'observations'; self.observations.mkdir(exist_ok=True)
        self.prefix = Path(python_prefix).resolve()
        self.mode, self.image = mode, image
        self.bwrap_executable = bwrap_executable
        self.child = None; self.index = 0
        self._control_buffer = b''
        if type(memory_limit_mib) is not int or not 512 <= memory_limit_mib <= 65536:
            raise ValueError('Python memory_limit_mib must be an integer between 512 and 65536')
        self.memory_limit_mib = memory_limit_mib
        self.generation = 0
        self.bootstrap_code = None
        self.observation_id = None
        self.workspace_ready = False
        self.last_result = {}
        self.runtime = {}

    @startup_limited
    def start(self, deadline=None):
        (self.work/'_kernel.py').write_text(KERNEL)
        clean = {'PYTHONPATH': '/work', 'PATH': '/opt/python/bin:/usr/bin:/bin', 'HOME': '/tmp', 'OPENBLAS_NUM_THREADS': '1',
                 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                 'OPENCV_FOR_THREADS_NUM': '1', 'NVIDIA_VISIBLE_DEVICES': 'void', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': '/tmp/matplotlib',
                  'LD_LIBRARY_PATH': '/opt/python/lib', 'MPLBACKEND': 'Agg',
                  'Q0_PREVIEW_PYTHON_MEMORY_MIB': str(self.memory_limit_mib)}
        if self.mode in ('bwrap', 'bwrap-sudo'):
            if not shutil.which(self.bwrap_executable):
                raise RuntimeError('bubblewrap is unavailable; configure Docker sandbox (never host execution)')
            prefix = ['sudo', '-n'] if self.mode == 'bwrap-sudo' else []
            # Some shared filesystems cannot resolve owner-only source paths through a root-only UID map.
            # Privileged setup uses host UID mapping, then setpriv irrevocably drops privilege
            # BEFORE Python starts. No host runtime code or credentials are mounted.
            namespaces = (['--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net',
                '--cap-drop', 'ALL', '--cap-add', 'CAP_SETUID', '--cap-add', 'CAP_SETGID', '--cap-add', 'CAP_SETPCAP']
                if self.mode == 'bwrap-sudo' else ['--unshare-all', '--cap-drop', 'ALL'])
            command = prefix + [self.bwrap_executable] + namespaces + ['--die-with-parent', '--new-session', '--clearenv',
                       '--ro-bind', '/usr', '/usr', '--ro-bind-try', '/lib', '/lib',
                       '--ro-bind-try', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
                       '--tmpfs', '/tmp', '--perms', '0755', '--dir', '/opt', '--ro-bind', str(self.prefix), '/opt/python',
                       '--bind', str(self.work), '/work',
                       '--ro-bind', str(self.observations), '/work/observations',
                       '--chdir', '/work']
            for k, v in clean.items(): command += ['--setenv', k, v]
            if self.mode == 'bwrap-sudo':
                command += ['/usr/bin/setpriv', f'--reuid={os.getuid()}', f'--regid={os.getgid()}',
                    '--clear-groups', '--inh-caps=-all', '--ambient-caps=-all', '--bounding-set=-all', '--no-new-privs']
            command += ['/opt/python/bin/python', '-u', '/work/_kernel.py']
        elif self.mode == 'docker':
            self.name = 'q0-preview-sandbox-'+uuid.uuid4().hex[:12]
            command = ['docker', 'run', '--pull=never', '-i', '--rm', '--name', self.name,
                       '--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                       '--pids-limit=64', '--memory='+str(self.memory_limit_mib)+'m',
                       '--memory-swap='+str(self.memory_limit_mib)+'m',
                       '--entrypoint=/opt/python/bin/python', '--cpus=2', '--user', f'{os.getuid()}:{os.getgid()}',
                       '--tmpfs', '/tmp:rw,nosuid,size=256m',
                       # Hide source/assets and caches baked into a rich runtime image.
                       '--tmpfs', '/workspace:ro,nosuid,noexec,size=1m',
                       '--tmpfs', '/root:ro,nosuid,noexec,size=1m',
                       '--tmpfs', '/mnt:ro,nosuid,noexec,size=1m',
                       '--mount', f'type=bind,src={self.prefix},dst=/opt/python,readonly',
                       '--mount', f'type=bind,src={self.work},dst=/work',
                       '--mount', f'type=bind,src={self.observations},dst=/work/observations,readonly',
                       '--workdir', '/work']
            for k, v in clean.items(): command += ['-e', f'{k}={v}']
            command += [self.image, '-u', '/work/_kernel.py']
        else:
            raise ValueError('sandbox must be bwrap, bwrap-sudo or docker')
        if deadline is not None and time.time()>=deadline:
            raise TimeoutError('Observation deadline before Python container startup')
        self.stderr = (self.work/'_kernel.stderr').open('a')
        self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, bufsize=0, env={'PATH': os.environ.get('PATH', '/usr/bin:/bin')})
        self._control_buffer = b''
        self.generation += 1
        self.workspace_ready = False
        started = time.monotonic()
        timeout = min(300., max(0., deadline-time.time())) if deadline is not None else 300.
        try:
            ready = self._receive(timeout)
            if not isinstance(ready, dict) or ready.get('kernel_ready') is not True:
                raise ValueError('Invalid Python kernel startup handshake')
            self.runtime = ready['runtime']
            (self.work/'startup.json').write_text(json.dumps({'generation':self.generation,
                'seconds':time.monotonic()-started, 'startup_limit_seconds':300.,
                'execution_limit_seconds':45., 'runtime':self.runtime},indent=2))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.child:
            if self.mode == 'docker':
                remove_container(self.name)
            self.child.kill(); self.child.wait(timeout=10)
            self.child.stdin.close(); self.child.stdout.close()
            self.stderr.close(); self.child = None
        self._control_buffer = b''
        self.workspace_ready = False

    def _exchange(self, request, timeout):
        """Read a complete, attributed reply without blocking on a partial line."""
        self.child.stdin.write((json.dumps(request)+'\n').encode())
        self.child.stdin.flush()
        result = self._receive(timeout)
        if (not isinstance(result, dict) or result.get('request_id') != request['request_id']
                or type(result.get('returncode')) is not int or result['returncode'] not in (0, 1)):
            raise ValueError('invalid or mismatched Python control response')
        return result

    def _receive(self, timeout):
        deadline = time.monotonic()+timeout
        while b'\n' not in self._control_buffer:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.child.stdout], [], [], remaining)[0]:
                raise TimeoutError('Python calculation exceeded its time limit')
            chunk = os.read(self.child.stdout.fileno(), 4096)
            if not chunk:
                raise RuntimeError('Python kernel exited before returning a result')
            self._control_buffer += chunk
            if len(self._control_buffer) > 4096:
                raise ValueError('oversized Python control response')
        line, self._control_buffer = self._control_buffer.split(b'\n', 1)
        return json.loads(line)

    def _execute_once(self, code, log, deadline):
        if not isinstance(code, str) or len(code) > 100000:
            raise ValueError('Python code too long')
        if time.time() >= deadline:
            raise TimeoutError('observation deadline')
        if self.child is None or self.child.poll() is not None:
            self.close(); self.start(deadline)
        self.index += 1
        name = f'_python_{self.index:05d}'
        script, output = self.work/(name+'.py'), self.work/(name+'.out')
        script.write_text(code)
        timeout = min(45, max(0, deadline-time.time()))
        request = {'request_id': uuid.uuid4().hex, 'script': '/work/'+script.name,
                   'output': '/work/'+output.name}
        try:
            result = self._exchange(request, timeout)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            self.close()
            result = {'returncode': 124 if isinstance(exc, TimeoutError) else 1,
                       'kernel_reset': True, 'error': str(exc), 'request_id': request['request_id']}
        if result.get('memory_error'):
            self.close()
            result['kernel_reset'] = True
        text = output.read_text(errors='replace') if output.exists() else ''
        log = Path(log); log.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, log/'code.py')
        (log/'output.txt').write_text(text)
        self.runtime = result.pop('runtime', None) or self.runtime
        result.update(kernel_generation=self.generation, observation_id=self.observation_id,
                      output_path='/work/'+output.name, code_executed=True)
        self.last_result = result
        (log/'meta.json').write_text(json.dumps(result))
        if self.runtime:
            (self.work/'python_environment.json').write_text(json.dumps(self.runtime, indent=2))
        return self._render(result, text, code, output.name)

    def _render(self, result, text, code, output_name):
        # Keep the entire output on disk, but avoid replaying huge arrays/listings.
        limit = 6000
        excerpt = text if len(text) <= limit else text[:1500]+'\n[output truncated; full text: /work/'+output_name+']\n'+text[-4500:]
        if 'FileNotFoundError' in text and ('cam_head.png' in text or 'cam_left_wrist.png' in text or 'cam_right_wrist.png' in text):
            excerpt += '\nCamera inputs are JPEG, not PNG. Use observation_io.image("cam_head") or observation_io.image_path(camera, observation_id); do not guess filenames.'
        if 'SyntaxError' in text and '\\n' in code:
            excerpt += '\nUse actual newlines in the JSON code string. Literal backslash-n outside a quoted Python string is invalid; resend corrected Python, not escaped source.'
        if result.get('kernel_reset') or result.get('variables_lost'):
            excerpt += '\nPython kernel reset. Previous calculated variables were lost; saved /work files remain.'
            if result.get('observation_restored'):
                excerpt += ' observation_io and vlm_geom are already configured for the current observation. Recompute any missing variables; the failed code was not rerun.'
            elif self.bootstrap_code:
                excerpt += ' Workspace recovery did not complete; inspect the reported recovery error before using observations.'
            else:
                excerpt += ' Reload observation_io when an observation is available.'
        visible = {k: v for k, v in result.items() if k not in ('memory_before_kib', 'memory_after_kib')}
        return json.dumps(visible)+'\n'+excerpt

    def configure(self, code, log, deadline, observation_id):
        """Restore trusted bookkeeping once after a failed kernel; never replay user code."""
        self.bootstrap_code = code
        self.observation_id = observation_id
        before = self.generation
        result = self._execute_once(code, log, deadline)
        failed = dict(self.last_result)
        if failed.get('returncode') != 0 and (failed.get('kernel_reset') or failed.get('memory_error')):
            self.close()
            result = self._execute_once(code, Path(log)/'recovery', deadline)
        self.workspace_ready = self.last_result.get('returncode') == 0
        self.last_result['variables_lost'] = before > 0 and self.generation != before
        self.last_result['observation_restored'] = self.workspace_ready
        (Path(log)/'configuration_summary.json').write_text(json.dumps({
            'first_attempt': failed, 'final_result': self.last_result,
            'observation_id': observation_id}, indent=2))
        return result

    def _restore_workspace(self, log, deadline):
        result = self._execute_once(self.bootstrap_code, log, deadline)
        self.workspace_ready = self.last_result.get('returncode') == 0
        return self.workspace_ready, result

    def execute(self, code, log, deadline):
        if not isinstance(code, str) or len(code) > 100000:
            raise ValueError('Python code too long')
        before = self.generation
        if self.bootstrap_code and (self.child is None or self.child.poll() is not None or not self.workspace_ready):
            ok, restored = self._restore_workspace(Path(log)/'restore_before', deadline)
            if not ok:
                self.last_result.update(code_executed=False, observation_restored=False)
                Path(log).mkdir(parents=True, exist_ok=True)
                (Path(log)/'code.py').write_text(code)
                (Path(log)/'output.txt').write_text('Requested code was not executed: observation restoration failed.\n'+restored)
                (Path(log)/'meta.json').write_text(json.dumps(self.last_result))
                return json.dumps(self.last_result)+'\nRequested code was not executed: observation restoration failed.\n'+restored
        reply = self._execute_once(code, log, deadline)
        original = dict(self.last_result)
        if original.get('kernel_reset') and self.bootstrap_code and time.time() < deadline:
            try:
                ok, _ = self._restore_workspace(Path(log)/'restore_after', deadline)
                original['observation_restored'] = ok
                if not ok:
                    original['recovery_error'] = self.last_result.get('error', 'observation restoration failed')
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                original.update(observation_restored=False, recovery_error=str(exc))
        original.update(kernel_generation=self.generation,
                        variables_lost=before > 0 and self.generation != before,
                        namespace_preserved=not original.get('kernel_reset') and self.generation == before)
        if self.workspace_ready:
            original['observation_restored'] = True
        self.last_result = original
        (Path(log)/'meta.json').write_text(json.dumps(original))
        text = (Path(log)/'output.txt').read_text()
        return self._render(original, text, code, Path(original['output_path']).name)

    def image_path(self, value):
        if not isinstance(value, str) or not value.startswith('/work/'):
            raise ValueError('image path must start with /work/')
        p = (self.work/value.removeprefix('/work/')).resolve()
        if not p.is_relative_to(self.work) or not p.is_file() or p.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            raise ValueError('image outside sandbox or unsupported')
        if p.stat().st_size > 8_000_000:
            raise ValueError('image exceeds 8 MB')
        # Read dimensions only; preserve original bytes and pixel coordinates.
        from PIL import Image
        with Image.open(p) as image:
            width, height = image.size
        if max(width, height) > MAX_VIEW_IMAGE_EDGE_PX:
            raise ValueError(
                f'image is {width}x{height} px; tool images allow a longest edge of '
                f'{MAX_VIEW_IMAGE_EDGE_PX} px. Regenerate a smaller image with run_python, '
                'then return it through return_images or view_image.')
        return p
