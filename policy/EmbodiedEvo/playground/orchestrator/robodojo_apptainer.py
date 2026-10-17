"""Run the pinned simulator filesystem as a shared, read-only Apptainer SIF."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from collections import deque

from orchestrator.config import REPO
from orchestrator.files import write_json
from orchestrator.robodojo_image import docker_command, option


def apptainer_command(argv, profile, bridge, sif):
    # Reuse the exact image environment, repository mounts and native invocation.
    docker = docker_command(argv, profile, bridge, 'unused')
    command = ['apptainer', 'exec', '--fakeroot', '--nvccli', '--writable-tmpfs',
               '--containall', '--cleanenv', '--home', f'{bridge / "home"}:/run/embodiedevo/home',
               '--no-mount', 'bind-paths,cwd',
               '--net', '--network', 'none', '--workdir', str(bridge / 'work')]
    index = 0
    while index < len(docker):
        arg = docker[index]
        if arg in ('-v', '-e', '-w'):
            if not (arg == '-e' and docker[index + 1].startswith('HOME=')):
                command += [{'-v': '--bind', '-e': '--env', '-w': '--pwd'}[arg],
                            docker[index + 1]]
            index += 1
        index += 1
    if threads := option(argv, '--cpu-threads'):
        if int(threads):
            # USD can be imported by the image entrypoint before worker().
            command += ['--env', f'PXR_WORK_THREAD_LIMIT={threads}']
    command += [str(sif), '/bin/bash', '-c', docker[-1]]
    return command


def main():
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != '--sif':
        raise ValueError('Specify --sif IMAGE followed by -m MODULE and --run-dir')
    sif, argv = Path(argv[1]).resolve(), argv[2:]
    if not sif.is_file():
        raise FileNotFoundError(sif)
    run = Path(option(argv, '--run-dir')).resolve()
    profile = json.loads((REPO / '.venv-robodojo/runtime.json').read_text())
    state = {'runtime': 'apptainer', 'pid': os.getpid(), 'image': profile['image'],
             'sif': str(sif), 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES']}

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    with tempfile.TemporaryDirectory(prefix='embodiedevo-sif-',
            dir=os.environ.get('EMBODIEDEVO_SIMULATOR_TMPDIR')) as folder:
        bridge = Path(folder)
        (bridge / 'work').mkdir()
        (bridge / 'home').mkdir()
        state['temporary_directory'] = folder
        write_json(run / 'simulator-container.json', dict(state, status='starting'))
        process = None
        stderr_tail = deque(maxlen=16)
        reader = None
        try:
            command = apptainer_command(argv, profile, bridge, sif)
            env = dict(os.environ, NVIDIA_VISIBLE_DEVICES=state['physical_gpu'],
                       NVIDIA_DRIVER_CAPABILITIES='compute,utility,graphics,display,video')
            if cpus := env.get('EMBODIEDEVO_SIMULATOR_CPUSET'):
                os.sched_setaffinity(0, {int(c) for c in cpus.split(',')})
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE)
            def drain():
                while chunk := process.stderr.read(4096):
                    stderr_tail.append(chunk)
            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            result = process.wait()
            state['exit_code'] = result
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if reader is not None:
                reader.join(timeout=5)
            state['stderr_tail'] = b''.join(stderr_tail).decode(errors='replace')
            for log in bridge.glob('kit/logs/Kit/Isaac-Sim/*/*.log'):
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size - 2*1024**2))
                    (run / 'native-kit-tail.log').write_bytes(stream.read())
            write_json(run / 'simulator-container.json', dict(state, status='stopped'))
    return result


if __name__ == '__main__':
    raise SystemExit(main())
