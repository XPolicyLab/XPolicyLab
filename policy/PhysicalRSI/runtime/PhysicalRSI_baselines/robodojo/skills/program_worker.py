"""Private, inherited-pipe worker for one verified code skill; no policy server."""
import importlib.util
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def run(configuration, pipe):
    processes, streams, reservations = [], [], []
    model = None
    output = Path(configuration['output'])
    try:
        spec = importlib.util.spec_from_file_location('physicalrsi_asset_launch', configuration['launcher'])
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        framework = Path(configuration['framework'])
        descriptor = launch.expand(json.loads(Path(configuration['descriptor']).read_text()), dict(
            assets=configuration['assets'], output=output, python=sys.executable,
            xpolicylab=framework, xpolicylab_parent=framework.parent))
        launch.validate(descriptor)
        for directory in descriptor.get('directories', []):
            Path(directory).mkdir(parents=True, exist_ok=True)
        inherited = dict(os.environ, PHYSICALRSI_PROGRAM_ROOT=configuration['assets'])
        bypass = ','.join(dict.fromkeys(['127.0.0.1', 'localhost', '::1'] + inherited.get('NO_PROXY', '').split(',')))
        inherited.update(NO_PROXY=bypass, no_proxy=bypass)
        reservations = launch.reserve_service_endpoints(descriptor)
        environment = launch.environment(descriptor, inherited)
        # Only this owned worker receives implementation-specific imports and env.
        os.environ.update(environment)
        sys.path[:0] = [str(Path(configuration['launcher']).parent), *descriptor['pythonpath']]
        launch.write_json(output / 'worker.json', descriptor)
        for external in descriptor.get('external_services', []):
            with launch.build_opener(launch.ProxyHandler({})).open(external['health_url'], timeout=5) as response:
                data = json.load(response)
            if any(data.get(key) != value for key, value in external['expected_health'].items()):
                raise RuntimeError('External perception service identity mismatch')
        for service, reservation in zip(descriptor.get('services', []), reservations):
            reservation.close()
            stream = (output / (service['name'] + '.log')).open('w')
            streams.append(stream)
            process = subprocess.Popen(service['command'], cwd=service['cwd'],
                                       env=launch.environment(service, environment),
                                       stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
            deadline = time.monotonic() + configuration['startup_timeout_s']
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Perception service exited: ' + service['name'])
                try:
                    launch.health(service)
                    break
                except (OSError, ValueError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Perception service startup timeout')
                    time.sleep(.2)
        from transport.binding import load_bound_model
        model = load_bound_model(descriptor)
        pipe.send(('ok', {'pid': os.getpid(), 'transport': 'inherited-pipe'}))
        while True:
            method, args = pipe.recv()
            if method == 'close':
                break
            if method not in {'reset', 'update_obs_batch', 'get_action_batch'}:
                raise ValueError('Unsupported code-skill method')
            if any(process.poll() is not None for process in processes):
                raise RuntimeError('Owned perception service exited')
            pipe.send(('ok', getattr(model, method)(*args)))
    except EOFError:
        pass
    except BaseException:
        traceback.print_exc()
        try:
            pipe.send(('error', 'Code skill failed; inspect ' + str(output / 'service.log')))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if model is not None and callable(getattr(model, 'close', None)):
                model.close()
        finally:
            for reservation in reservations:
                reservation.close()
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            for stream in streams:
                stream.close()
            pipe.close()


if __name__ == '__main__':
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    run(json.loads(Path(sys.argv[1]).read_text()), Connection(int(sys.argv[2])))
