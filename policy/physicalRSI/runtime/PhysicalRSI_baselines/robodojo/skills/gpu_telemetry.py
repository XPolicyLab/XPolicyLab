"""Record GPU utilization during a skill service's lifetime."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', required=True)
    args = parser.parse_args()
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    samples = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while running:
        sample = {'unix_time': time.time()}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            sample.update(returncode=result.returncode, csv=result.stdout.strip(), error=result.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            sample['error'] = str(exc)
        samples.append(sample)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps({'schema': 'physicalrsi.gpu-telemetry/v1',
            'configuration': args.config, 'seed': args.seed, 'samples': samples}, indent=2)+'\n')
        temporary.replace(args.output)
        for _ in range(20):
            if not running:
                break
            time.sleep(0.1)


if __name__ == '__main__':
    main()
