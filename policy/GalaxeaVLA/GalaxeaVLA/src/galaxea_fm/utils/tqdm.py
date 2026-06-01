import tqdm
import signal


def handle_resize(signum, frame):
    for instance in list(tqdm.tqdm._instances):
        if hasattr(instance, 'refresh'):
            instance.refresh()

signal.signal(signal.SIGWINCH, handle_resize)