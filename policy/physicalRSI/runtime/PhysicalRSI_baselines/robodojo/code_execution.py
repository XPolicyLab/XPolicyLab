"""Run proposed policy(robot, memory) with explicit host primitive capabilities."""

from PhysicalRSI_core.infra.isolation import execute

_DRIVER = """
import json as _json
import time as _time

def _publish(message):
    with open('/output/result.json', 'w') as stream:
        _json.dump(message, stream, allow_nan=False)

class _Robot:
    def __init__(self):
        self._index = 0

    def __getattr__(self, name):
        def call(*args, **kwargs):
            index = self._index
            self._index += 1
            _publish(dict(kind='call', id=index, method=name,
                          args=args, kwargs=kwargs))
            while True:
                with open('/response.json') as stream:
                    reply = _json.load(stream)
                if reply.get('id') == index:
                    return reply['value']
                _time.sleep(0.01)
        return call

_input = _json.load(open('/input.json'))
_scope = {}
exec(compile(_input['source'], '<candidate>', 'exec'), _scope)
_value = _scope['policy'](_Robot(), _input['memory'])
_publish(dict(kind='result', value=_value))
"""


def execute_policy(source, memory, *, handlers, runtime, output, **limits):
    """Handlers receive (args, kwargs, deadline=monotonic_absolute_deadline).

    Handlers are trusted bounded primitive clients, never raw environment objects.
    They must enforce their deadline, validate action contracts, and return only
    policy-visible feedback. This result does not determine episode success.
    """
    return execute(
        _DRIVER,
        dict(source=source, memory=memory),
        runtime=runtime,
        output=output,
        handlers=handlers,
        **limits,
    )
