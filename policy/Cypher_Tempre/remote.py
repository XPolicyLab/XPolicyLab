"""Authenticated observation/action transport to the one continuing agent host."""
import json
import math
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid


class RemoteContinualClient:
    def __init__(self, env):
        self.env = dict(env)
        self.base = self.env['CYPHER_TEMPRE_BROKER_URL'].rstrip('/')
        parsed = urlparse(self.base)
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1', '::1'}):
            raise ValueError('Remote agent transport requires HTTPS; HTTP is limited to loopback.')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use the token environment variable, not credentials in a URL.')
        if not parsed.hostname or parsed.path not in {'', '/'}:
            raise ValueError('The broker URL must name its origin without a path.')
        self.token = self.env['CYPHER_TEMPRE_BROKER_TOKEN']
        if len(self.token) < 32:
            raise ValueError('The broker token must have at least 32 characters.')
        self.timeout = float(self.env.get('CYPHER_TEMPRE_REPLY_TIMEOUT', '600'))
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 7200:
            raise ValueError('Reply timeout must be finite and within 1..7200 seconds.')
        self.session = uuid.uuid4().hex
        self.started = False

    def _post(self, suffix, value):
        request = Request(self.base + suffix, data=json.dumps(value, allow_nan=False).encode('utf-8'),
                          headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=self.timeout + 10) as response:
            return json.load(response)

    def complete(self, messages, tools):
        if not self.started:
            self._post('/sessions/' + self.session, {})
            self.started = True
        current = next(m for m in reversed(messages) if isinstance(m.get('content'), list))
        # The controller already retains its own experience. This transport sends
        # the same current observations/goal/feedback as the local file transport.
        packet = [messages[0], messages[1], *[m for m in messages if m['role'] == 'tool'][-2:], current]
        return self._post('/sessions/' + self.session + '/complete', {'messages': packet, 'tools': tools})

    def finish(self):
        if self.started:
            self._post('/sessions/' + self.session + '/finish', {})
