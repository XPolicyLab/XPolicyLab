import base64
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from ..client import ContinualClient, atomic_json, validate_reply
from ..broker import Broker
from ..remote import RemoteContinualClient

TOOLS = [{'function': {'name': 'give_up', 'parameters': {}}},
         {'function': {'name': 'move_eef', 'parameters': {'properties': {'targets': {'properties': {'left_x': {}}}}}}}]
MESSAGES = [{'role': 'system', 'content': 'test embodiment'}, {'role': 'user', 'content': 'test goal'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'own test state'},
             {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(b'synthetic-no-robot-image').decode()}}]}]
REPLY = {'name': 'give_up', 'arguments': {'reason': 'synthetic transport check', 'hindsight': 'not a robot trial'}}


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.skill = self.root / 'skill'
        (self.skill / 'chain').mkdir(parents=True)
        self.ledger = self.skill / 'chain/rings.jsonl'
        self.ledger.write_text(json.dumps({'index': 0, 'ring_hash': 'synthetic-provenance-only'}) + '\n')
        self.original = self.ledger.read_bytes()
        (self.skill / 'active_ops.py').write_text('OPS = {}\n')
        self.env = {'CYPHER_TEMPRE_SKILL_ROOT': str(self.skill), 'CYPHER_TEMPRE_STREAM_ROOT': str(self.root / 'stream'),
                    'CYPHER_TEMPRE_REPLY_TIMEOUT': '2'}

    def tearDown(self):
        self.temp.cleanup()

    def answer_when_requested(self, stream):
        for _ in range(100):
            requests = list((Path(stream) / 'sessions').glob('*/request-*.json'))
            for request in requests:
                reply = request.with_name(request.name.replace('request-', 'reply-'))
                if not reply.exists():
                    atomic_json(reply, REPLY)
                    return
            time.sleep(.01)
        raise AssertionError('No synthetic request arrived')

    def test_memory_and_new_faculty_survive_two_episode_boundaries(self):
        for episode in range(2):
            client = ContinualClient(self.env)
            worker = threading.Thread(target=self.answer_when_requested, args=(client.root,))
            worker.start()
            result = client.complete(MESSAGES, TOOLS)
            worker.join()
            self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'], 'give_up')
            client.finish()
            if episode == 0:
                (self.skill / 'generated_faculty.py').write_text('experience = "retained"\n')
            self.assertEqual(self.ledger.read_bytes(), self.original)
            self.assertEqual((self.skill / 'generated_faculty.py').read_text(), 'experience = "retained"\n')

    def test_missing_memory_fails_instead_of_starting_empty(self):
        self.env['CYPHER_TEMPRE_SKILL_ROOT'] = str(self.root / 'missing')
        with self.assertRaises(FileNotFoundError):
            ContinualClient(self.env)
        self.assertFalse((self.root / 'missing/chain').exists())

    def test_invalid_or_unavailable_target_data_rejected(self):
        for targets in [{'left_x': float('nan')}, {'left_x': True}, {'hidden_object_x': .1}, {}]:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                validate_reply({'name': 'move_eef', 'arguments': {'targets': targets, 'note': 'test'}}, TOOLS)

    def test_remote_transport_auth_single_stream_and_same_memory(self):
        token = 'synthetic-test-token-' * 3
        server = Broker(('127.0.0.1', 0), self.env, token)
        serving = threading.Thread(target=server.serve_forever)
        serving.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/sessions/' + '0' * 32, data=b'{}', method='POST'), timeout=2)
            self.assertEqual(error.exception.code, 401)
            remote = RemoteContinualClient({**self.env, 'CYPHER_TEMPRE_BROKER_URL': base, 'CYPHER_TEMPRE_BROKER_TOKEN': token})
            worker = threading.Thread(target=self.answer_when_requested, args=(self.env['CYPHER_TEMPRE_STREAM_ROOT'],))
            worker.start()
            self.assertEqual(remote.complete(MESSAGES, TOOLS)['choices'][0]['finish_reason'], 'tool_calls')
            worker.join()
            other = RemoteContinualClient({**self.env, 'CYPHER_TEMPRE_BROKER_URL': base, 'CYPHER_TEMPRE_BROKER_TOKEN': token})
            with self.assertRaises(HTTPError) as conflict:
                other._post('/sessions/' + other.session, {})
            self.assertEqual(conflict.exception.code, 409)
            remote.finish()
            self.assertIsNone(server.active)
            self.assertEqual(self.ledger.read_bytes(), self.original)
        finally:
            server.shutdown()
            server.server_close()
            serving.join()

    def test_external_plain_http_is_refused(self):
        with self.assertRaises(ValueError):
            RemoteContinualClient({'CYPHER_TEMPRE_BROKER_URL': 'http://example.com', 'CYPHER_TEMPRE_BROKER_TOKEN': 'x' * 32})


if __name__ == '__main__':
    unittest.main()
