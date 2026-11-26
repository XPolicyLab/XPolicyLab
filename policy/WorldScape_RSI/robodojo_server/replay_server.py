"""Recovery protocol around the official binary WebSocket envelope."""

import asyncio
from contextvars import ContextVar
import hmac
import json
import logging
import time
from uuid import uuid4

from client_server.ws.protocol.codec import decode_envelope, encode_frame
from client_server.ws.protocol.exceptions import ErrorCode, WsError
from client_server.ws.protocol.messages import MessageType as M
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.server import serve

from .replay import Ledger, PROTOCOL, ReplayError, fingerprint

LOG = logging.getLogger(__name__)
CONNECTION = ContextVar('policy_connection', default=None)


class ReplayServerMixin:
    async def start(self):
        if self._server is None:
            self._server = await serve(self._handle_connection, self.config.host, self.config.port,
                max_size=64*1024*1024, max_queue=4, compression=None,
                ping_interval=self.config.ws_ping_interval_s, ping_timeout=self.config.ws_ping_timeout_s)

    def init_replay(self, ledger=None, disconnect_grace=180, replay_token=None):
        self.ledger = ledger or Ledger()
        self.disconnect_grace = disconnect_grace
        self.replay_token = replay_token
        self.trials = {}  # Never discard tombstones during this process lifetime.
        self.prepared_cases = {}
        self.connections = {}
        self.worker_generations = {}
        self.metrics = dict(replay_done=0, replay_pending=0, conflicts=0, not_admitted=0, reconnects=0)

    def replay_error(self, frame, reason, *, admitted=False):
        message = {'CAPACITY_EXHAUSTED': 'Session capacity exhausted',
                   'BATCH_CAPACITY_EXHAUSTED': 'Batch capacity exhausted'}.get(reason, reason)
        return self._error_reply(frame, ErrorCode.CALL_FAILED, message, dict(
            reason=reason, admitted=admitted, server_instance_id=self._instance_id))

    def _trial(self, key):
        if key not in self.trials:
            if len(self.trials) >= self.ledger.max_records:
                raise ReplayError('NOT_ADMITTED')
            self.trials[key] = dict(state='not_created', owner=None, detached_at=None,
                                    session_generation=None, worker_generation=None,
                                    strict=False, task_identity={})
        return self.trials[key]

    def _worker_stamp(self, index):
        pool = getattr(self.manager, 'pool', self.manager)
        workers = getattr(pool, '_workers', None)
        worker = workers[index] if workers is not None else self.manager
        process = getattr(worker, 'process', None)
        stamp = (id(worker), id(process), getattr(process, 'pid', None))
        prior = self.worker_generations.get(index)
        if prior is None or prior[0] != stamp:
            self.worker_generations[index] = (stamp, uuid4().hex)
        alive = (process is None or process.is_alive()) and index not in getattr(pool, '_faulted_workers', set())
        return self.worker_generations[index][1], alive

    def _check_worker(self, key):
        entry, trial = self.sessions.get(key), self.trials.get(key)
        if entry is None or trial is None:
            return
        generation, alive = self._worker_stamp(entry['worker'])
        if generation != trial['worker_generation'] or not alive:
            trial['state'] = 'lost'
            entry['closing'] = True
            raise ReplayError('WORKER_GENERATION_CHANGED', admitted=True)

    def _check_deadline(self, key):
        trial = self.trials.get(key)
        if (trial and trial['state'] == 'detached'
                and time.monotonic() >= trial['detached_at'] + self.disconnect_grace):
            trial['state'] = 'closing'
            if key in self.sessions:
                self.sessions[key]['closing'] = True

    def _attach(self, key, connection):
        self._check_deadline(key)
        trial = self.trials.get(key)
        if trial is None:
            return  # HELLO does not create a model session or ledger scope.
        self._check_worker(key)
        if trial['state'] in ('closing', 'ended', 'lost'):
            raise ReplayError('SESSION_LOST' if trial['state'] == 'lost' else 'SESSION_ENDED')
        if trial['owner'] is not None and trial['owner'] != connection['id']:
            self.metrics['reconnects'] += 1
        trial['owner'] = connection['id']
        trial['detached_at'] = None
        if trial['state'] == 'detached':
            trial['state'] = 'active'
        connection['keys'].add(key)

    def _detach(self, connection):
        for key in connection['keys']:
            trial = self.trials.get(key)
            if trial and trial['owner'] == connection['id'] and trial['state'] in ('active', 'not_created'):
                trial['state'] = 'detached'
                trial['detached_at'] = time.monotonic()
                LOG.info('session_detached evaluation=%s trial=%s boot=%s', *key, self._instance_id)

    async def _handle_connection(self, websocket):
        if len(self.connections) >= 120:
            await websocket.close(code=1013, reason='NOT_ADMITTED')
            return
        connection = dict(id=uuid4().hex, keys=set(), authenticated=False, strict=False)
        self.connections[connection['id']] = connection
        token = CONNECTION.set(connection)
        waiters = set()
        send_lock = asyncio.Lock()

        async def respond(frame):
            request_key = (frame.evaluation_id, frame.trial_id, frame.request_id)
            process = self.process_frame(frame)
            del frame  # Waiters must not retain multi-camera observations.
            try:
                response = await process
                if response is None:
                    await websocket.close()
                    return
                async with send_lock:
                    await websocket.send(encode_frame(response))
                record = self.ledger.records.get(request_key)
                if record:
                    record['sent_at'] = time.monotonic()
                LOG.info('response_sent evaluation=%s trial=%s request=%s',
                         *request_key)
            except ConnectionClosed as exc:
                LOG.info('response_detached close=%s', exc)

        try:
            async for raw in websocket:
                if not isinstance(raw, bytes):
                    continue
                try:
                    frame = decode_envelope(raw)
                except WsError as exc:
                    await self._reject_invalid_frame(websocket, send_lock, raw, exc)
                    continue
                # HELLO establishes ownership before later messages can run.
                if frame.message_type == M.HELLO:
                    await respond(frame)
                elif len(waiters) >= self.ledger.per_session_pending + 4:
                    await websocket.send(encode_frame(self.replay_error(frame, 'NOT_ADMITTED')))
                else:
                    task = asyncio.create_task(respond(frame))
                    waiters.add(task)
                    task.add_done_callback(waiters.discard)
                raw = frame = None
        except ConnectionClosed as exc:
            LOG.info('socket_closed code_reason=%s', exc)
        finally:
            # Detach immediately, not after a long inference/drain timeout.
            self._detach(connection)
            self.connections.pop(connection['id'], None)
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            CONNECTION.reset(token)

    def _binding(self, frame, trial, connection):
        binding = frame.payload.get('_replay', {})
        strict = (connection and connection['strict']) or (trial and trial['strict'])
        if not strict and not binding:
            return
        if not connection or not connection['authenticated']:
            raise ReplayError('AUTH_REQUIRED')
        if not isinstance(binding, dict) or binding.get('server_instance_id') != self._instance_id:
            raise ReplayError('SERVER_INSTANCE_CHANGED')
        if trial and trial['session_generation']:
            original = self.ledger.records.get((frame.evaluation_id, frame.trial_id, frame.request_id))
            if original and original.get('creation') and not binding.get('session_generation') and not binding.get('worker_generation'):
                return
            for field, reason in [('session_generation', 'SESSION_LOST'),
                                  ('worker_generation', 'WORKER_GENERATION_CHANGED')]:
                if binding.get(field) != trial[field]:
                    raise ReplayError(reason)

    def _status_result(self, key, request_id):
        self.ledger.expire()
        self._check_deadline(key)
        try:
            self._check_worker(key)
        except ReplayError:
            pass
        trial = self.trials.get(key, {})
        record = self.ledger.records.get((*key, request_id))
        state = record['state'] if record else ('missing' if trial.get('state', 'not_created') in ('not_created', 'active', 'detached') else 'uncertain')
        if trial.get('state') in ('lost', 'closing', 'ended'):
            state = 'uncertain'
        return dict(state=state, digest=record['digest'] if record else None,
                    session_state=trial.get('state', 'not_created'), server_instance_id=self._instance_id,
                    session_generation=trial.get('session_generation'),
                    worker_generation=trial.get('worker_generation'))

    async def process_frame(self, frame):
        key = (frame.evaluation_id, frame.trial_id)
        connection = CONNECTION.get()
        name = frame.payload.get('func_name') if frame.message_type == M.CALL else None
        try:
            if frame.message_type == M.HELLO:
                if connection:
                    offered = frame.payload.get('replay')
                    if offered is not None:
                        if not isinstance(offered, dict):
                            raise ReplayError('REPLAY_UNSUPPORTED')
                        # Internal-network deployments enable recovery by default.
                        # An explicitly configured token remains an extra gate.
                        if self.replay_token and (not isinstance(offered.get('token'), str)
                                or not hmac.compare_digest(offered['token'], self.replay_token)):
                            raise ReplayError('AUTH_REQUIRED')
                        if offered.get('protocol') != PROTOCOL:
                            raise ReplayError('REPLAY_UNSUPPORTED')
                        expected = offered.get('server_instance_id')
                        if expected is not None and expected != self._instance_id:
                            raise ReplayError('SERVER_INSTANCE_CHANGED')
                        connection['authenticated'] = connection['strict'] = True
                    trial = self.trials.get(key)
                    if trial and trial['strict'] and not connection['strict']:
                        raise ReplayError('AUTH_REQUIRED')
                    if trial and connection['strict']:
                        if not offered or offered.get('server_instance_id') != self._instance_id:
                            raise ReplayError('SERVER_INSTANCE_CHANGED')
                    self._attach(key, connection)
                response = await super()._dispatch_frame(frame)
                if connection and connection['strict']:
                    response.payload.update(replay_protocol=PROTOCOL,
                        replay_retention_s=self.ledger.retention, disconnect_grace_s=self.disconnect_grace,
                        fingerprint='tagged-sha256-v1', durability='process_memory')
                    response.payload['token_required'] = bool(self.replay_token)
                    response.payload.update(self._status_result(key, ''))
                return response
            if frame.message_type in (M.HEARTBEAT, M.CLOSE):
                return await super()._dispatch_frame(frame)
            if name == '_replay_status':
                if not connection or not connection['authenticated']:
                    raise ReplayError('AUTH_REQUIRED')
                # Query pins the boot but can discover session generations after
                # losing a creation response; it never invokes the model.
                binding = frame.payload.get('_replay', {})
                if not isinstance(binding, dict) or binding.get('server_instance_id') != self._instance_id:
                    raise ReplayError('SERVER_INSTANCE_CHANGED')
                obs = frame.payload.get('obs', {})
                if not isinstance(obs, dict) or not isinstance(obs.get('request_id'), str):
                    raise ReplayError('INVALID_REQUEST')
                return self._reply(frame, M.CALL_RESULT, {'ok': True, 'result': self._status_result(key, obs['request_id'])})
            if name in ('online_info', 'session_status', 'get_runtime_info'):
                if self.replay_token and (not connection or not connection['authenticated']):
                    raise ReplayError('AUTH_REQUIRED')
                return await self._dispatch_frame(frame)
            pretrial = frame.message_type == M.PREPARE_CASE and not frame.trial_id
            identities = key[:1] if pretrial else key
            if (not all(isinstance(v, str) and v.strip() and len(v) <= 256 for v in identities)
                    or (frame.action_case_id is not None and len(frame.action_case_id) > 256)
                    or not isinstance(frame.request_id, str) or not 1 <= len(frame.request_id) <= 256):
                raise ReplayError('INVALID_REQUEST')
            if self._stopping:
                raise ReplayError('NOT_ADMITTED')
            trial = self.trials.get(key)
            self._binding(frame, trial, connection)
            self._check_deadline(key)
            self._check_worker(key)
            digest = fingerprint(frame)
            request_key = (*key, frame.request_id)
            self.ledger.expire()
            record = self.ledger.records.get(request_key)
            if record:
                if digest != record['digest']:
                    self.metrics['conflicts'] += 1
                    raise ReplayError('REQUEST_ID_CONFLICT', admitted=True)
                if record['state'] == 'done':
                    self.metrics['replay_done'] += 1
                    LOG.info('replay_hit state=done evaluation=%s trial=%s request=%s', *key, frame.request_id)
                    return decode_envelope(record['response'])
                if record['state'] == 'pending':
                    self.metrics['replay_pending'] += 1
                    LOG.info('replay_hit state=pending evaluation=%s trial=%s request=%s', *key, frame.request_id)
                    del frame
                    return await asyncio.shield(record['task'])
                raise ReplayError('REPLAY_EXPIRED' if record['state'] == 'expired' else 'REQUEST_UNCERTAIN', admitted=True)
            ending = frame.message_type == M.TRIAL_END or name == 'on_trial_end'
            if trial and trial['state'] in ('ended', 'closing', 'lost') and not (ending and trial['state'] == 'ended'):
                raise ReplayError('SESSION_LOST' if trial['state'] == 'lost' else 'SESSION_ENDED')
            # The pinned official legacy SDK sends HELLO without trial_id.
            # Its first scoped request can attach a detached legacy trial.
            if trial and connection and not trial['strict'] and trial['state'] == 'detached':
                self._attach(key, connection)
            if trial and connection and trial['owner'] not in (None, connection['id']):
                raise ReplayError('CONNECTION_FENCED')
            if trial and trial['strict'] and (not connection or not connection['strict']):
                raise ReplayError('AUTH_REQUIRED')
            # Register without any await, before allocation, decode or dispatch.
            wire = encode_frame(frame)
            trial = self._trial(key)
            try:
                record = self.ledger.register(request_key, digest, len(wire))
            except (MemoryError, OSError):
                raise ReplayError('NOT_ADMITTED') from None
            record['creation'] = trial['session_generation'] is None
            if connection:
                trial['owner'] = connection['id']
                trial['strict'] = trial['strict'] or connection['strict']
                connection['keys'].add(key)
            scoped = json.dumps(request_key, separators=(',', ':'))
            self._arrivals[scoped] = time.monotonic()
            task = asyncio.create_task(self._execute_record(decode_envelope(wire), record, scoped))
            record['task'] = task
            self._inflight[scoped] = (frame.message_type, task)
            task.add_done_callback(lambda _, rid=scoped: self._inflight.pop(rid, None))
            del frame, wire
            return await asyncio.shield(task)
        except ReplayError as exc:
            if exc.reason == 'NOT_ADMITTED':
                self.metrics['not_admitted'] += 1
            return self.replay_error(frame, exc.reason, admitted=exc.admitted)
        except (ValueError, TypeError) as exc:
            LOG.warning('invalid_replay_request type=%s', type(exc).__name__)
            return self.replay_error(frame, 'INVALID_REQUEST')

    async def _execute_record(self, frame, record, scoped):
        key = (frame.evaluation_id, frame.trial_id)
        try:
            try:
                response = await self._dispatch_frame(frame)
            except WsError as exc:
                LOG.exception('request_failed evaluation=%s trial=%s request=%s', *key, frame.request_id)
                response = self._error_reply(frame, exc.code, 'Policy request failed', {
                    'reason': 'MODEL_ERROR', 'admitted': True, 'batch_state': 'uncertain'})
            except ReplayError as exc:
                response = self.replay_error(frame, exc.reason, admitted=exc.admitted)
            except Exception:
                LOG.exception('request_failed evaluation=%s trial=%s request=%s', *key, frame.request_id)
                response = self.replay_error(frame, 'REQUEST_UNCERTAIN', admitted=True)
                self.trials[key]['state'] = 'lost'
            trial = self.trials[key]
            if trial['strict']:
                response.payload['replay'] = dict(server_instance_id=self._instance_id,
                    session_generation=trial['session_generation'], worker_generation=trial['worker_generation'])
            wire = encode_frame(response)
            if len(wire) <= self.ledger.response_reserve:
                record['response'] = wire  # Retain real bytes even if save bookkeeping fails.
            self.ledger.complete(record, wire)
            LOG.info('result_saved evaluation=%s trial=%s request=%s digest=%s boot=%s session=%s worker=%s elapsed=%.6f',
                     *key, frame.request_id, record['digest'], self._instance_id,
                     trial['session_generation'], trial['worker_generation'], time.monotonic()-record['received'])
            return decode_envelope(record['response'])
        except BaseException as exc:
            # Preserve evidence and any already encoded result; never rerun.
            record['state'] = 'uncertain'
            self.trials[key]['state'] = 'lost'
            LOG.error('result_save_failed evaluation=%s trial=%s request=%s type=%s', *key, frame.request_id, type(exc).__name__)
            return self.replay_error(frame, 'REQUEST_UNCERTAIN', admitted=True)
        finally:
            self.ledger.finish_pending(record)
            record['task'] = None
            self._arrivals.pop(scoped, None)
