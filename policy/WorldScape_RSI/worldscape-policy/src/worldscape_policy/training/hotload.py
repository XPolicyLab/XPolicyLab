"""Collectively adopt immutable RoboDojo snapshots without restarting training.

The durable journal records the *decision* before any rank changes its loader.
Preparation failures leave the old loader usable. Failures after that decision
are fatal; a checkpoint and the journal then replay the same decision boundary.
"""
from __future__ import annotations

import copy
from bisect import bisect_right
from contextlib import contextmanager
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import queue
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist

from worldscape_policy.data.robodojo_snapshots import SnapshotStore, validate_writable_path
from worldscape_policy.training.global_buckets import GlobalBucketLoader


class _PreparationCleanupFailure(RuntimeError):
    """An owned subprocess remains alive; continuing could leak preparations."""


def _digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _same_statistics(left, right):
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_same_statistics(left[key], right[key]) for key in left))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(_same_statistics(a, b) for a, b in zip(left, right)))
    return left == right


def _statistics_digest(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'unsupported statistics value: {type(item).__name__}')
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), default=convert).encode()
    return hashlib.sha256(raw).hexdigest()


def _writable(path: str | Path) -> Path:
    return validate_writable_path(path)


def _prepare_worker(connection, factory, snapshot_dir, descriptor, planner_seed,
                    base_hash, stats_path, stats_hash):
    """Spawned CPU-only worker. Raw pickle avoids torch multiprocessing IPC."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        record = SnapshotStore(snapshot_dir).resolve(descriptor, verify_files=True)
        if record['manifest']['base_index_sha256'] != base_hash:
            raise ValueError('candidate snapshot has a different base index')
        if _digest(stats_path) != stats_hash:
            raise ValueError('z-score statistics changed during training')
        candidate = factory(record['index_path'], planner_seed, validate_distributed=False)
        if not isinstance(candidate, GlobalBucketLoader):
            raise TypeError('hotload loader_factory must return GlobalBucketLoader')
        connection.send_bytes(b'O' + pickle.dumps(candidate, protocol=pickle.HIGHEST_PROTOCOL))
    except BaseException as error:
        connection.send_bytes(b'E' + f'{type(error).__name__}: {error}'.encode())
    finally:
        connection.close()


def _spawn_preparation(factory, snapshot_dir, descriptor, planner_seed,
                       base_hash, stats_path, stats_hash, timeout):
    context = multiprocessing.get_context('spawn')
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_prepare_worker, args=(
        sending, factory, snapshot_dir, descriptor, planner_seed,
        base_hash, stats_path, stats_hash), daemon=True)
    results = queue.Queue(maxsize=1)

    def receive():
        try:
            results.put((True, receiving.recv_bytes()))
        except BaseException as error:
            results.put((False, error))

    receiver = threading.Thread(target=receive, daemon=True, name='hotload-candidate-receive')
    started = time.monotonic()
    try:
        process.start()
        sending.close()
        receiver.start()
        try:
            ok, payload = results.get(timeout=max(.001, timeout - (time.monotonic() - started)))
        except queue.Empty as error:
            raise TimeoutError(f'hotload candidate preparation exceeded {timeout:g} seconds') from error
        if not ok:
            raise RuntimeError(f'hotload preparation worker exited without a candidate: {payload}')
        if payload[:1] != b'O':
            raise RuntimeError(payload[1:].decode(errors='replace'))
        candidate = pickle.loads(payload[1:])
        if not isinstance(candidate, GlobalBucketLoader):
            raise TypeError('preparation worker returned an invalid loader')
        return candidate
    finally:
        sending.close()
        if process.pid is not None:
            process.join(timeout=.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            if process.is_alive():
                raise _PreparationCleanupFailure('hotload preparation worker could not be stopped')
        receiving.close()
        if receiver.ident is not None:
            receiver.join(timeout=2)


@contextmanager
def _isolated_rng():
    """Dataset construction is not allowed to perturb model/dropout randomness."""
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class DynamicGlobalBucketLoader:
    """Versioned wrapper; disabled training continues to use the original loader.

    ``loader_factory(index_path, planner_seed, validate_distributed=False)``
    must perform only local work and leave transform/dataset seeds unchanged.
    Call ``after_train_step`` after training has consumed the yielded batch; a
    true result asks the caller to replace its iterator with ``iter(self)``.
    """

    supports_exact_worker_resume = True
    supports_static_checkpoint_upgrade = True
    FORMAT_VERSION = 1

    def __init__(self, loader: GlobalBucketLoader, *, loader_factory: Callable,
                 snapshot_dir: str | Path, journal_dir: str | Path,
                 base_index_path: str | Path, stats_path: str | Path, seed: int,
                 prepare_timeout_seconds: float = 120):
        if not isinstance(loader, GlobalBucketLoader):
            raise TypeError('hotload requires GlobalBucketLoader')
        self.loader = loader
        self._initial_statistics = copy.deepcopy([
            (getattr(child, 'joint_stats', None), getattr(child, 'velocity_stats', None))
            for child in loader.sampler.mixture.datasets])
        self.loader_factory = loader_factory
        self.snapshot_dir = str(Path(snapshot_dir).resolve())
        self.prepare_timeout_seconds = float(prepare_timeout_seconds)
        if not math.isfinite(self.prepare_timeout_seconds) or self.prepare_timeout_seconds <= 0:
            raise ValueError('prepare_timeout_seconds must be positive and finite')
        self.store = SnapshotStore(snapshot_dir)
        self.journal_dir = _writable(journal_dir)
        self.base_index_path = str(Path(base_index_path).resolve())
        self.stats_path = str(Path(stats_path).resolve())
        self.seed = int(seed)
        contract = {'error': None, 'seed': self.seed}
        try:
            statistics_bytes = Path(self.stats_path).read_bytes()
            statistics_values = json.loads(statistics_bytes)
            if any(joint is not None and not _same_statistics(joint, statistics_values)
                   for joint, _ in self._initial_statistics):
                raise ValueError('initial dataset normalization values differ from the statistics file')
            contract.update(base_index_sha256=_digest(self.base_index_path),
                            stats_sha256=hashlib.sha256(statistics_bytes).hexdigest(),
                            normalization_sha256=_statistics_digest(self._initial_statistics))
        except Exception as error:
            contract['error'] = f'{type(error).__name__}: {error}'
        contracts = self._gather(contract)
        if any(item.get('error') or item != contracts[0] for item in contracts):
            raise ValueError(f'hotload base index, statistics or seed differ across ranks: {contracts}')
        self.base_index_sha256 = contract['base_index_sha256']
        self.stats_sha256 = contract['stats_sha256']
        self.normalization_sha256 = contract['normalization_sha256']
        self.active = self._leader_call(lambda: self.store.initialize(self.base_index_path))
        self.initial = copy.deepcopy(self.active)
        self.planner_seed = self.seed
        self.schedule_prefix: list[int] = []
        self.journal_sequence = 0
        self.journal_sha256 = None
        self.step = 0
        self.data_batches_consumed = 0
        self._switch_pending = False
        self._last_error = None
        self._last_event = None
        self._replay_cache_ready = False
        self._replay_cache = None
        self._deferred_count = 0
        self._prepare_seconds = 0.0
        self._switch_seconds = 0.0
        self._new_samples = 0

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    @classmethod
    def _rank(cls) -> int:
        return dist.get_rank() if cls._distributed() else 0

    @classmethod
    def _gather(cls, value):
        if not cls._distributed():
            return [value]
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        return values

    @classmethod
    def _leader_call(cls, callback):
        packet = None
        if cls._rank() == 0:
            try:
                packet = {'value': callback(), 'error': None}
            except Exception as error:
                packet = {'value': None, 'error': f'{type(error).__name__}: {error}'}
        if cls._distributed():
            payload = [packet]
            dist.broadcast_object_list(payload, src=0)
            packet = payload[0]
        if packet['error']:
            raise RuntimeError(packet['error'])
        return packet['value']

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def num_workers(self):
        return self.loader.num_workers

    @property
    def last_batch_info(self):
        return dict(self.loader.last_batch_info,
                    data_version=int(self.active['version']),
                    data_manifest_sha256=self.active['manifest_sha256'],
                    hotload_journal_sequence=self.journal_sequence,
                    hotload_deferred=self._last_error,
                    **{'hotload/version': int(self.active['version']),
                       'hotload/episodes': sum(len(child.paths) for child in self.sampler.mixture.datasets),
                       'hotload/samples': sum(len(child.index) for child in self.sampler.mixture.datasets),
                       'hotload/new_samples': self._new_samples,
                       'hotload/prepare_seconds': self._prepare_seconds,
                       'hotload/switch_seconds': self._switch_seconds,
                       'hotload/deferred_count': self._deferred_count})

    def _defer(self, reason):
        self._last_error = str(reason)
        self._deferred_count += 1
        if self._rank() == 0:
            print(json.dumps({'hotload': 'deferred', 'reason': self._last_error,
                              'version': self.active['version'], 'step': self.step,
                              'data_batches_consumed': self.data_batches_consumed}), flush=True)

    def __iter__(self):
        yield from self.loader

    def close(self):
        self.loader.close()

    def _event_path(self, sequence):
        return self.journal_dir / 'events' / f'{sequence:08d}.json'

    def _read_event(self, sequence, previous_hash):
        path = self._event_path(sequence)
        if not path.exists():
            if any(int(item.stem) > sequence for item in path.parent.glob('[0-9]*.json')):
                raise ValueError('hotload journal has a missing event in its history')
            return None
        raw = path.read_bytes()
        event = json.loads(raw)
        if (event.get('format_version') != self.FORMAT_VERSION
                or event.get('sequence') != sequence
                or event.get('previous_sha256') != previous_hash):
            raise ValueError('hotload journal chain does not match checkpoint')
        return {'event': event, 'sha256': hashlib.sha256(raw).hexdigest()}

    def _write_event(self, event):
        path = _writable(self._event_path(event['sequence']))
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = _json_bytes(event)
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError('hotload journal decision already exists with different contents')
        else:
            descriptor, temporary = tempfile.mkstemp(prefix='.event-', dir=path.parent)
            try:
                with os.fdopen(descriptor, 'wb') as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {'event': event, 'sha256': hashlib.sha256(raw).hexdigest()}

    def _verify_journal_prefix(self, sequence, expected_hash):
        digest = None
        for number in range(1, sequence + 1):
            entry = self._read_event(number, digest)
            if entry is None:
                raise ValueError('checkpoint references a missing hotload journal event')
            digest = entry['sha256']
        if digest != expected_hash:
            raise ValueError('checkpoint hotload journal digest does not match')

    def _version_seed(self, version):
        if version == 0:
            return self.seed
        return int(np.random.SeedSequence([self.seed, int(version), 0x484F54])
                   .generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))

    def _candidate(self, descriptor, planner_seed):
        if getattr(self.loader_factory, 'isolated_preparation', False):
            candidate = _spawn_preparation(
                self.loader_factory, self.snapshot_dir, descriptor, planner_seed,
                self.base_index_sha256, self.stats_path, self.stats_sha256,
                self.prepare_timeout_seconds)
        else:
            record = self.store.resolve(descriptor, verify_files=True)
            if record['manifest']['base_index_sha256'] != self.base_index_sha256:
                raise ValueError('candidate snapshot has a different base index')
            if _digest(self.stats_path) != self.stats_sha256:
                raise ValueError('z-score statistics changed during training')
            candidate = self.loader_factory(record['index_path'], planner_seed,
                                            validate_distributed=False)
        if not isinstance(candidate, GlobalBucketLoader):
            raise TypeError('hotload loader_factory must return GlobalBucketLoader')
        actual = [(getattr(child, 'joint_stats', None), getattr(child, 'velocity_stats', None))
                  for child in candidate.sampler.mixture.datasets]
        if not _same_statistics(self._initial_statistics, actual):
            candidate.close()
            raise ValueError('candidate normalization values differ from initial in-memory statistics')
        return candidate

    @staticmethod
    def _prefix_schedule(loader, prefix):
        """Synthetic occurrences make carried references valid in a new schedule.

        They are never sampled again: the committed scan position starts AFTER
        this prefix. Multiple draws of one frame remain distinct occurrences.
        """
        sampler = loader.sampler
        mixture = sampler.mixture
        if any(not 0 <= int(shard) < len(mixture.shards) for shard in prefix):
            raise ValueError('invalid hotload carried shard reference')
        if prefix:
            mixture.schedule = np.concatenate((np.asarray(prefix, dtype=np.int64),
                                               mixture.schedule))
        sampler.__init__(mixture, sampler.batch_size, sampler.window_batches)

    def _carry(self, candidate):
        old = self.loader.sampler
        new = candidate.sampler
        if len(old.mixture.datasets) != len(new.mixture.datasets):
            raise ValueError('hotload changed the mixture structure')
        for before, after in zip(old.mixture.datasets, new.mixture.datasets):
            if (len(after.index) < len(before.index)
                    or not np.array_equal(before.index, after.index[:len(before.index)])
                    or list(map(str, before.paths)) != list(map(str, after.paths[:len(before.paths)]))):
                raise ValueError('hotload snapshot must preserve the existing index prefix')
            old_artifact = getattr(before, '_precomputed_index', None)
            if old_artifact is not None:
                new_artifact = getattr(after, '_precomputed_index', None)
                if new_artifact is None:
                    raise ValueError('hotload candidate lost precomputed metadata')
                for field in ('lengths', 'sampling_progress'):
                    if not np.array_equal(old_artifact[field],
                                          new_artifact[field][:len(before.index)]):
                        raise ValueError(f'hotload changed existing {field} metadata')
                old_manifest, new_manifest = old_artifact['manifest'], new_artifact['manifest']
                for field in ('episodes', 'episode_lengths', 'source_files'):
                    if old_manifest[field] != new_manifest[field][:len(old_manifest[field])]:
                        raise ValueError(f'hotload changed existing {field} metadata')
                for field in ('options', 'tasks'):
                    if old_manifest.get(field) != new_manifest.get(field):
                        raise ValueError(f'hotload changed fixed {field} metadata')
        old_state = old.state_dict()['state']
        remaining = old_state['ready'][old_state['delivered']:]
        # The same old schedule occurrence may contribute to several buckets.
        # Keep its identity without deduplicating any individual sample draw.
        mapping = {}
        prefix = []
        shards = {}
        for shard_id, shard in enumerate(new.mixture.shards):
            shards.setdefault(shard.dataset_index, []).append((shard.start, shard.stop, shard_id))
        shard_lookup = {}
        for dataset_index, ranges in shards.items():
            ranges.sort()
            if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
                raise ValueError('hotload candidate contains overlapping shard ranges')
            shard_lookup[dataset_index] = [item[0] for item in ranges], ranges

        def translate(ref):
            position, index = map(int, ref)
            old_shard = old.mixture.shards[int(old.mixture.schedule[position])]
            starts, ranges = shard_lookup[old_shard.dataset_index]
            location = bisect_right(starts, index) - 1
            if location < 0 or index >= ranges[location][1]:
                raise ValueError('cannot resolve a carried sample in the candidate snapshot')
            shard_id = ranges[location][2]
            occurrence = position, shard_id
            if occurrence not in mapping:
                mapping[occurrence] = len(prefix)
                prefix.append(shard_id)
            return mapping[occurrence], index

        ready = [(key, [translate(ref) for ref in refs]) for key, refs in remaining]
        pending = {key: [translate(ref) for ref in refs]
                   for key, refs in old_state['pending'].items()}
        self._prefix_schedule(candidate, prefix)
        state = candidate.sampler._initial(0)
        state.update(position=len(prefix), batches=old_state['batches'],
                     ready=ready, pending=pending)
        candidate.sampler.load_state_dict({'identity': candidate.sampler.identity, 'state': state})
        return prefix

    def after_train_step(self, step: int, data_batches_consumed: int,
                         optimizer_boundary: bool, *, _replay_only=False) -> bool:
        self.step = int(step)
        self.data_batches_consumed = int(data_batches_consumed)
        committed = self.loader.sampler.committed
        if (not _replay_only and committed['ready']
                and committed['delivered'] == len(committed['ready'])):
            self._switch_pending = True
        if not optimizer_boundary:
            return False
        # Journal replay takes precedence, including when a switch originally
        # occurred just after an accumulation crossed a planner window boundary.
        if not self._replay_cache_ready:
            self._replay_cache = self._leader_call(lambda: self._read_event(
                self.journal_sequence + 1, self.journal_sha256))
            self._replay_cache_ready = True
        packet = self._replay_cache
        replay = packet is not None
        if replay:
            event = packet['event']
            target = int(event['data_batches_consumed'])
            if target < self.data_batches_consumed:
                raise RuntimeError('training passed a recorded hotload adoption boundary')
            if target != self.data_batches_consumed:
                # Continuous execution consumed any pending window notification
                # at this optimizer boundary, even when latest was unchanged.
                # Replay must preserve that same checkpointable control state.
                if not _replay_only:
                    self._switch_pending = False
                return False
            if int(event['step']) != self.step or event['from_snapshot'] != self.active:
                raise RuntimeError('hotload journal replay disagrees with training progress')
            descriptor = event['to_snapshot']
            planner_seed = int(event['planner_seed'])
        else:
            if _replay_only:
                return False
            if not self._switch_pending:
                return False
            self._switch_pending = False
            try:
                descriptor = self._leader_call(self.store.latest)
            except RuntimeError as error:
                self._defer(error)
                return False
            if descriptor is None or descriptor == self.active:
                return False
            if int(descriptor['version']) <= int(self.active['version']):
                self._defer('latest snapshot version regressed or was replaced')
                return False
            planner_seed = self._version_seed(int(descriptor['version']))
        # Everything here may fail locally. No local constructor may initiate a
        # collective: gather readiness only after every rank reaches this point.
        candidate = None
        prefix = None
        error = None
        fatal = False
        prepare_started = time.perf_counter()
        try:
            with _isolated_rng():
                candidate = self._candidate(descriptor, planner_seed)
                prefix = self._carry(candidate)
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
            fatal = isinstance(failure, _PreparationCleanupFailure)
        preparation = self._gather({'error': error, 'fatal': fatal,
                                    'identity': None if error else candidate.sampler.identity,
                                    'state_sha256': None if error else hashlib.sha256(
                                        pickle.dumps(candidate.sampler.committed, protocol=4)).hexdigest(),
                                    'step': self.step, 'consumed': self.data_batches_consumed})
        failures = [entry['error'] for entry in preparation if entry['error']]
        self._prepare_seconds = time.perf_counter() - prepare_started
        if not failures and any(entry != preparation[0] for entry in preparation):
            failures = ['candidate identity or training progress differs across ranks']
        if failures:
            if candidate is not None:
                candidate.close()
            self._defer('; '.join(failures))
            if replay or any(entry['fatal'] for entry in preparation):
                raise RuntimeError(f'cannot replay committed data version: {self._last_error}')
            return False
        # A durable decision must precede switching. The old iterator remains
        # intact if journal publication itself fails before the commit decision.
        if not replay:
            event = dict(format_version=self.FORMAT_VERSION,
                         sequence=self.journal_sequence + 1,
                         previous_sha256=self.journal_sha256,
                         from_snapshot=self.active, to_snapshot=descriptor,
                         step=self.step, data_batches_consumed=self.data_batches_consumed,
                         planner_seed=planner_seed)
            try:
                packet = self._leader_call(lambda: self._write_event(event))
            except Exception:
                candidate.close()
                # The durable file may have been installed before fsync failed.
                # Treat this ambiguous decision as fatal, never silently retry.
                raise
        switch_started = time.perf_counter()
        self._new_samples = sum(len(child.index) for child in candidate.sampler.mixture.datasets) - sum(
            len(child.index) for child in self.sampler.mixture.datasets)
        self.loader.close()
        self.loader = candidate
        self.active = copy.deepcopy(descriptor)
        self.planner_seed = planner_seed
        self.schedule_prefix = prefix
        self.journal_sequence = int(packet['event']['sequence'])
        self.journal_sha256 = packet['sha256']
        self._last_event = packet['event']
        self._switch_pending = False
        self._last_error = None
        self._replay_cache_ready = False
        self._replay_cache = None
        # No rank may hand out a batch until every rank has adopted the decision.
        statuses = self._gather((self.journal_sequence, self.journal_sha256))
        if any(value != statuses[0] for value in statuses):
            raise RuntimeError('ranks disagreed after committing a data version')
        self._switch_seconds = time.perf_counter() - switch_started
        return True

    def before_iteration(self, step: int, data_batches_consumed: int) -> bool:
        """Replay a decision made at the checkpoint boundary, before next()."""
        return self.after_train_step(step, data_batches_consumed, True, _replay_only=True)

    def state_dict(self):
        return dict(format_version=self.FORMAT_VERSION, kind='dynamic_global_buckets',
                    base_index_sha256=self.base_index_sha256, stats_sha256=self.stats_sha256,
                    normalization_sha256=self.normalization_sha256,
                    seed=self.seed, active=copy.deepcopy(self.active), planner_seed=self.planner_seed,
                    schedule_prefix=list(self.schedule_prefix), loader=self.loader.state_dict(),
                    journal_sequence=self.journal_sequence, journal_sha256=self.journal_sha256,
                    step=self.step, data_batches_consumed=self.data_batches_consumed,
                    switch_pending=self._switch_pending)

    def load_state_dict(self, state):
        if state.get('kind') != 'dynamic_global_buckets':
            # One-way upgrade of a static checkpoint; the CLI separately checks
            # every static training option before allowing this loader upgrade.
            if 'identity' not in state or 'state' not in state:
                raise ValueError('invalid static loader checkpoint')
            if self.active != self.initial:
                raise ValueError('static checkpoint upgrade requires the base snapshot')
            self.loader.load_state_dict(state)
            self.data_batches_consumed = int(state['state']['batches'])
            return
        if (state.get('format_version') != self.FORMAT_VERSION
                or state.get('base_index_sha256') != self.base_index_sha256
                or state.get('stats_sha256') != self.stats_sha256
                or state.get('normalization_sha256') != self.normalization_sha256
                or state.get('seed') != self.seed):
            raise ValueError('hotload checkpoint data or normalization contract differs')
        self._leader_call(lambda: self._verify_journal_prefix(
            int(state['journal_sequence']), state['journal_sha256']))
        candidate = None
        error = None
        try:
            with _isolated_rng():
                candidate = self._candidate(state['active'], int(state['planner_seed']))
                self._prefix_schedule(candidate, state['schedule_prefix'])
                candidate.load_state_dict(state['loader'])
        except Exception as failure:
            error = f'{type(failure).__name__}: {failure}'
        errors = self._gather(error)
        if any(errors):
            if candidate is not None:
                candidate.close()
            raise RuntimeError(f'hotload checkpoint restore failed: {errors}')
        candidate.validate_distributed()
        self.loader.close()
        self.loader = candidate
        self.active = copy.deepcopy(state['active'])
        self.planner_seed = int(state['planner_seed'])
        self.schedule_prefix = list(state['schedule_prefix'])
        self.journal_sequence = int(state['journal_sequence'])
        self.journal_sha256 = state['journal_sha256']
        self.step = int(state['step'])
        self.data_batches_consumed = int(state['data_batches_consumed'])
        self._switch_pending = bool(state['switch_pending'])
        self._last_error = None
        self._replay_cache_ready = False
        self._replay_cache = None
