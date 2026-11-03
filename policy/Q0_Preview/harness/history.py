"""Append-only archive and a planning-round window, without replaying removed reasoning."""
from __future__ import annotations

import copy
import hashlib
import json
import math

from ..runtime.protocol import MEMORY_MAX_CHARACTERS, MEMORY_TARGET_TOKENS

DEFAULT_CONTEXT_MAX_BYTES = 32 * 1024 * 1024


def json_bytes(value):
    """UTF-8 JSON size, not a model token count (same formatting as the archive)."""
    return len(json.dumps(value, ensure_ascii=False, indent=2).encode())


def memory_metrics(value):
    """A diagnostic heuristic, NOT a Claude tokenizer or an enforcement boundary."""
    ascii_count = sum(ord(c) < 128 for c in value)
    estimate = math.ceil(ascii_count / 4 + (len(value) - ascii_count) * 1.5)
    return {'characters': len(value), 'estimated_tokens': estimate,
            'token_count_method': 'heuristic: ASCII characters / 4 + non-ASCII characters * 1.5',
            'token_count_is_exact': False, 'target_tokens': MEMORY_TARGET_TOKENS,
            'estimated_over_target': estimate > MEMORY_TARGET_TOKENS}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


class PlanningHistory:
    """Only append()/extend() write messages; active and archive never share mutable blocks.

    Window changes filter the previous active view, never reconstruct it from the raw
    archive. Thus a removed signed block cannot silently return on a later request.
    IDs are zero-based indices in the complete archive, independent of window offsets.
    """
    def __init__(self, mode='first_recent', recent_rounds=5, initial=(), journal=None,
                 max_context_bytes=DEFAULT_CONTEXT_MAX_BYTES):
        if mode not in ('first_recent', 'full'):
            raise ValueError('context_history must be first_recent or full')
        if type(recent_rounds) is not int or recent_rounds != 5:
            raise ValueError('This version retains five completed planning rounds after a rebuild')
        if type(max_context_bytes) is not int or max_context_bytes <= 0:
            raise ValueError('context_max_bytes must be a positive integer')
        self.mode, self.recent_rounds = mode, recent_rounds
        self.rebuild_at_rounds, self.max_context_bytes = 10, max_context_bytes
        self.archive, self.active, self.active_ids, self.message_rounds = [], [], [], []
        self.rounds, self.selected_rounds = [], []
        self.epoch = 0
        self.removed_thinking = {}  # archive message ID -> original block indices
        self.latest_memory = None
        self.memory_carrier_round = None
        self.last_window_change = None
        self.injected_memory = None
        self.identity = None
        self.journal = journal
        self.extend(initial)

    def _record(self, kind, value):
        if self.journal is not None:
            self.journal.append(kind, value)

    def __len__(self):
        return len(self.active)

    def __getitem__(self, index):
        return self.active[index]

    @property
    def round_id(self):
        return len(self.rounds)

    def append(self, message):
        self._record('message', {'message_id': len(self.archive), 'round_id': self.round_id,
                                 'message': message})
        self.active_ids.append(len(self.archive))
        self.message_rounds.append(self.round_id)
        self.archive.append(copy.deepcopy(message))
        self.active.append(copy.deepcopy(message))

    def extend(self, messages):
        for message in messages:
            self.append(message)

    def pending_tools(self):
        pending = {}
        for message in self.active:
            if message.get('type') == 'function_call':
                pending[message['call_id']] = message.get('name')
            elif message.get('type') == 'function_call_output':
                pending.pop(message['call_id'], None)
            for block in message.get('content', []) if isinstance(message.get('content'), list) else []:
                if block.get('type') == 'tool_use':
                    pending[block['id']] = block.get('name')
                elif block.get('type') == 'tool_result':
                    if block['tool_use_id'] not in pending:
                        raise ValueError('orphan tool_result in active planning context')
                    del pending[block['tool_use_id']]
        return pending

    def begin_round(self, req, upcoming=(), prefix_bytes=0):
        if req['kind'] != 'plan':
            raise ValueError('Only a new planning observation opens a round')
        identity = (req.get('run_id'), req.get('task'))
        if self.identity is not None and self.identity != identity:
            raise ValueError('planning history cannot cross episodes or tasks')
        self.identity = identity
        if any(r['observation_id'] == req['observation_id'] for r in self.rounds):
            raise ValueError('duplicate planning observation; do not resubmit an executed decision')
        if self.pending_tools():
            raise ValueError('cannot open a planning round with pending tool results')
        n = self.round_id + 1
        completed = [r for r in self.selected_rounds if r > 1]
        before_bytes = json_bytes(self.active + list(upcoming)) + prefix_bytes
        reasons = []
        if self.mode == 'first_recent':
            if len(completed) >= self.rebuild_at_rounds:
                reasons.append('completed_round_limit')
            if before_bytes > self.max_context_bytes:
                reasons.append('input_byte_limit')
        # Round 12 keeps 1, 7..11, then appends round 12. Next normal rebuild: 17.
        retained = completed[-self.recent_rounds:] if reasons else completed
        selected = sorted({1, *retained, n})
        dropped = sorted(set(self.selected_rounds) - set(selected))
        removed = 0
        if reasons:
            active, ids = [], []
            for message_id, message in zip(self.active_ids, self.active):
                round_id = self.message_rounds[message_id]
                if round_id and round_id not in selected:
                    continue
                if round_id > 1:
                    if message.get('type') == 'reasoning':
                        self.removed_thinking[str(message_id)] = [0]
                        removed += 1
                        continue
                    if message.get('role') == 'assistant' and isinstance(message.get('content'), list):
                        indices = [i for i, block in enumerate(message['content'])
                                   if block.get('type') in ('thinking', 'redacted_thinking')]
                        if indices:
                            self.removed_thinking[str(message_id)] = indices
                            removed += len(indices)
                            message = copy.deepcopy(message)
                            message['content'] = [b for i, b in enumerate(message['content']) if i not in indices]
                            if not message['content']:
                                continue
                active.append(message)
                ids.append(message_id)
            self.active, self.active_ids = active, ids
        changed = bool(dropped or removed)
        if changed:
            self.epoch += 1
            self.last_window_change = {'epoch': self.epoch, 'at_round': n,
                'reasons': reasons, 'dropped_rounds': dropped, 'thinking_blocks_removed': removed,
                'estimated_request_bytes_before': before_bytes,
                'estimated_request_bytes_after': json_bytes(self.active + list(upcoming)) + prefix_bytes}
        self.selected_rounds = selected
        self.injected_memory = None
        if self.latest_memory and self.memory_carrier_round not in selected:
            # A previously injected copy also carries this summary. Do not repeat it
            # while that copy remains in the retained window.
            self.injected_memory = copy.deepcopy(self.latest_memory)
            self.memory_carrier_round = n
        self.rounds.append({'round_id': n, 'observation_id': req['observation_id'],
                            'request_id': req.get('request_id'), 'archive_start': len(self.archive)})
        self._record('round_started', {'round': self.rounds[-1], 'selection': self.selection(),
                                      'latest_memory': self.latest_memory})
        return changed

    def first_round_end(self):
        """Last message of the closed first round; its archive ID never shifts.

        Includes feedback/checkpoints assigned to round 1, without moving messages.
        No anchor is frozen while the first round is still being appended.
        """
        if self.round_id < 2:
            return None
        return next((i for i in range(len(self.active_ids)-1, -1, -1)
                     if self.message_rounds[self.active_ids[i]] == 1), None)

    def accept_memory(self, plan, req, raw_memory=None):
        value = plan.get('memory')
        if not isinstance(value, str) or not value.strip():
            return
        if not self.rounds or self.rounds[-1]['observation_id'] != req['observation_id']:
            raise ValueError('memory attribution mismatch')
        raw = raw_memory if isinstance(raw_memory, str) else value
        self.latest_memory = {'text': value, 'source_round': self.round_id,
            'source_observation_id': req['observation_id'], 'source_request_id': req.get('request_id'),
            'status': 'model_summary_at_accepted_submission; execution_not_confirmed',
            'truncated': len(raw) > MEMORY_MAX_CHARACTERS, **memory_metrics(value),
            'original_characters': len(raw)}
        self.memory_carrier_round = self.round_id
        self._record('memory_accepted', {'memory': self.latest_memory})

    def observation_context(self):
        value = {'planning_round': self.round_id, 'retained_planning_rounds': self.selected_rounds.copy()}
        if self.injected_memory:
            value['latest_memory'] = {k: self.injected_memory[k] for k in (
                'text', 'source_round', 'source_observation_id', 'status', 'truncated')}
        return value

    def selection(self):
        return {'mode': self.mode, 'recent_rounds': self.recent_rounds,
                'window_policy': ('full_append_only' if self.mode == 'full'
                                  else 'grow_to_10_then_keep_5_completed_plus_current'),
                'rebuild_at_completed_rounds': self.rebuild_at_rounds,
                'context_max_bytes': self.max_context_bytes,
                'first_round_end_active_index': self.first_round_end(),
                'first_round_end_message_id': (self.active_ids[self.first_round_end()]
                    if self.first_round_end() is not None else None),
                'current_round': self.round_id, 'retained_rounds': self.selected_rounds.copy(),
                'epoch': self.epoch, 'active_message_ids': self.active_ids.copy(),
                'archive_message_count': len(self.archive), 'active_message_count': len(self.active),
                'removed_thinking_blocks': copy.deepcopy(self.removed_thinking),
                'last_window_change': copy.deepcopy(self.last_window_change),
                'latest_memory_source_round': (self.latest_memory or {}).get('source_round'),
                'memory_carrier_round': self.memory_carrier_round,
                'memory_injected_in_current_observation': self.injected_memory is not None,
                'injected_memory_source_round': (self.injected_memory or {}).get('source_round')}

    def archive_state(self):
        return {'messages': self.archive, 'history_policy': 'complete_append_only_with_rolling_memory',
                'rounds': self.rounds, 'message_rounds': self.message_rounds,
                'active_context': self.selection(), 'latest_memory': self.latest_memory}


def reconstruct_active(archive, selection):
    """Diagnostic replay of one saved selection, never a fresh history selection."""
    messages = []
    for message_id in selection['active_message_ids']:
        message = copy.deepcopy(archive[message_id])
        removed = selection['removed_thinking_blocks'].get(str(message_id), [])
        if removed:
            message['content'] = [b for i, b in enumerate(message['content']) if i not in removed]
        messages.append(message)
    return messages
