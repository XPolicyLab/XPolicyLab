"""Knowledge storage: experiences, skills with evidence, task briefs,
changelog, generated index and statistics, and snapshots.

Layout of a campaign knowledge directory:

  task_briefs/<task>.md        task brief
  skills/<name>.md             one embodied skill: frontmatter with evidence, then the body
  tools/<task>/<name>.py       one in-task tool: a Python module of functions that compute from what they are
                               passed; tools/<task>/<name>.json holds its doc, the code's sha256, status and
                               evidence
  experience/<run_id>.jsonl    experiences of one run
  handoffs/<run_id>.json       development handoffs
  video/<task>/                video analysis
  versions/<name>/<sha>.md     earlier versions of revised skills
  versions/tools/<task>/<name>/<sha>.py|.json  earlier versions of revised or retired tools
  changelog.jsonl              every change: file, hashes before/after, reason, source run
  INDEX.md, stats.json         generated from skills and experiences
  run_kinds.json               {run_id: kind} of the development episodes (fresh, history, retry, replay,
                               unrecorded), written by the manager; the index counts evidence by kind
  snapshots/<id>/              frozen copies

Files are written only through this module and are left read-only; a snapshot is
refused when a file no longer matches its last changelog entry.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import uuid

from v2.knowledge import task_tools
from v2.knowledge.task_tools import ToolError, is_tool_name
from v2.tasks import TASKS

RESULTS = ('success', 'partial', 'fail', 'unclear')
VERDICTS = ('effective', 'failed')
SKILL_TYPES = ('manipulation', 'perception', 'recovery', 'strategy')
SKILL_NAME = re.compile(r'[a-z0-9]+(?:\.[a-z0-9-]+)+')
RUN_ID = r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}'
REF = re.compile(rf'(?:(?P<run>{RUN_ID}):)?(?P<ref>frame_\d+|[0-9a-f]{{8}}|[0-9a-f]{{32}})')
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
BODY_LIMIT = 8000            # about 2k tokens
WHEN_LIMIT = 300
BRIEF_SECTIONS = ('Task instruction', 'Authorized priors', 'From the demo video', 'Best-known approach',
                  'Key concepts and pitfalls', 'Open questions')
DEVELOPER_SECTIONS = ('Task instruction', 'Authorized priors')
BRIEF_LIMIT = 20000           # about 5k tokens
VIDEO_SECTION_LIMIT = 1200    # about 300 tokens
INDEX_BUDGET = 12000          # about 3k tokens per task; more triggers curation


def video_time(t):
    """A frame's video time: seconds as a float, or a range 'a-b' in seconds as text."""
    if isinstance(t, numbers.Real) and not isinstance(t, bool) and math.isfinite(float(t)) and float(t) >= 0:
        return round(float(t), 3)
    if isinstance(t, str) and re.fullmatch(r'\s*\d+(\.\d+)?\s*-\s*\d+(\.\d+)?\s*', t):
        return t.strip()
    raise KnowledgeError("t is the frame's video time in seconds, or a range such as '9.40-10.48'")


class KnowledgeError(ValueError):
    """A knowledge write or snapshot that was refused; the message is meant for the model."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def _text(name, value, limit, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise KnowledgeError(f'{name} must be non-empty text of at most {limit} characters')
    return value.strip()


def normalize_newlines(text):
    """Files are read with universal newlines, so a bare '\\r' would become a line break later."""
    return text.replace('\r\n', '\n').replace('\r', '\n') if isinstance(text, str) else text


def _write_readonly(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.chmod(0o444)
    os.replace(temp, path)


# ------------------------------------------------------------- experiences

class ExperienceLog:
    """Append-only JSONL experience log for one run.

    An experience is a descriptive record written in real time by the actor: the
    situation, what was done, what was observed and the actor's own judgment of
    that attempt. The harness fills in where and when it happened.
    """

    def __init__(self, path, context):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = {key: context.get(key) for key in ('run_id', 'task', 'seed', 'layout_id')}
        # Opened once, so later writes need no new file access.
        self.stream = self.path.open('a', encoding='utf-8')

    def append(self, description, result, insight=None, skills_used=(), frames=(),
               step=None, frame_id=None, command_ids=(), origin='actor'):
        if result not in RESULTS:
            raise ValueError('result must be one of ' + ', '.join(RESULTS))
        skills = list(skills_used or ())
        if len(skills) > 20 or any(not isinstance(s, str) or not (SKILL_NAME.fullmatch(s) or is_tool_name(s))
                                   for s in skills):
            raise ValueError('skills_used must be up to 20 skill names like "grasp.upright-cup-side"')
        refs = list(frames or ())
        if len(refs) > 20 or any(not isinstance(f, str) or not REF.fullmatch(f) for f in refs):
            raise ValueError('frames must be up to 20 frame or command IDs like "frame_000123" or "c1a2b3c4"')
        record = {'id': uuid.uuid4().hex[:12], **self.context,
                  'time': now(), 'step': step, 'frame_id': frame_id, 'command_ids': list(command_ids),
                  'description': _text('description', description, 4000),
                  'result': result, 'insight': _text('insight', insight, 2000, required=False),
                  'skills_used': skills, 'frames': refs, 'origin': origin}
        self.stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.stream.flush()
        return {'id': record['id'], 'result': result}

    def close(self):
        self.stream.close()


def jsonl_lines(text):
    """Lines of a JSONL text. Only '\\n' separates records: str.splitlines() would also split at
    U+2028, U+2029 and U+0085, which json.dumps(ensure_ascii=False) leaves inside a record."""
    return [line for line in text.split('\n') if line.strip()]


def read_experiences(path):
    """Records of a JSONL file (experiences, changelog), skipping a torn final line."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in jsonl_lines(path.read_text(encoding='utf-8')):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a torn final line from a crash
    return rows


class Results(list):
    """Experience search results; printing shows one compact line per experience."""

    def __str__(self):
        if not self:
            return '(no matching experiences)'
        return '\n'.join(experience_line(row) for row in self)

    __repr__ = __str__


def experience_line(row):
    def short(text, limit):
        text = ' '.join(str(text or '').split())
        return text if len(text) <= limit else text[:limit - 1] + '…'
    parts = [f"{row.get('id')} {row.get('run_id')} L{row.get('layout_id')} step {row.get('step')}",
             f"[{row.get('result')}]", short(row.get('description'), 240)]
    if row.get('insight'):
        parts.append('| insight: ' + short(row['insight'], 160))
    if row.get('skills_used'):
        parts.append('| skills: ' + ','.join(row['skills_used']))
    return ' '.join(parts)


# ------------------------------------------------------------------ skills

def parse_frontmatter(text):
    """Split `---` frontmatter from the body; return (fields, evidence_lines, body)."""
    text = text.lstrip('﻿').strip('\n')
    if not text.startswith('---\n'):
        raise KnowledgeError('a skill starts with a frontmatter block: ---, when:, type:, tasks:, ---')
    end = text.find('\n---', 4)
    if end < 0:
        raise KnowledgeError('the frontmatter block is not closed with ---')
    header, body = text[4:end], text[end + 4:].lstrip('\n')
    fields, evidence, current = {}, [], None
    for line in header.split('\n'):
        if not line.strip():
            continue
        if line.startswith('  - ') and current == 'evidence':
            evidence.append(line[4:])
            continue
        if ':' not in line or line.startswith(' '):
            raise KnowledgeError(f'cannot read frontmatter line: {line.strip()[:80]}')
        key, value = line.split(':', 1)
        current = key.strip()
        fields[current] = value.strip()
    return fields, evidence, body


def parse_tasks(value):
    value = value.strip()
    if not (value.startswith('[') and value.endswith(']')):
        raise KnowledgeError('tasks must be a list like [pour_balls_into_vase] or [] for a general skill')
    tasks = [item.strip().strip('\'"') for item in value[1:-1].split(',') if item.strip()]
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise KnowledgeError('unknown tasks: ' + ', '.join(unknown))
    return sorted(set(tasks))


def render_skill(meta, body):
    lines = ['---', f"name: {meta['name']}", f"when: {meta['when']}", f"type: {meta['type']}",
             'tasks: [' + ', '.join(meta['tasks']) + ']', f"status: {meta['status']}", 'evidence:']
    lines += ['  - ' + json.dumps(entry, ensure_ascii=True) for entry in meta['evidence']]
    lines += ['---', body.strip(), '']
    return '\n'.join(lines)


def load_skill(path):
    fields, evidence, body = parse_frontmatter(Path(path).read_text(encoding='utf-8'))
    meta = {'name': fields.get('name'), 'when': fields.get('when', ''), 'type': fields.get('type'),
            'tasks': parse_tasks(fields.get('tasks', '[]')), 'status': fields.get('status', 'active'),
            'evidence': [json.loads(line) for line in evidence]}
    return meta, body


def render_tool(meta):
    """A tool's metadata file: name, task, doc, the names it defines, the code's sha256, status, evidence."""
    return json.dumps(meta, indent=1, ensure_ascii=False) + '\n'


def tool_meta_problem(meta, name, task):
    """Why a tool's metadata is malformed, or None."""
    if not isinstance(meta, dict) or meta.get('name') != name or meta.get('task') != task:
        return 'name or task does not match the file'
    if meta.get('status') not in ('active', 'retired'):
        return 'invalid status'
    if not isinstance(meta.get('evidence'), list) or any(e.get('verdict') not in VERDICTS for e in meta['evidence']):
        return 'invalid evidence'
    if not isinstance(meta.get('doc'), str) or not isinstance(meta.get('sha256'), str):
        return 'missing doc or sha256'
    return None


def check_body(body):
    body = body.strip()
    if not body:
        raise KnowledgeError('the skill body is empty')
    if len(body) > BODY_LIMIT:
        raise KnowledgeError(f'the skill body has {len(body)} characters; the limit is {BODY_LIMIT}')
    if '```' in body:
        raise KnowledgeError('skills are text only: no code blocks')
    if not re.search(r'^## How\s*$', body, re.M):
        raise KnowledgeError('the body needs a "## How" section (then "## How to check", '
                             '"## Failure signals and recovery", "## Limits")')
    return body


def evidence_summary(evidence):
    """`judged effective on layouts 0,1; failed on 2` from evidence entries."""
    def layouts(verdict):
        values = sorted({(entry.get('task'), entry.get('layout')) for entry in evidence
                         if entry.get('verdict') == verdict}, key=lambda value: (str(value[0]), value[1] or 0))
        tasks = {task for task, _ in values}
        if len(tasks) > 1:
            return ', '.join(f'{task} {layout}' for task, layout in values)
        return ','.join(str(layout) for _, layout in values)
    effective, failed = layouts('effective'), layouts('failed')
    parts = []
    if effective:
        parts.append('judged effective on layouts ' + effective)
    if failed:
        parts.append(('failed on ' if effective else 'failed on layouts ') + failed)
    return '; '.join(parts) or 'no evidence yet'


# How the skill index labels the kinds of development episode a piece of evidence comes from. 'current' is the
# episode that is running, not classified yet.
EVIDENCE_KINDS = (('fresh', 'fresh'), ('history', 'history'), ('retry', 'retry'), ('replay', 'replay'),
                  ('unrecorded', 'unrecorded'), ('current', 'this episode'), ('other', 'other'))


def evidence_sources(evidence, kinds, current=None):
    """`evidence: fresh 2 effective 1 failed; replay 3 effective`: valid evidence counted by the kind of episode it
    cites (kinds: {run_id: kind}; current: the running episode's run ID), or '' when no cited episode is
    classified."""
    counts = {}
    for entry in evidence:
        run = entry.get('run')
        kind = 'current' if current is not None and run == current else kinds.get(run, 'other')
        kind = kind if kind in dict(EVIDENCE_KINDS) else 'other'
        verdicts = counts.setdefault(kind, {'effective': 0, 'failed': 0})
        if entry.get('verdict') in verdicts:
            verdicts[entry['verdict']] += 1
    if set(counts) <= {'other'}:
        return ''  # nothing classified (e.g. knowledge from before run_kinds.json): the index says as before
    parts = []
    for kind, label in EVIDENCE_KINDS:
        if kind in counts:
            numbers = ' '.join(f'{n} {verdict}' for verdict, n in counts[kind].items() if n)
            parts.append(f'{label} {numbers}')
    return 'evidence: ' + '; '.join(parts) if parts else ''


# -------------------------------------------------------------- task brief

def render_brief(task, sections):
    lines = [f'# {task}']
    for name in BRIEF_SECTIONS:
        lines += ['', f'## {name}', (sections.get(name) or '').strip() or '(none yet)']
    return '\n'.join(lines).strip() + '\n'


def parse_brief(text):
    sections, current = {}, None
    for line in text.split('\n'):
        match = re.fullmatch(r'## (.+?)\s*', line)
        if match and match.group(1) in BRIEF_SECTIONS:
            current = match.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: '\n'.join(lines).strip() for name, lines in sections.items()
            if '\n'.join(lines).strip() != '(none yet)'}


# ------------------------------------------------------------------ store

class Knowledge:
    """A knowledge directory, live (development) or a frozen snapshot (evaluation).

    `run` describes the session that uses it: run_id, task, seed, layout_id and
    `recording` (the current run's recording directory) for resolving evidence.
    `recordings` maps other run IDs to their recording directories.
    """

    def __init__(self, root, run=None, recordings=None, writable=False, allow_write=None, layouts=None):
        self.root = Path(root)
        self.run = dict(run or {})
        self.recordings = {key: Path(value) for key, value in (recordings or {}).items()}
        self.layouts = dict(layouts or {})
        self.run_kinds = {str(k): v for k, v in (self._load_json('run_kinds.json') or {}).items()}
        self.writable = writable
        # Context manager factory that opens a write window in the worker's guard.
        self.allow_write = allow_write
        self.invalid = self._load_invalid()
        self.excluded_experiences = self._load_excluded()

    def _load_json(self, *names):
        for name in names:
            path = self.root / name
            if path.is_file():
                try:
                    return json.loads(path.read_text())
                except ValueError:
                    return {}
        return {}

    def _load_invalid(self):
        """Evidence found invalid at the last snapshot (a snapshot's manifest, or the live directory's list)."""
        issues = self._load_json('manifest.json', 'evidence_issues.json').get('invalid_evidence') or []
        return {(issue['run'], issue['ref']): issue['issue'] for issue in issues}

    def _load_excluded(self):
        """IDs of experiences refused at the last snapshot; they are not searchable."""
        rows = self._load_json('manifest.json', 'experience_admission.json').get('excluded_experiences') or []
        return {row['id'] for row in rows}

    def valid_evidence(self, evidence):
        return [entry for entry in evidence if (entry.get('run'), entry.get('ref')) not in self.invalid]

    # paths
    @property
    def skills_dir(self):
        return self.root / 'skills'

    def skill_path(self, name):
        if not isinstance(name, str) or not SKILL_NAME.fullmatch(name) or len(name) > 80:
            raise KnowledgeError(f'invalid skill name {name!r}: use action.object-or-situation, lowercase, '
                                 'e.g. grasp.upright-cup-side')
        return self.skills_dir / f'{name}.md'

    def brief_path(self, task):
        if task not in TASKS:
            raise KnowledgeError(f'unknown task {task!r}')
        return self.root / 'task_briefs' / f'{task}.md'

    # ------------------------------------------------------------- reading

    def skill_names(self):
        return sorted(path.stem for path in self.skills_dir.glob('*.md')) if self.skills_dir.is_dir() else []

    def read_skill(self, name):
        path = self.skill_path(name)
        if not path.is_file():
            raise KnowledgeError(f'no skill named {name}; skills() lists them')
        return load_skill(path)

    def skills_for(self, task, include_retired=False):
        rows = []
        for name in self.skill_names():
            meta, _ = load_skill(self.skill_path(name))
            if (include_retired or meta['status'] == 'active') and (not meta['tasks'] or task in meta['tasks']):
                rows.append(meta)
        return rows

    def index_lines(self, task):
        lines = []
        for meta in self.skills_for(task):
            scope = '' if meta['tasks'] else ' (general)'
            valid = self.valid_evidence(meta['evidence'])
            sources = evidence_sources(valid, self.run_kinds, self.run.get('run_id'))
            lines.append(f"- {meta['name']} [{meta['type']}]{scope} {meta['when']} — {evidence_summary(valid)}"
                         + (f' ({sources})' if sources else ''))
        return lines

    def brief(self, task):
        path = self.brief_path(task)
        return path.read_text(encoding='utf-8') if path.is_file() else None

    def show_skill(self, name):
        """The text a model sees for skill(name): header, evidence digest and body."""
        meta, body = self.read_skill(name)
        scope = ', '.join(meta['tasks']) or 'general'
        lines = [f"Skill {meta['name']} [{meta['type']}; {scope}; {meta['status']}]", f"When: {meta['when']}",
                 'Evidence: ' + evidence_summary(self.valid_evidence(meta['evidence']))]
        for entry in meta['evidence'][-8:]:
            note = f" — {entry['note']}" if entry.get('note') else ''
            issue = self.invalid.get((entry['run'], entry['ref']))
            flag = f' (not counted: {issue})' if issue else ''
            lines.append(f"  {entry['verdict']}: {entry.get('task')} layout {entry.get('layout')}, "
                         f"{entry['run']}:{entry['ref']}{note}{flag}")
        if len(meta['evidence']) > 8:
            lines.append(f"  ({len(meta['evidence']) - 8} earlier entries not shown)")
        return '\n'.join(lines) + '\n\n' + body.strip()

    # ------------------------------------------------------- tools

    def tool_paths(self, name, task=None):
        """(code, metadata) paths of a tool of `task`, this session's task by default: tools belong to one task."""
        try:
            task_tools.check_name(name)
        except ToolError as exc:
            raise KnowledgeError(str(exc)) from None
        task = task or self.run.get('task')
        if task not in TASKS:
            raise KnowledgeError(f'a tool belongs to a task; unknown task {task!r}')
        folder = self.root / 'tools' / task
        return folder / f'{name}.py', folder / f'{name}.json'

    def tool_names(self, task):
        folder = self.root / 'tools' / str(task)
        return sorted(path.stem for path in folder.glob('*.json')) if folder.is_dir() else []

    def read_tool(self, name, task=None):
        """(metadata, code) of a tool; the code must match the sha256 in its metadata."""
        code_path, meta_path = self.tool_paths(name, task)
        if not meta_path.is_file():
            raise KnowledgeError(f'no tool named {name} for {task or self.run.get("task")}; the tools are listed '
                                 'after the skills')
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except ValueError:
            raise KnowledgeError(f'tool {name}: its metadata cannot be read') from None
        data = code_path.read_bytes() if code_path.is_file() else b''
        if sha256_bytes(data) != meta.get('sha256'):
            raise KnowledgeError(f'tool {name}: the code does not match its recorded hash')
        return meta, data.decode('utf-8')

    def tools_for(self, task, include_retired=False):
        rows = []
        for name in self.tool_names(task):
            try:
                meta = json.loads((self.root / 'tools' / task / f'{name}.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue  # reported by problems()
            if include_retired or meta.get('status') == 'active':
                rows.append(meta)
        return rows

    def tool_lines(self, task):
        """One line per active tool of the task: name, the names it defines, the doc's first line, the evidence."""
        lines = []
        for meta in self.tools_for(task):
            valid = self.valid_evidence(meta.get('evidence') or [])
            sources = evidence_sources(valid, self.run_kinds, self.run.get('run_id'))
            first = ' '.join(str(meta.get('doc') or '').strip().split('\n', 1)[0].split())
            first = first if len(first) <= 200 else first[:199] + '…'
            names = ', '.join(meta.get('functions') or [])
            lines.append(f"- {meta['name']} [{names}] {first} — {evidence_summary(valid)}"
                         + (f' ({sources})' if sources else ''))
        return lines

    def show_tool(self, name, task=None):
        """The text shown when a tool is imported: header, evidence digest and doc."""
        meta, _ = self.read_tool(name, task)
        lines = [f"Tool {name} [{meta['task']}; {meta['status']}; defines {', '.join(meta.get('functions') or [])}]",
                 'Evidence: ' + evidence_summary(self.valid_evidence(meta['evidence']))]
        for entry in meta['evidence'][-8:]:
            note = f" — {entry['note']}" if entry.get('note') else ''
            issue = self.invalid.get((entry['run'], entry['ref']))
            flag = f' (not counted: {issue})' if issue else ''
            lines.append(f"  {entry['verdict']}: {entry.get('task')} layout {entry.get('layout')}, "
                         f"{entry['run']}:{entry['ref']}{note}{flag}")
        if len(meta['evidence']) > 8:
            lines.append(f"  ({len(meta['evidence']) - 8} earlier entries not shown)")
        return '\n'.join(lines) + '\n\n' + meta['doc'].strip()

    def experience_files(self):
        folder = self.root / 'experience'
        return sorted(folder.glob('*.jsonl')) if folder.is_dir() else []

    def search_experience(self, query=None, task=None, layout=None, run=None, result=None, limit=20,
                          extra_files=()):
        if run == 'current':
            run = self.run.get('run_id')
        task = task if task is not None else self.run.get('task')
        words = [word.lower() for word in str(query or '').split()]
        rows = []
        for path in list(self.experience_files()) + [Path(p) for p in extra_files]:
            for row in read_experiences(path):
                if row.get('id') in self.excluded_experiences and row.get('run_id') != self.run.get('run_id'):
                    continue
                if task not in (None, 'all') and row.get('task') != task:
                    continue
                if layout is not None and row.get('layout_id') != layout:
                    continue
                if run is not None and row.get('run_id') != run:
                    continue
                if result is not None and row.get('result') != result:
                    continue
                if words:
                    haystack = ' '.join(str(row.get(k) or '') for k in ('description', 'insight')).lower()
                    haystack += ' ' + ' '.join(row.get('skills_used') or [])
                    if not all(word in haystack for word in words):
                        continue
                rows.append(row)
        unique = {row.get('id'): row for row in rows}
        rows = sorted(unique.values(), key=lambda row: (row.get('time') or '', row.get('id') or ''))
        return Results(rows[-int(limit):] if limit else rows)

    # ------------------------------------------------------------- writing

    def _require_writable(self):
        if not self.writable:
            raise KnowledgeError('knowledge is read-only in this session')

    def _window(self):
        return self.allow_write() if self.allow_write else _NoWindow()

    def _log(self, op, path, before, after, reason, extra=None):
        record = {'time': now(), 'op': op, 'file': str(Path(path).relative_to(self.root)),
                  'before_sha256': before, 'after_sha256': after, 'reason': reason,
                  'run_id': self.run.get('run_id'), 'session': self.run.get('session', 'actor')}
        if extra:
            record.update(extra)
        with (self.root / 'changelog.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _save_version(self, name, path):
        if path.is_file():
            data = path.read_bytes()
            target = self.root / 'versions' / name / f'{sha256_bytes(data)[:16]}.md'
            if not target.exists():
                _write_readonly(target, data.decode('utf-8'))

    def write_skill(self, name, text, reason):
        """Create or revise a skill from model-written text: frontmatter (when, type, tasks) and body."""
        self._require_writable()
        path = self.skill_path(name)
        reason = _text('reason', reason, 1000)
        text = normalize_newlines(text)
        if not isinstance(text, str) or len(text) > BODY_LIMIT + 2000:
            raise KnowledgeError(f'text must be a skill document of at most about {BODY_LIMIT} characters of body')
        fields, evidence, body = parse_frontmatter(text)
        if evidence or 'evidence' in fields or 'status' in fields:
            raise KnowledgeError('evidence is added with add_evidence and status changes with retire_skill; '
                                 'write only name/when/type/tasks in the frontmatter')
        unknown = set(fields) - {'name', 'when', 'type', 'tasks'}
        if unknown:
            raise KnowledgeError('unknown frontmatter fields: ' + ', '.join(sorted(unknown)))
        if fields.get('name', name) != name:
            raise KnowledgeError(f"the frontmatter name {fields['name']!r} differs from {name!r}")
        when = _text('when', fields.get('when'), WHEN_LIMIT)
        if fields.get('type') not in SKILL_TYPES:
            raise KnowledgeError('type must be one of ' + ', '.join(SKILL_TYPES))
        if 'tasks' not in fields:
            raise KnowledgeError('tasks is required: [<task>] for a task skill, [] for a general skill')
        tasks = parse_tasks(fields['tasks'])
        body = check_body(body)
        existing, before = [], None
        if path.is_file():
            before = sha256_file(path)
            old, _ = load_skill(path)
            existing = old['evidence']
        meta = {'name': name, 'when': when, 'type': fields['type'], 'tasks': tasks, 'status': 'active',
                'evidence': existing}
        rendered = render_skill(meta, body)
        with self._window():
            self._save_version(name, path)
            _write_readonly(path, rendered)
            self._log('revise_skill' if before else 'create_skill', path, before, sha256_file(path), reason)
        return f"{'Revised' if before else 'Created'} skill {name}."

    def retire_skill(self, name, reason):
        self._require_writable()
        reason = _text('reason', reason, 1000)
        path = self.skill_path(name)
        meta, body = self.read_skill(name)
        if meta['status'] == 'retired':
            return f'Skill {name} is already retired.'
        before = sha256_file(path)
        meta['status'] = 'retired'
        with self._window():
            self._save_version(name, path)
            _write_readonly(path, render_skill(meta, body))
            self._log('retire_skill', path, before, sha256_file(path), reason)
        return f'Retired skill {name}.'

    def _save_tool_version(self, task, name, *paths):
        folder = self.root / 'versions' / 'tools' / task / name
        for path in paths:
            if path.is_file():
                data = path.read_bytes()
                target = folder / f'{sha256_bytes(data)[:16]}{path.suffix}'
                if not target.exists():
                    _write_readonly(target, data.decode('utf-8'))

    def write_tool(self, name, code, doc, reason):
        """Create or revise a tool of this session's task: Python code that passes task_tools.check_code, and a doc
        that says what each function takes and returns and how to check its output. Evidence is kept across
        revisions, as for skills; each piece names the code version it was added to."""
        self._require_writable()
        task = self.run.get('task')
        code_path, meta_path = self.tool_paths(name, task)
        reason = _text('reason', reason, 1000)
        code, doc = normalize_newlines(code), normalize_newlines(doc)
        try:
            functions = task_tools.check_code(code)
            doc = task_tools.check_doc(doc)
        except ToolError as exc:
            raise KnowledgeError(str(exc)) from None
        code = code if code.endswith('\n') else code + '\n'
        evidence, before, code_before = [], None, None
        if meta_path.is_file():
            old, _ = self.read_tool(name, task)
            evidence = old['evidence']
            before = sha256_file(meta_path)
            code_before = old['sha256']
        meta = {'name': name, 'task': task, 'doc': doc, 'functions': functions,
                'sha256': sha256_bytes(code.encode('utf-8')), 'status': 'active', 'evidence': evidence}
        with self._window():
            self._save_tool_version(task, name, code_path, meta_path)
            _write_readonly(code_path, code)
            _write_readonly(meta_path, render_tool(meta))
            self._log('revise_tool' if before else 'create_tool', meta_path, before, sha256_file(meta_path), reason,
                      {'tool': name, 'code_file': str(code_path.relative_to(self.root)),
                       'code_before_sha256': code_before, 'code_after_sha256': meta['sha256']})
        return f"{'Revised' if before else 'Created'} tool {name} (defines {', '.join(functions)})."

    def retire_tool(self, name, reason):
        self._require_writable()
        reason = _text('reason', reason, 1000)
        code_path, meta_path = self.tool_paths(name)
        meta, _ = self.read_tool(name)
        if meta['status'] == 'retired':
            return f'Tool {name} is already retired.'
        before = sha256_file(meta_path)
        meta['status'] = 'retired'
        with self._window():
            self._save_tool_version(meta['task'], name, meta_path)
            _write_readonly(meta_path, render_tool(meta))
            self._log('retire_tool', meta_path, before, sha256_file(meta_path), reason, {'tool': name})
        return f'Retired tool {name}.'

    def resolve_ref(self, ref):
        """(run_id, kind, ref, sha256) for a frame or command reference, current run by default."""
        match = REF.fullmatch(str(ref or '').strip())
        if not match:
            raise KnowledgeError('ref must be a frame ID (frame_000123) or command ID (c1a2b3c4), optionally '
                                 'prefixed with a run ID and a colon')
        run_id = match.group('run') or self.run.get('run_id')
        value = match.group('ref')
        if run_id == self.run.get('run_id') and self.run.get('recording'):
            recording = Path(self.run['recording'])
        elif run_id in self.recordings:
            recording = self.recordings[run_id]
        else:
            raise KnowledgeError(f'run {run_id} is not available here; cite this episode or a run under history/')
        if value.startswith('frame_'):
            folder = recording / 'observations' / value
            files = [folder / f'{camera}.png' for camera in CAMERAS]
            if not all(path.is_file() for path in files):
                raise KnowledgeError(f'frame {value} is not recorded in run {run_id}')
            digest = sha256_bytes(''.join(sha256_file(path) for path in files).encode())
            return run_id, 'frame', value, digest
        commands = recording / 'commands.jsonl'
        lines = jsonl_lines(commands.read_text(encoding='utf-8')) if commands.is_file() else []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if value in (row.get('request', {}).get('id'), row.get('result', {}).get('command_id')):
                return run_id, 'command', value, sha256_bytes(line.encode())
        raise KnowledgeError(f'command {value} is not recorded in run {run_id}')

    def add_evidence(self, name, verdict, ref, note='', layout=None, task=None):
        self._require_writable()
        if verdict not in VERDICTS:
            raise KnowledgeError('verdict must be effective or failed')
        note = _text('note', note, 300, required=False) if note else None
        tool = is_tool_name(name)
        if tool:
            path = self.tool_paths(name)[1]
            if not path.is_file():
                raise KnowledgeError(f'no tool named {name} (skill names contain a dot, e.g. grasp.upright-cup-side)')
            meta, _ = self.read_tool(name)
        else:
            if not isinstance(name, str) or not SKILL_NAME.fullmatch(name):
                raise KnowledgeError(f'invalid name {name!r}: a skill is named action.object-or-situation (e.g. '
                                     'grasp.upright-cup-side), a tool with a lowercase Python name (e.g. camera_model)')
            path = self.skill_path(name)
            meta, body = self.read_skill(name)
        run_id, kind, value, digest = self.resolve_ref(ref)
        if run_id == self.run.get('run_id'):
            layout, task = self.run.get('layout_id'), self.run.get('task')
        elif run_id in self.layouts:
            layout, task = self.layouts[run_id], task or self.run.get('task')
        if layout is None or task is None:
            raise KnowledgeError(f'the layout and task of run {run_id} are unknown here')
        entry = {'run': run_id, 'task': task, 'layout': layout, 'ref': value, 'kind': kind, 'sha256': digest,
                 'verdict': verdict}
        if note:
            entry['note'] = note
        if tool:
            entry['tool_version'] = meta['sha256'][:16]  # the code this evidence was added to
        if any(e['run'] == run_id and e['ref'] == value and e['verdict'] == verdict for e in meta['evidence']):
            return f'This evidence is already recorded for {name}.'
        before = sha256_file(path)
        meta['evidence'].append(entry)
        with self._window():
            _write_readonly(path, render_tool(meta) if tool else render_skill(meta, body))
            self._log('add_evidence', path, before, sha256_file(path), f'{verdict} {run_id}:{value}',
                      {'tool' if tool else 'skill': name})
        return f'Added {verdict} evidence to {name} ({task} layout {layout}, {run_id}:{value}).'

    def init_brief(self, task, instruction, prior=None, reason='developer initialization'):
        """Developer-written sections; models cannot change them."""
        path = self.brief_path(task)
        sections = parse_brief(path.read_text(encoding='utf-8')) if path.is_file() else {}
        sections['Task instruction'] = instruction
        sections['Authorized priors'] = prior or ''
        before = sha256_file(path) if path.is_file() else None
        text = render_brief(task, sections)
        if before == sha256_bytes(text.encode()):
            return 'unchanged'
        with self._window():
            _write_readonly(path, text)
            self._log('init_brief', path, before, sha256_file(path), reason)
        return 'initialized'

    def update_brief(self, section, text, reason, task=None, video=False):
        self._require_writable()
        task = task or self.run.get('task')
        reason = _text('reason', reason, 1000)
        if section not in BRIEF_SECTIONS:
            raise KnowledgeError('section must be one of ' + ', '.join(BRIEF_SECTIONS))
        if section in DEVELOPER_SECTIONS:
            raise KnowledgeError(f'"{section}" is written by the developer')
        if section == 'From the demo video' and not video:
            raise KnowledgeError('"From the demo video" is written by the video analysis session')
        text = normalize_newlines(text)
        if not isinstance(text, str) or '```' in text:
            raise KnowledgeError('text must be plain text without code blocks')
        if re.search(r'^## ', text, re.M):
            raise KnowledgeError('text is the body of one section: it must not contain "## " headings '
                                 '(use "### " for sub-headings)')
        path = self.brief_path(task)
        if not path.is_file():
            raise KnowledgeError(f'the task brief for {task} has not been initialized')
        sections = parse_brief(path.read_text(encoding='utf-8'))
        sections[section] = text.strip()
        if section == 'From the demo video' and len(text.strip()) > VIDEO_SECTION_LIMIT:
            raise KnowledgeError(f'the video section is limited to {VIDEO_SECTION_LIMIT} characters')
        rendered = render_brief(task, sections)
        if len(rendered) > BRIEF_LIMIT:
            raise KnowledgeError(f'the task brief would have {len(rendered)} characters; the limit is '
                                 f'{BRIEF_LIMIT}. Shorten this or another section.')
        before = sha256_file(path)
        with self._window():
            _write_readonly(path, rendered)
            self._log('update_brief', path, before, sha256_file(path), reason, {'section': section})
        return f'Updated "{section}" in the {task} brief.'

    def save_video_analysis(self, text, reason, task=None):
        """Write video/<task>/analysis.md (video analysis sessions only)."""
        self._require_writable()
        task = task or self.run.get('task')
        text = _text('analysis', normalize_newlines(text), 60000)
        path = self.root / 'video' / task / 'analysis.md'
        before = sha256_file(path) if path.is_file() else None
        with self._window():
            _write_readonly(path, text + '\n')
            self._log('video_analysis', path, before, sha256_file(path), _text('reason', reason, 1000))
        return f'Saved the video analysis ({len(text)} characters).'

    def save_video_frame(self, name, png_bytes, t, task=None, reason='frame cited by the analysis'):
        """Save a cited frame or crop; t is its video time in seconds (for a crop, of its source frame),
        or a range like '9.40-10.48' for a contact sheet."""
        self._require_writable()
        task = task or self.run.get('task')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,80}', str(name)):
            raise KnowledgeError('frame names use letters, digits, dots, dashes and underscores')
        t = video_time(t)
        name = name if name.endswith('.png') else name + '.png'
        path = self.root / 'video' / task / 'frames' / name
        with self._window():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f'.{name}.{uuid.uuid4().hex[:8]}.tmp')
            temp.write_bytes(png_bytes)
            temp.chmod(0o444)
            os.replace(temp, path)
            digest = sha256_file(path)
            with (self.root / 'video' / task / 'frames.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'name': name, 't': t, 'sha256': digest, 'time': now()}) + '\n')
            self._log('video_frame', path, None, digest, reason, {'t': t})
        return f'video/{task}/frames/{name}'

    def import_video(self, source_root, task, label):
        """Copy another knowledge root's video analysis of `task`: its analysis, cited frames (with their
        video times) and the brief's video section. Each source file is first checked against the source's
        own changelog or frame record, nothing is written unless all match, and every write names `label`."""
        self._require_writable()
        source_root = Path(source_root)
        source = Knowledge(source_root)
        own = self.root / 'video' / task
        if (own / 'analysis.md').exists() or (own / 'frames.jsonl').exists():
            raise KnowledgeError(f'this knowledge already has a video analysis of {task}')
        last = {record['file']: record.get('after_sha256') for record in source.changelog()}
        analysis, brief = source_root / 'video' / task / 'analysis.md', source.brief_path(task)
        for path in (analysis, brief):
            relative = str(path.relative_to(source_root))
            if not path.is_file() or last.get(relative) != sha256_file(path):
                raise KnowledgeError(f'the source {relative} is missing or does not match its changelog')
        section = parse_brief(brief.read_text(encoding='utf-8')).get('From the demo video', '').strip()
        if not section or section == '(none yet)':
            raise KnowledgeError('the source brief has no video section')
        frames = {row['name']: row for row in read_experiences(source_root / 'video' / task / 'frames.jsonl')}
        images = {}
        for name, row in frames.items():
            data = (source_root / 'video' / task / 'frames' / name).read_bytes()
            if sha256_bytes(data) != row.get('sha256'):
                raise KnowledgeError(f'the source frame {name} does not match its record')
            images[name] = data
        reason = f'imported from {label}'
        self.save_video_analysis(analysis.read_text(encoding='utf-8'), reason, task=task)
        for name in sorted(frames):
            self.save_video_frame(name, images[name], frames[name]['t'], task=task, reason=reason)
        self.update_brief('From the demo video', section, reason, task=task, video=True)
        return {'analysis_sha256': sha256_file(own / 'analysis.md'), 'frames': len(frames),
                'video_section_chars': len(section)}

    # ---------------------------------------------------------- generation

    def stats(self, rows=None):
        """Usage statistics from actor experiences; retrieval hints only."""
        skills = {}
        if rows is None:
            rows = [row for path in self.experience_files() for row in read_experiences(path)
                    if row.get('id') not in self.excluded_experiences]
        for row in rows:
            if row.get('origin', 'actor') != 'actor':
                continue
            for name in row.get('skills_used') or []:
                entry = skills.setdefault(name, {'uses': 0, 'results': {r: 0 for r in RESULTS},
                                                 'last_used': None, 'layouts': []})
                entry['uses'] += 1
                if row.get('result') in RESULTS:
                    entry['results'][row['result']] += 1
                entry['last_used'] = max(filter(None, [entry['last_used'], row.get('time')]), default=None)
                key = f"{row.get('task')}:{row.get('layout_id')}"
                if key not in entry['layouts']:
                    entry['layouts'].append(key)
        return {'generated': now(), 'skills': skills}

    def build_generated(self, rows=None):
        """Write INDEX.md (per task) and stats.json; both are derived, not logged."""
        lines = ['# Skill index (generated)', '']
        for task in TASKS:
            task_lines = [line for line in self.index_lines(task)]
            tool_lines = self.tool_lines(task)
            if task_lines or tool_lines:
                lines += [f'## {task}'] + task_lines
                lines += (['', '### Tools'] + tool_lines if tool_lines else []) + ['']
        with self._window():
            _write_readonly(self.root / 'INDEX.md', '\n'.join(lines).strip() + '\n')
            _write_readonly(self.root / 'stats.json', json.dumps(self.stats(rows), indent=1, ensure_ascii=False))

    def prompt_text(self, task, video_root=None):
        """Task brief and skill index for an actor's fixed opening.

        The instruction and priors are already in the task section, and empty
        sections are left out.
        """
        parts = []
        brief = self.brief(task)
        sections = parse_brief(brief) if brief else {}
        learned = [f'## {name}\n{sections[name]}' for name in BRIEF_SECTIONS
                   if name not in DEVELOPER_SECTIONS and sections.get(name)]
        if learned:
            parts.append('Task brief (from the demo video and earlier development; check it against what you '
                         'see)\n' + '\n'.join(learned))
        else:
            parts.append('Task brief: nothing learned yet.')
        lines = self.index_lines(task)
        if lines:
            parts.append('Skills (open one with skill(name); "judged effective" is a model judgment, not a '
                         'verified success; the evidence is counted by the kind of episode it comes from: fresh = a '
                         'new layout without replay or history/, history = a new layout with history/ opened, '
                         'retry = a layout played before, replay, unrecorded = a new layout from before history/ use '
                         'was recorded, this episode = the one now running, other = not classified)\n'
                         + '\n'.join(lines))
        else:
            parts.append('Skills: none yet.')
        tools = self.tool_lines(task)
        if tools:
            parts.append('Tools for this task (import one with tool(name): it returns a Python module of the names in '
                         'brackets, and its doc says what each function takes and returns; evidence as for skills)\n'
                         + '\n'.join(tools))
        else:
            parts.append('Tools: none yet.')
        video = Path(video_root or self.root) / 'video' / task
        if (video / 'analysis.md').is_file():
            parts.append(f'The full demo video analysis is in {video / "analysis.md"} (saved frames in '
                         f'{video / "frames"}); read it with Python.')
        return '\n\n'.join(parts)

    # ----------------------------------------------------------- integrity

    def changelog(self):
        path = self.root / 'changelog.jsonl'
        return read_experiences(path)

    def evidence_issues(self, run_checker):
        """Evidence entries whose source run or hash no longer checks out."""
        issues = []
        for name in self.skill_names():
            try:
                meta, _ = load_skill(self.skill_path(name))
            except (KnowledgeError, ValueError, KeyError):
                continue  # reported by problems()
            for entry in meta['evidence']:
                issue = run_checker(entry)
                if issue:
                    issues.append({'skill': name, 'run': entry.get('run'), 'ref': entry.get('ref'), 'issue': issue})
        for meta_path in sorted((self.root / 'tools').glob('*/*.json')):
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                evidence = list(meta['evidence'])
            except (OSError, ValueError, KeyError, TypeError):
                continue  # reported by problems()
            for entry in evidence:
                issue = run_checker(entry)
                if issue:
                    issues.append({'tool': meta_path.stem, 'task': meta_path.parent.name, 'run': entry.get('run'),
                                   'ref': entry.get('ref'), 'issue': issue})
        return issues

    def problems(self, run_checker=None):
        """Format and integrity problems; empty when a snapshot may be taken.

        Evidence whose source no longer checks out does not block a snapshot; it is listed
        as invalid and left out of the index (see snapshot).
        """
        problems = []
        last = {}
        for record in self.changelog():
            last[record['file']] = record.get('after_sha256')
        tracked = [path for folder in ('skills', 'task_briefs') for path in sorted((self.root / folder).glob('*.md'))]
        tracked += sorted((self.root / 'video').glob('*/analysis.md'))
        for path in tracked:
            relative = str(path.relative_to(self.root))
            if last.get(relative) != sha256_file(path):
                problems.append(f'{relative} was changed without the knowledge helpers')
        names = set()
        for path in sorted(self.skills_dir.glob('*.md')) if self.skills_dir.is_dir() else []:
            try:
                meta, body = load_skill(path)
                if meta['name'] != path.stem or not SKILL_NAME.fullmatch(meta['name'] or ''):
                    raise KnowledgeError('name does not match the file name')
                if meta['type'] not in SKILL_TYPES or meta['status'] not in ('active', 'retired'):
                    raise KnowledgeError('invalid type or status')
                check_body(body)
                if meta['name'] in names:
                    raise KnowledgeError('duplicate name')
                names.add(meta['name'])
                for entry in meta['evidence']:
                    if entry.get('verdict') not in VERDICTS:
                        raise KnowledgeError('invalid evidence verdict')
            except (KnowledgeError, ValueError, KeyError) as exc:
                problems.append(f'{path.name}: {exc}')
        problems += self._tool_problems(last)
        for task in TASKS:
            brief = self.brief(task)
            if brief and len(brief) > BRIEF_LIMIT:
                problems.append(f'task_briefs/{task}.md exceeds {BRIEF_LIMIT} characters')
        return problems

    def _tool_problems(self, last):
        """Each tool's metadata matches its last changelog entry, is well formed, and names the hash of code that still
        passes the checks; every code file has its metadata."""
        problems = []
        folder = self.root / 'tools'
        for task_folder in sorted(folder.iterdir()) if folder.is_dir() else []:
            if not task_folder.is_dir() or task_folder.name not in TASKS:
                problems.append(f'tools/{task_folder.name} is not the folder of a task')
                continue
            for path in sorted(task_folder.iterdir()):
                relative = str(path.relative_to(self.root))
                if path.name.startswith('.'):
                    continue
                if path.suffix == '.py':
                    if not path.with_suffix('.json').is_file():
                        problems.append(f'{relative} has no metadata')
                    continue
                if path.suffix != '.json':
                    problems.append(f'{relative} is not a tool file')
                    continue
                if last.get(relative) != sha256_file(path):
                    problems.append(f'{relative} was changed without the knowledge helpers')
                try:
                    meta, code = self.read_tool(path.stem, task_folder.name)
                    problem = tool_meta_problem(meta, path.stem, task_folder.name)
                    if problem:
                        raise KnowledgeError(problem)
                    task_tools.check_code(code)
                    task_tools.check_doc(meta['doc'])
                except (KnowledgeError, ToolError, ValueError, KeyError, TypeError) as exc:
                    problems.append(f'{relative}: {exc}')
        return problems

    def admit(self, run_checker=None, experience_checker=None, run_kinds=None):
        """Check evidence and admit experiences one by one; refused rows stop being searchable.

        Returns (invalid evidence, admitted rows per experience file, refused rows) and records the
        outcome in evidence_issues.json and experience_admission.json. run_kinds ({run_id: kind} of the
        development episodes), when given, is saved as run_kinds.json for the index.
        """
        issues = self.evidence_issues(run_checker) if run_checker is not None else []
        self.invalid = {(issue['run'], issue['ref']): issue['issue'] for issue in issues}
        admitted, excluded = {}, []
        for path in self.experience_files():
            rows = []
            for row in read_experiences(path):
                reason = experience_checker(row) if experience_checker is not None else None
                if reason:
                    excluded.append({'run': row.get('run_id'), 'id': row.get('id'), 'reason': reason})
                else:
                    rows.append(row)
            admitted[path] = rows
        self.excluded_experiences = {row['id'] for row in excluded}
        with self._window():
            _write_readonly(self.root / 'evidence_issues.json', json.dumps({'invalid_evidence': issues}, indent=1))
            _write_readonly(self.root / 'experience_admission.json',
                            json.dumps({'excluded_experiences': excluded}, indent=1))
            if run_kinds is not None:
                self.run_kinds = {str(k): v for k, v in run_kinds.items()}
                _write_readonly(self.root / 'run_kinds.json', json.dumps(self.run_kinds, indent=1, sort_keys=True))
        return issues, admitted, excluded

    def snapshot(self, run_checker=None, full=False, label=None, experience_checker=None, run_kinds=None):
        """Freeze the knowledge.

        Every snapshot holds the skills, tools, task briefs, generated index and statistics, and the video
        analysis, so a run's snapshot ID pins the exact analysis it could read. A full (release)
        snapshot also holds the admitted experiences and only the active skills, and the active tools of
        the released task (self.run['task']), with at least one valid piece of evidence. Evidence whose source no longer checks out (run_checker) is listed
        and not counted; experiences are admitted one by one (experience_checker(row) -> reason or
        None) and refused ones are no longer searchable. Format errors and unlogged edits refuse the
        snapshot.
        """
        problems = self.problems()
        if problems:
            raise KnowledgeError('snapshot refused:\n' + '\n'.join(problems))
        issues, admitted, excluded = self.admit(run_checker, experience_checker, run_kinds)
        admitted_rows = [row for rows in admitted.values() for row in rows]
        self.build_generated(admitted_rows)
        excluded_skills, excluded_tools = [], []
        staging = self.root / 'snapshots' / f'.staging.{uuid.uuid4().hex[:8]}'
        previous = self._latest_snapshot_files()
        try:
            def copy(path):
                relative = path.relative_to(self.root)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Snapshots are read-only, so an unchanged file (e.g. a video frame) is a hard link
                # to the latest snapshot's copy instead of another copy.
                earlier = previous.get(str(relative))
                if earlier is not None and earlier[1] == sha256_file(path):
                    try:
                        os.link(earlier[0], destination)
                        return
                    except OSError:
                        pass
                shutil.copy2(path, destination)
            for path in sorted(self.skills_dir.glob('*.md')) if self.skills_dir.is_dir() else []:
                meta, _ = load_skill(path)
                if full and meta['status'] != 'active':
                    excluded_skills.append({'skill': meta['name'], 'reason': 'retired'})
                    continue
                if full and not self.valid_evidence(meta['evidence']):
                    excluded_skills.append({'skill': meta['name'], 'reason': 'no valid evidence'})
                    continue
                copy(path)
            # Tools: a release holds only the active tools of the released task with valid evidence.
            for meta_path in sorted((self.root / 'tools').glob('*/*.json')):
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                reason = None
                if full and self.run.get('task') and meta['task'] != self.run['task']:
                    reason = 'another task'
                elif full and meta['status'] != 'active':
                    reason = 'retired'
                elif full and not self.valid_evidence(meta['evidence']):
                    reason = 'no valid evidence'
                if reason:
                    excluded_tools.append({'tool': meta['name'], 'task': meta['task'], 'reason': reason})
                    continue
                copy(meta_path.with_suffix('.py'))
                copy(meta_path)
            for path in sorted((self.root / 'task_briefs').glob('*.md')):
                copy(path)
            if (self.root / 'run_kinds.json').is_file():
                copy(self.root / 'run_kinds.json')  # the index counts each skill's evidence by kind of episode
            video = self.root / 'video'
            for path in sorted(video.rglob('*')) if video.is_dir() else []:
                if path.is_file() and not path.name.startswith('.'):
                    copy(path)
            if full:
                for path, rows in admitted.items():
                    target_file = staging / path.relative_to(self.root)
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                                           encoding='utf-8')
            frozen = Knowledge(staging)
            frozen.invalid = dict(self.invalid)
            frozen.run_kinds = dict(self.run_kinds)
            frozen.build_generated(admitted_rows)
            files = sorted(path for path in staging.rglob('*') if path.is_file())
            content = {str(path.relative_to(staging)): sha256_file(path) for path in files}
            experience_lines = {str(path.relative_to(self.root)): len(rows) for path, rows in admitted.items()}
            identity = {'content': {k: v for k, v in content.items() if k != 'stats.json'},
                        'experience_lines': experience_lines, 'full': full, 'invalid_evidence': issues,
                        'excluded_experiences': excluded, 'excluded_skills': excluded_skills}
            if excluded_tools:  # absent otherwise, so knowledge without tools keeps its snapshot IDs
                identity['excluded_tools'] = excluded_tools
            snapshot_id = sha256_bytes(json.dumps(identity, sort_keys=True).encode())[:16]
            target = self.root / 'snapshots' / snapshot_id
            if not target.exists():
                manifest = {'id': snapshot_id, 'created': now(), 'full': full, 'label': label, 'files': content,
                            'experience_lines': experience_lines, 'invalid_evidence': issues,
                            'excluded_experiences': excluded, 'excluded_skills': excluded_skills,
                            'excluded_tools': excluded_tools}
                (staging / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
                for path in staging.rglob('*'):
                    if path.is_file():
                        path.chmod(0o444)
                os.replace(staging, target)
        finally:
            if staging.exists():
                # Files may be hard links into earlier snapshots: remove them without changing their mode.
                shutil.rmtree(staging)
        return json.loads((target / 'manifest.json').read_text())

    def _latest_snapshot_files(self):
        """{relative path: (path, sha256)} of the most recent snapshot, for hard links to unchanged files."""
        folder = self.root / 'snapshots'
        manifests = sorted(folder.glob('*/manifest.json'), key=lambda path: path.stat().st_mtime) \
            if folder.is_dir() else []
        if not manifests:
            return {}
        latest = manifests[-1].parent
        try:
            files = json.loads(manifests[-1].read_text())['files']
        except (OSError, ValueError, KeyError):
            return {}
        return {relative: (latest / relative, digest) for relative, digest in files.items()}


class _NoWindow:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def verify_snapshot(path):
    """Check a frozen snapshot against its manifest; returns the manifest."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        if sha256_file(path / relative) != digest:
            raise KnowledgeError(f'snapshot {manifest["id"]}: {relative} was modified')
    return manifest
