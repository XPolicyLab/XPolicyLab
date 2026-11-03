"""One agent per unique run directory; never schedules tasks or retries episodes."""
from __future__ import annotations
import traceback
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import signal
import time
from ..runtime.io_utils import atomic, read_json
from ..runtime.task_context import task_reference
from ..runtime.protocol import PLAN_SCHEMA, DECISION_SCHEMA, obj, normalize_plan, validate_decision
from ..runtime.termination import END_SCHEMA, validate_termination, TerminationRejected
from .api import API, APIError, image_content, text_content, output_limit_reached
from .sandbox import Sandbox, MAX_VIEW_IMAGE_EDGE_PX
from .observations import stage_observation, compact_numbers, digest
from .model_context import model_observation, model_json
from .history import PlanningHistory, fingerprint, memory_metrics
from .archive import HistoryJournal
from .planning_budget import PlanningResponseLog

OUTPUT_LIMIT_HINT = (
    'Your previous response reached its output limit. You have already spent substantial effort reasoning. '
    'Use the current observation and available plan to decide the next useful action promptly. '
    'Avoid repeating settled analysis.')

POLICY = Path(__file__).resolve().parents[1]
PLAN_TOOLS = [
    {'name': 'submit_trajectory', 'description': 'Submit pose targets and gripper commands for one action group, or two with an intermediate visual checkpoint.', 'schema': PLAN_SCHEMA},
    {'name': 'run_python', 'description': 'Offline persistent Python with np, Image, plt, obs and geom preloaded. Use print() for values. Save crops with obs.save_crop and pass their paths in return_images to display them after successful execution, without another tool call. No simulator or network access.', 'schema': obj({'code': {'type': 'string', 'description': 'Executable Python source; use real newlines after JSON decoding, not double-escaped source.'}, 'return_images': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4, 'description': f'Optional absolute /work PNG/JPEG paths, in display order. Each image must have long edge <= {MAX_VIEW_IMAGE_EDGE_PX} px and size <= 8 MB. Image errors do not undo or rerun Python.'}}, required=['code'])},
    {'name': 'view_image', 'description': f'Inspect a saved /work image. Long edge must be <= {MAX_VIEW_IMAGE_EDGE_PX} px; oversized images return an error.', 'schema': obj({'path': {'type': 'string'}})},
    {'name': 'end_attempt', 'description': 'Abandon only an irrecoverable physical state with no feasible progress or recovery. Low budget is not a stopping reason.', 'schema': END_SCHEMA}]


CHECK_TOOLS = [{'name': 'checkpoint_decision', 'description': 'Approve the pending second group only if current images support the expected intermediate outcome and no mishap; otherwise discard it and replan.', 'schema': DECISION_SCHEMA}]

def planning_budget_notice(req, now=None):
    seconds = max(0, int(req['deadline']-(time.time() if now is None else now)-1))
    return f'{seconds} wall seconds remain in this planning window.'


def view_image_context(path, sandbox, root, req):
    """Attribute canonical images and hash-matched crop metadata; never guess sources."""
    relative = path.relative_to(sandbox.work)
    context = {'tool_name': 'view_image', 'path': '/work/'+relative.as_posix(),
               'requested_at_observation_id': req['observation_id'],
               'source_observation_id': None, 'camera': None,
               'crop_or_transform': 'Not recorded for generated files; consult the producing run_python code.'}
    if len(relative.parts) == 3 and relative.parts[0] == 'observations':
        observation_id = relative.parts[1]
        manifest = read_json(root/'observations'/observation_id/'manifest.json', {})
        for camera in manifest.get('cameras', []):
            item = manifest.get('files', {}).get(camera, {})
            if item.get('name') == relative.name and item.get('sha256') == digest(path):
                context.update(source_observation_id=observation_id, camera=camera,
                               crop_or_transform='None; exact original camera file verified by SHA-256.')
                break
    # Crop transforms are helper-reported metadata, not independently measured geometry.
    # Ignore stale/invalid sidecars; never attribute an overwritten image from old metadata.
    sidecar = path.with_name(path.name+'.crop.json')
    if (context['camera'] is None and sidecar.is_file() and not sidecar.is_symlink()
            and sidecar.stat().st_size <= 16000):
        try:
            from PIL import Image
            crop = json.loads(sidecar.read_text())
            obs_id, camera = crop['source_observation_id'], crop['camera']
            if not isinstance(obs_id, str) or not obs_id.startswith('obs_') or not obs_id[4:].isdigit():
                return context
            source = root/'observations'/obs_id
            manifest = read_json(source/'manifest.json', {})
            item = manifest['files'][camera]
            bounds = crop['bbox_xyxy']
            if not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds):
                return context
            x0,y0,x1,y1 = bounds
            with Image.open(source/item['name']) as original, Image.open(path) as image:
                valid = (camera in manifest['cameras'] and crop['sha256'] == digest(path)
                         and crop['source_image_sha256'] == item['sha256'] == digest(source/item['name'])
                         and 0 <= x0 < x1 <= original.width and 0 <= y0 < y1 <= original.height
                         and image.size == (x1-x0,y1-y0))
            if valid:
                context.update(source_observation_id=obs_id, camera=camera,
                    crop_or_transform={'type':'native_resolution_crop', 'bbox_xyxy':bounds,
                        'original_pixel_offset_xy':[x0,y0], 'scale_xy':[1,1],
                        'attribution':'helper-reported transform; source and displayed image hashes verified'})
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return context


def tool_image(path_value, sandbox, root, req, log, provider, name, index):
    """Validate once, archive exact displayed bytes and attribute each attachment."""
    path = sandbox.image_path(path_value)
    log.mkdir(parents=True, exist_ok=True)
    saved = log/f'image_{index:02d}{path.suffix.lower()}'
    shutil.copy2(path, saved)
    context = view_image_context(path, sandbox, root, req)
    context.update(tool_name=name, sha256=digest(saved), archive_file=str(saved.relative_to(root)))
    atomic(log/f'image_{index:02d}.json', context)
    return image_content(saved, provider), context


def python_image_paths(arguments):
    if not isinstance(arguments, dict) or set(arguments)-{'code', 'return_images'}:
        raise ValueError('run_python accepts code and optional return_images only')
    if not isinstance(arguments.get('code'), str):
        raise ValueError('code must be a Python source string')
    paths = arguments.get('return_images', [])
    if not isinstance(paths, list) or len(paths) > 4 or any(not isinstance(p, str) for p in paths):
        raise ValueError('return_images must be an array of at most four /work paths')
    return paths


class Agent:
    def __init__(self, root, config):
        self.root, self.config = Path(root).resolve(), config
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root/'.agent.lock').open('w')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((self.root/name).exists() for name in ('planning_prefix.json', 'conversation_state.json', 'history_journal', 'api')):
            raise RuntimeError('Existing Agent archive; start a new session. Automatic resume is not supported.')
        self.api = API(self.root, config)
        self.sandbox = Sandbox(self.root/'sandbox', config['python_prefix'], config.get('sandbox', 'docker'),
                               config.get('sandbox_image','ubuntu:24.04'), config.get('bwrap_executable','bwrap'),
                               memory_limit_mib=config.get('python_memory_limit_mib', 16384))
        shutil.copy2(POLICY/'harness/helpers/vlm_geom.py', self.sandbox.work/'vlm_geom.py')
        shutil.copy2(POLICY/'harness/helpers/observation_io.py', self.sandbox.work/'observation_io.py')
        shutil.copy2(POLICY/'assets/geometry/calibration_notes.md', self.sandbox.work/'geometry_reference.md')
        for name in ('geometry_interfaces.md', 'geometry_evidence.md'):
            shutil.copy2(POLICY/'assets/geometry'/name, self.sandbox.work/name)
        self._ensure_history()

    def _ensure_history(self):
        if not hasattr(self, 'history'):
            self.history = PlanningHistory(self.config.get('context_history', 'first_recent'),
                initial=getattr(self, 'plan_messages', ()), journal=HistoryJournal(self.root/'history_journal'),
                max_context_bytes=self.api.context_max_bytes)
            self.plan_messages = self.history.archive  # Existing diagnostics retain the complete archive.

    def _planning_prefix(self):
        if not hasattr(self, '_frozen_prefix'):
            system = (POLICY/'assets/prompts/prompt.md').read_text()
            system += '\n'+(POLICY/'assets/prompts/geometry_context.md').read_text()
            runtime = getattr(self.sandbox, 'runtime', {})
            if runtime:
                system += '\nActual Python environment: '+json.dumps(runtime, separators=(',', ':'))+'\n'
            tools = copy.deepcopy(PLAN_TOOLS)
            if self.config.get('calibration_mode'):
                raise ValueError('calibration_mode is not supported in this review package; disable it for evaluation.')
            self._frozen_prefix = {'system': system, 'tools': tools}
            self._prefix_sha256 = fingerprint(self._frozen_prefix)
            atomic(self.root/'planning_prefix.json', self._frozen_prefix | {'sha256': self._prefix_sha256})
        return self._frozen_prefix['system'], self._frozen_prefix['tools']

    def _check_current(self, req):
        latest = read_json(self.root/'READY.json', {})
        if (time.time() >= req['deadline']-1 or (self.root/'END.json').exists()
                or (self.root/'STOP').exists()
                or (latest and any(latest.get(k) != req.get(k) for k in
                                   ('run_id', 'request_id', 'observation_id')))):
            raise TimeoutError('reply completed after deadline')

    def prepare(self, req):
        source, manifest, obs = stage_observation(self.root, self.sandbox, req)
        # The same current-task definitions are injected in task_requirements and queryable here.
        atomic(self.sandbox.work/'task_reference.json', task_reference(req['task']))
        visible = model_observation(obs, req['observation_id'], checking=req['kind'] == 'checkpoint')
        content = [text_content(model_json(visible), self.api.provider)]
        for camera in manifest['cameras']:
            content += [text_content('CURRENT '+req['observation_id']+' '+camera, self.api.provider),
                        image_content(source/manifest['files'][camera]['name'], self.api.provider)]
        if req['kind'] == 'checkpoint':
            # Action values stay at full precision; this reviewer cannot calculate new targets.
            content.append(text_content(model_json({
                'expectation': req['expectation'], 'completed_group': req['completed_group'],
                'remaining_groups': req['remaining_groups']}), self.api.provider))
        return [{'role': 'user', 'content': content}]

    def respond(self, req):
        self._ensure_history()
        payload = None
        self._checkpoint_current = []
        checking = req['kind'] == 'checkpoint'
        try:
            payload = self._review(req) if checking else self._respond(req)
            return payload
        finally:
            if checking:
                verdict = payload or {'decision': 'REPLAN', 'reason': 'checkpoint interrupted; second group not authorized'}
                self.history.extend(self._checkpoint_current)
                self.history.append({'role': 'user', 'content': [text_content(
                    'External visual checkpoint result for '+req['observation_id']+': '+model_json(verdict)+
                    '. This is a reviewer decision, not proof that the pending group executed or that the task succeeded. '
                    'REPLAN discards the pending group; recover from the next observation.', self.api.provider)]})
                atomic(self.root/'checkpoint_reviews'/(req['observation_id']+'.json'), {'request': req, 'verdict': verdict})
            elif self.history.pending_tools():
                self.api.append_results(self.history, [(call_id,
                    'Decision interrupted. No trajectory accepted by this tool call; any Python side effects may remain. Reassess from the next observation.', True)
                    for call_id in self.history.pending_tools()])
            atomic(self.root/'conversation_state.json', {'run_id':req.get('run_id'),
                'task':req.get('task'), 'latest_observation_id':req['observation_id'],
                **self.history.archive_state(), 'planning_prefix_sha256':getattr(self, '_prefix_sha256', None)})
            atomic(self.root/'latest_memory.json', self.history.latest_memory)

    def _review(self, req):
        """One isolated low-effort response; no Python, planning tools or main history."""
        def check_current():
            latest = read_json(self.root/'READY.json', {})
            if (time.time() >= req['deadline']-1 or (self.root/'END.json').exists()
                    or (self.root/'STOP').exists()
                    or (latest and latest.get('request_id') != req.get('request_id'))):
                raise TimeoutError('stale or expired checkpoint; second group not authorized')
        try:
            check_current()
            self._checkpoint_current = self.prepare(req)
            system = (POLICY/'assets/prompts/checkpoint_prompt.md').read_text()
            payload = self.api.payload(system, self._checkpoint_current, CHECK_TOOLS, checkpoint=True)
            response = self.api.call(payload, {**req, 'context_selection': {
                'mode':'isolated_checkpoint', 'parent_planning_round':self.history.round_id,
                'retained_rounds':[]}})
            check_current()
            calls = self.api.calls(response)
            if len(calls) != 1 or calls[0]['name'] != 'checkpoint_decision':
                raise ValueError('exactly one checkpoint_decision is required')
            return validate_decision(calls[0]['input'])
        except Exception as exc:
            atomic(self.root/'errors'/(req['observation_id']+'.json'), {'request': req, 'error': repr(exc)})
            return {'decision': 'REPLAN', 'reason': ('checkpoint unavailable or invalid: '+str(exc))[:300]}

    def _respond(self, req):
        if req['kind'] != 'plan':
            raise ValueError('Only planning observations are supported')
        self._check_current(req)
        system, tools = self._planning_prefix()
        current = copy.deepcopy(self.prepare(req))
        # Reserve room for the request envelope and per-call budget notices. This
        # boundary estimate triggers early rebuilds; API enforces the exact byte cap.
        prefix_bytes = len(json.dumps({'system':system,'tools':tools}, ensure_ascii=False,
                                     indent=2).encode()) + 4096
        if self.history.begin_round(req, upcoming=current, prefix_bytes=prefix_bytes):
            self.api.cache_anchor = None  # Numeric offsets no longer identify the same messages.
        current[0]['content'].append(text_content(
            'Planning context: '+model_json(self.history.observation_context()), self.api.provider))
        self.history.extend(current)
        messages = self.history
        # Persistent code namespace, fresh camera calibration, zero model calls for bookkeeping.
        code = ('import numpy as np, scipy\n'
                'from PIL import Image\n'
                'import matplotlib\n'
                'matplotlib.use("Agg")\n'
                'import matplotlib.pyplot as plt\n'
                'import vlm_geom, observation_io\n'
                'obs, geom = observation_io, vlm_geom\n'
                'observation_io.configure("/work/observations/'+req['observation_id']+'/manifest.json")\n'
                'vlm_geom.configure("/work/observations/'+req['observation_id']+'/observation.json")\n'
                'assert all(observation_io.image(c).shape[-1] == 3 for c in ("cam_head","cam_left_wrist","cam_right_wrist"))\n'
                'print("OBSERVATION_READY")')
        configured = self.sandbox.configure(code, self.root/'tools'/req['observation_id']/'configure',
                                            req['deadline']-1, req['observation_id'])
        status = json.loads(configured.split('\n', 1)[0])
        if status.get('returncode') != 0 or 'OBSERVATION_READY' not in configured:
            raise RuntimeError('observation_workspace_error: canonical images/calibration did not validate; no API call made')
        if getattr(self.sandbox, 'last_result', {}).get('variables_lost'):
            messages.append({'role':'user','content':[text_content(
                'Python workspace recovered with a fresh kernel. Calculated variables were lost; saved files and the retained conversation remain. '
                'np, Image, plt, obs and geom are available, with helpers configured for '+req['observation_id']+'. Recompute any needed variables.', self.api.provider)]})
        response_log, responses_used = PlanningResponseLog(), 0

        def record_budget(outcomes, status='tools'):
            atomic(self.root/'planning_budget'/req['observation_id']/f'{responses_used:02d}.json',
                   {'request_id': req.get('request_id'), 'observation_id': req['observation_id'],
                    'response_index': responses_used, **response_log.record(outcomes, status)})

        while True:
            self._check_current(req)
            if getattr(self, '_output_limit_hint_pending', False):
                messages.append({'role':'user','content':[text_content(OUTPUT_LIMIT_HINT, self.api.provider)]})
                self._output_limit_hint_pending = False
            messages.append({'role': 'user', 'content': [text_content(
                planning_budget_notice(req), self.api.provider)]})
            responses_used += 1
            selection = self.history.selection() | {'planning_prefix_sha256': self._prefix_sha256}
            try:
                response = self.api.call(self.api.payload(system, messages.active, tools, history=self.history),
                                         {**req, 'context_selection': selection})
            except APIError as exc:
                if exc.reason != 'api_generation_incomplete':
                    raise
                # Raw output stays in the API archive. Never append or execute partial
                # assistant/tool output; continue with explicit feedback.
                self._output_limit_hint_pending = exc.output_limit_reached
                record_budget([], 'api_generation_incomplete')
                self._check_current(req)
                messages.append({'role':'user','content':[text_content(
                    'The previous response was incomplete; no action or tool ran.',
                    self.api.provider)]})
                continue
            self._output_limit_hint_pending = output_limit_reached(response)
            self._check_current(req)
            self.api.append_response(messages, response)
            calls = self.api.calls(response)
            finals = [c for c in calls if c['name'] in ('submit_trajectory','end_attempt')]
            validation_error = None
            termination_rejected = False
            if len(calls) == 1 and finals:
                try:
                    call = calls[0]
                    if call['name'] == 'submit_trajectory':
                        plan, repairs = normalize_plan(call['input'])
                        atomic(self.root/'protocol_audit'/f'{req["observation_id"]}_{responses_used:02d}.json',
                               {'request': req, 'raw_input': call['input'], 'normalized_input': plan, 'repairs': repairs,
                                'memory_metrics': memory_metrics(plan.get('memory', ''))})
                        self._check_current(req)
                        self.history.accept_memory(plan, req, call['input'].get('memory'))
                        note = 'Trajectory accepted for execution. This confirms protocol validity only. The next observation will report actual state; do not assume execution or success.'
                        if any(r['path'] == '$.memory' and 'truncated' in r['reason'] for r in repairs):
                            note += ' The stored summary copy exceeded the 6,000-character guard and is incomplete. Replace it with a concise self-contained memory at the next submission; do not repeat these actions.'
                        self.api.append_results(messages, [(call['id'], note)])
                        record_budget([{'id': call['id'], 'name': call['name'], 'is_error': False}], 'accepted')
                        return plan
                    if call['name'] == 'end_attempt':
                        value = call['input']
                        observation = read_json(self.root/'observations'/req['observation_id']/'observation.json', {})
                        validate_termination(value, observation)
                        self.api.append_results(messages, [(call['id'], 'Termination request accepted; official environment result remains authoritative.')])
                        record_budget([{'id': call['id'], 'name': call['name'], 'is_error': False}], 'accepted')
                        return {'tool': 'end_attempt', **value}
                except TerminationRejected as e:
                    validation_error = str(e)
                    termination_rejected = True
                except (ValueError, TypeError, KeyError) as e:
                    validation_error = str(e)
            if validation_error and not termination_rejected:
                atomic(self.root/'protocol_errors'/f'{req["observation_id"]}_{responses_used:02d}.json',
                       {'request':req,'error':validation_error,'calls':calls})
            results = []
            outcomes = []
            images = []
            image_context = {}
            for i, call in enumerate(calls):
                log = self.root/'tools'/req['observation_id']/f'{responses_used:02d}_{i}'
                is_error = False
                repair_tools = []
                try:
                    if finals:
                        is_error = True
                        result = 'No motion submitted. ' + (validation_error or 'Submit one trajectory or termination tool alone.')
                        if not termination_rejected:
                            result += ' Fix only the named fields. Unused null fields may be omitted.'
                    elif call['name'] == 'run_python':
                        repair_tools = ['run_python']
                        paths = python_image_paths(call['input'])
                        result = self.sandbox.execute(call['input']['code'], log, req['deadline']-1)
                        is_error = json.loads(result.split('\n', 1)[0]).get('returncode') != 0
                        # Never rerun code after an image error, or attach potentially
                        # stale output images from a failed Python execution.
                        if paths and not is_error:
                            blocks = [text_content(result, self.api.provider)]
                            contexts = []
                            for image_index, value in enumerate(paths, 1):
                                try:
                                    block, context = tool_image(value, self.sandbox, self.root, req,
                                                               log, self.api.provider, 'run_python', image_index)
                                    contexts.append(context)
                                    if self.api.provider == 'anthropic':
                                        blocks.append(block)
                                    else:
                                        images.extend([text_content('Tool image provenance: '+model_json(
                                            {**context, 'tool_use_id':call['id'], 'image_index':image_index}), self.api.provider), block])
                                except Exception as exc:
                                    is_error = True
                                    repair_tools = ['run_python', 'view_image']
                                    error = f'Image {image_index} ({value}): {type(exc).__name__}: {exc}. Python already ran; inspect or display the saved file without repeating the calculation.'
                                    blocks.append(text_content(error, self.api.provider))
                            if self.api.provider == 'anthropic':
                                result = blocks
                                image_context[call['id']] = contexts
                            else:
                                result = '\n'.join(b['text'] for b in blocks)
                    elif call['name'] == 'view_image':
                        repair_tools = ['run_python', 'view_image']
                        block, context = tool_image(call['input']['path'], self.sandbox, self.root,
                                                   req, log, self.api.provider, 'view_image', 1)
                        if self.api.provider == 'anthropic':
                            result = [block]
                            image_context[call['id']] = context
                        else:
                            # Images are user input items in Responses; function output remains text.
                            result = 'Requested generated image follows.'
                            images.extend([text_content('Tool image provenance: '+model_json(
                                {**context, 'tool_use_id':call['id']}), self.api.provider), block])
                    else:
                        is_error = True
                        result = 'Unknown tool; no action executed.'
                except Exception as e:
                    is_error = True
                    result = f'Tool error: {type(e).__name__}: {e}'
                results.append((call['id'], result, is_error))
                outcomes.append({'id': call['id'], 'name': call['name'], 'is_error': is_error,
                                 'repair_tools': repair_tools if is_error else []})
            if results:
                self.api.append_results(messages, results, image_context=image_context)
                if images:
                    messages.append({'role': 'user', 'content': images})
            elif not calls:
                messages.append({'role': 'user', 'content': [text_content('No action executed. Use submit_trajectory for motion or an explicit settling wait; end_attempt is only for irrecoverable abandonment.', self.api.provider)]})
            record_budget(outcomes, 'termination_rejected' if termination_rejected else
                          'format_error' if finals else 'tools' if calls else 'no_tool')

    def run(self):
        seen = set()
        started = time.monotonic()
        try:
            # Prove isolation/runtime works BEFORE paid requests.
            result = self.sandbox.execute('import numpy, scipy, PIL, matplotlib\n'
                'assert numpy.__version__ == "1.26.4", "Expected pinned NumPy 1.26.4"\n'
                'matplotlib.use("Agg")\nimport matplotlib.pyplot as plt\n'
                'import cv2; assert cv2.getNumThreads() == 1\n'
                'print("SANDBOX_READY")', self.root/'sandbox_preflight', time.time()+360)
            if json.loads(result.split('\n', 1)[0]).get('returncode') != 0 or 'SANDBOX_READY' not in result:
                raise RuntimeError('Python sandbox preflight failed')
            while not (self.root/'END.json').exists():
                if not (self.root/'episode.json').exists() and time.monotonic()-started > 1800:
                    raise TimeoutError('simulator_startup_timeout')
                req = read_json(self.root/'READY.json')
                if not req or req['request_id'] in seen:
                    if (self.root/'STOP').exists():
                        return
                    time.sleep(.2); continue
                seen.add(req['request_id'])
                episode = read_json(self.root/'episode.json', {})
                if req['run_id'] != episode.get('run_id') or req['task'] != episode.get('task'):
                    raise RuntimeError('episode/request attribution mismatch')
                if time.time() >= req['deadline']-1:
                    continue
                try:
                    payload = self.respond(req)
                except Exception as e:
                    observation_timeout = (
                        isinstance(e, APIError) and e.reason == 'api_request_deadline'
                    ) or (isinstance(e, TimeoutError) and str(e) == 'reply completed after deadline')
                    atomic(self.root/'errors'/f'{req["observation_id"]}.json', {'request': req, 'error': repr(e), 'exception_type': type(e).__name__,
                        'infrastructure': not observation_timeout and isinstance(e, (APIError, OSError)),
                        'traceback': traceback.format_exc()})
                    # A planning-observation deadline consumes this turn, not the episode.
                    # Cancellation and other API failures retain their existing stop policy.
                    reason = 'planning_observation_wait_limit' if observation_timeout else e.reason if isinstance(e, APIError) else (
                        'observation_workspace_error' if 'observation_' in str(e) else 'api_or_planning_error')
                    payload = {'tool': 'harness_replan' if reason == 'planning_observation_wait_limit' else 'harness_stop', 'reason': reason}
                if req['kind'] == 'checkpoint' and payload.get('tool') == 'harness_replan':
                    payload = {'decision': 'REPLAN', 'reason': payload['reason']}
                latest = read_json(self.root/'READY.json', {})
                if time.time() < req['deadline'] and not (self.root/'END.json').exists() and latest.get('request_id') == req['request_id']:
                    atomic(self.root/'observations'/req['observation_id']/'reply.json',
                           {k: req[k] for k in ('run_id','request_id','observation_id')} | {'payload': payload})
        finally:
            self.sandbox.close(); self.lock.close()


def main():
    def interrupted(_signum, _frame):
        raise SystemExit('Harness termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--enable-openai', action='store_true', help='Explicitly enable the optional OpenAI transport')
    args = parser.parse_args()
    config = read_json(args.config)
    if config['provider'] == 'openai' and not args.enable_openai:
        parser.error('OpenAI transport requires --enable-openai')
    Agent(args.run_dir, config).run()


if __name__ == '__main__':
    main()
