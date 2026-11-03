"""API planning for executable skill selection and optional advisory supervision."""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
import urllib.error
import time
from copy import deepcopy

from .capability_memory import catalogue_with_observations


def agent_observation(observation):
    """Expose only RGB, camera calibration and robot proprioception to planning."""
    state_fields = {
        'arm_joint_state', 'ee_joint_state', 'ee_pose', 'tcp_pose', 'delta_ee_pose',
        'left_arm_joint_state', 'right_arm_joint_state',
        'left_ee_joint_state', 'right_ee_joint_state',
        'left_ee_pose', 'right_ee_pose', 'left_tcp_pose', 'right_tcp_pose',
        'left_delta_ee_pose', 'right_delta_ee_pose',
    }
    camera_fields = {'color', 'intrinsic_matrix', 'extrinsic_matrix', 'shape'}
    result = {}
    if 'instruction' in observation:
        result['instruction'] = observation['instruction']
    if isinstance(observation.get('state'), dict):
        result['state'] = {key: value for key, value in observation['state'].items()
                           if key in state_fields}
    if isinstance(observation.get('vision'), dict):
        result['vision'] = {
            name: {key: value for key, value in camera.items() if key in camera_fields}
            for name, camera in observation['vision'].items() if isinstance(camera, dict)
        }
    return result


def _planner_value(value, key=None):
    """Project implementation memory into a compact capability view for the agent."""
    if isinstance(value, dict):
        result = {}
        for name, child in value.items():
            if name in {'implementation', 'source', 'source_name', 'provenance', 'capability_id', 'skill_name',
                        'configuration', 'evidence'}:
                continue
            if name == 'operation_memory':
                if isinstance(child, list):
                    result[name] = [{'operation_count': len(item.get('operation_ids', []))}
                                    for item in child if isinstance(item, dict)]
                elif child:
                    result[name] = {'available': True}
                continue
            result[name] = _planner_value(child, name)
        return result
    if isinstance(value, list):
        return [_planner_value(item, key) for item in value]
    return value


def _planner_catalogue(value):
    return _planner_value(value)


class AgentPlanner:
    def __init__(self, *, endpoint=None, model=None):
        self.api_key = os.environ.get('PHYSICALRSI_AGENT_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('ARK_API_KEY')
        if not self.api_key:
            raise RuntimeError('PHYSICALRSI_AGENT_API_KEY or OPENAI_API_KEY is required')
        self.endpoint = endpoint or os.environ.get('PHYSICALRSI_AGENT_ENDPOINT')
        self.model = model or os.environ.get('PHYSICALRSI_AGENT_MODEL')
        if not self.endpoint or not self.model:
            raise ValueError('An explicit agent endpoint and model are required')

    def plan(self, *, instruction, observation, memory, previous_plan, compositions=None,
             operation_program=None):
        if compositions is not None and operation_program is not None:
            if not isinstance(compositions, dict) or not isinstance(operation_program, dict):
                raise ValueError('Compositions and operation registry must be mappings')
        images = []

        def encode(value):
            if isinstance(value, dict):
                excluded = {'task_name', 'benchmark_task', 'layout_id', 'seed_id', 'env_idx', 'depth', 'segmentation', 'pointcloud'}
                return {k: encode(v) for k, v in value.items() if k not in excluded}
            if isinstance(value, (bytes, bytearray)):
                from XPolicyLab.utils.process_data import decode_image_bit
                return encode(decode_image_bit(value))
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            if hasattr(value, 'shape') and hasattr(value, 'tolist'):
                if len(value.shape) == 3:
                    from PIL import Image
                    import numpy as np
                    array = np.asarray(value)
                    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
                        array = array.transpose(1, 2, 0)
                    if array.dtype != np.uint8 or array.shape[-1] not in (3, 4):
                        raise ValueError('Agent cameras require uint8 RGB or RGBA images')
                    stream = io.BytesIO()
                    Image.fromarray(array).convert('RGB').save(stream, format='PNG')
                    images.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}})
                    return {'image_index': len(images) - 1}
                return value.tolist()
            if hasattr(value, 'item'):
                return value.item()
            return value

        context = dict(instruction=instruction, observation=encode(agent_observation(observation)),
                       task_specific_policy_memory=memory, previous_plan=previous_plan)
        if compositions is not None:
            context['available_compositions'] = _planner_catalogue(catalogue_with_observations(compositions, observation))
        if operation_program is not None:
            context['operation_program'] = _planner_catalogue(operation_program)
        messages = [
            {'role': 'system', 'content': 'Interpret the instruction, decompose the next subgoal, and replan using current observations and the previous plan. Provide an advisory subgoal for the execution trace; it will not replace the policy instruction. Return only JSON with one nonempty string field: subgoal. Do not choose models, checkpoints, or independent task policies.'},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(context, allow_nan=False)}] + images},
        ]
        if compositions is not None and operation_program is None:
            messages[0]['content'] = (
                'Choose the supplied skill composition whose capability memory fits the instruction, '
                'current observations, and available contracts. Return JSON with exactly '
                'composition and rationale as nonempty strings. Use a supplied composition '
                'name and the capability contracts, geometry limits, and observed conditions '
                'provided in memory. Ground the choice in observed requirements rather than score priors. '
                'Bind its declared operations in order and keep the implementation stable for the episode.'
            )
        if operation_program is not None:
            messages[0]['content'] = (
                'Use the supplied capability memory and operation library to construct the episode plan. '
                'Return JSON with exactly composition, operations, and rationale. Choose a '
                'supplied composition name. For a composition without the operation library, '
                'return an empty operations list. For the operation library, select exactly '
                'one operation handle when its declared capability covers the requested '
                'manipulation and its execution conditions fit the visible scene. Compare '
                'the supplied capabilities by supported behavior, geometry and input/output '
                'contracts. Current RGB availability is not a reason to prefer a learned '
                'composition: structured visual operations also use RGB. When a structured '
                'operation has explicit compatible support, use that evidence; otherwise '
                'choose a suitable learned composition. Semantic similarity alone does not '
                'establish geometric compatibility. A missing structured object label is '
                'unknown, not a mismatch: use the instruction together with RGB and memory. '
                'Do not infer unsupported behavior from a capability name. Treat identifiers '
                'as opaque handles and use only the capability contracts and parameters '
                'described in memory. Episode reset or return-home behavior is a shared '
                'lifecycle action; it must not disqualify a capability that covers the '
                'instruction\'s manipulation objective.'
            )
            if compositions is None:
                messages[0]['content'] = (
                    'Construct an executable sequence from the supplied operation memory using the '
                    'instruction and observations. Return JSON with exactly operations '
                    '(a nonempty list of operation IDs) and rationale (a nonempty string). '
                    'Match input and output contracts and use each operation at most once.'
                )
        body = json.dumps(dict(model=self.model, temperature=0, messages=messages,
                               response_format={'type': 'json_object'})).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers={
            'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'})
        result = None
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read())
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                last_error = error
                if attempt == 2:
                    raise RuntimeError('Agent API unavailable after three transport attempts') from error
                time.sleep(0.5 * (attempt + 1))
        plan = json.loads(result['choices'][0]['message']['content'])
        if operation_program is not None:
            expected = {'composition', 'operations', 'rationale'} if compositions is not None else {'operations', 'rationale'}
            if (not isinstance(plan, dict) or set(plan) != expected
                    or not isinstance(plan.get('rationale'), str) or not plan['rationale'].strip()
                    or not isinstance(plan.get('operations'), list)
                    or not 0 <= len(plan['operations']) <= 64
                    or any(not isinstance(name, str) for name in plan['operations'])):
                raise ValueError('Agent must return a bounded executable operation sequence')
            names = plan['operations']
            if compositions is not None:
                if (not isinstance(plan.get('composition'), str)
                        or plan['composition'] not in compositions):
                    raise ValueError('Agent operation plan contains an unknown composition')
                uses_library = compositions[plan['composition']]['steps'][-1] == 'memory-program-library'
                if uses_library and not names:
                    raise ValueError('Operation library requires an executable operation')
                if not uses_library and names:
                    raise ValueError('Operations cannot be attached to an independent execution skill')
            if not names:
                if compositions is not None:
                    return plan
                raise ValueError('Agent must return a nonempty executable operation sequence')
            registry = operation_program['operations']
            if len(set(names)) != len(names) or any(name not in registry for name in names):
                raise ValueError('Agent operation sequence contains duplicate or unregistered operations')
            contract = operation_program['input']
            for name in names:
                if registry[name]['input'] != contract:
                    raise ValueError('Agent operation sequence has incompatible contracts')
                contract = registry[name]['output']
            if contract != operation_program['output']:
                raise ValueError('Agent operation sequence does not produce the required output')
            return plan
        if compositions is not None:
            if (set(plan) != {'composition', 'rationale'}
                    or plan.get('composition') not in compositions
                    or not isinstance(plan.get('rationale'), str) or not plan['rationale'].strip()):
                raise ValueError('Agent must select a registered skill composition')
            return plan
        if set(plan) != {'subgoal'} or not isinstance(plan['subgoal'], str) or not plan['subgoal'].strip():
            raise ValueError('Agent must return exactly one nonempty subgoal')
        return plan


class AgentLoop:
    """A low-frequency supervisor records advice without modifying policy inputs."""

    def __init__(self, planner, memory, record, replan_interval=100, max_calls_per_episode=1):
        if type(replan_interval) is not int or replan_interval < 1:
            raise ValueError("Positive integer replan_interval required")
        if type(max_calls_per_episode) is not int or max_calls_per_episode < 1:
            raise ValueError("Positive integer max_calls_per_episode required")
        self.max_calls_per_episode = max_calls_per_episode
        self.calls = {}
        self.replan_interval = replan_interval
        self.ages = {}
        self.instructions = {}
        self.planner = planner
        self.memory = deepcopy(memory)
        self.record = record
        self.previous = {}
        self.ready = set()

    def prepare(self, observations):
        self.ready.clear()
        prepared = []
        indices = [obs.get('env_idx', 0) for obs in observations]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError('Distinct observed environments required')
        for obs, index in zip(observations, indices):
            instruction = obs.get('instruction')
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('Natural-language instruction required for agent planning')
            needs_plan = (index not in self.previous or self.ages[index] >= self.replan_interval
                          or self.instructions.get(index) != instruction)
            if needs_plan and self.calls.get(index, 0) < self.max_calls_per_episode:
                plan = self.planner.plan(instruction=instruction, observation=obs,
                                         memory=deepcopy(self.memory), previous_plan=self.previous.get(index))
                if set(plan) != {'subgoal'} or not isinstance(plan['subgoal'], str) or not plan['subgoal'].strip():
                    raise ValueError('Invalid agent subgoal')
                self.record(dict(env_idx=index, instruction=instruction, plan=plan))
                self.previous[index] = deepcopy(plan)
                self.instructions[index] = instruction
                self.ages[index] = 0
                self.calls[index] = self.calls.get(index, 0) + 1
            plan = self.previous[index]
            prepared.append(obs)
        self.ready = set(indices)
        return prepared

    def consume(self, indices):
        indices = list(indices)
        if not indices or len(set(indices)) != len(indices) or not set(indices) <= self.ready:
            raise RuntimeError('Fresh agent planning is required before each action chunk')
        self.ready.difference_update(indices)
        for index in indices:
            self.ages[index] += 1

    def reset(self):
        self.calls.clear()
        self.ages.clear()
        self.instructions.clear()
        self.previous.clear()
        self.ready.clear()
