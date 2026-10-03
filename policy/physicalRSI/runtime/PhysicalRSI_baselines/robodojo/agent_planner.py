"""API planning for executable skill selection and optional advisory supervision."""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from copy import deepcopy


class AgentPlanner:
    def __init__(self, *, endpoint=None, model=None):
        self.api_key = os.environ.get('PHYSICALRSI_AGENT_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('ARK_API_KEY')
        if not self.api_key:
            raise RuntimeError('PHYSICALRSI_AGENT_API_KEY or OPENAI_API_KEY is required')
        self.endpoint = endpoint or os.environ.get('PHYSICALRSI_AGENT_ENDPOINT')
        self.model = model or os.environ.get('PHYSICALRSI_AGENT_MODEL')
        if not self.endpoint or not self.model:
            raise ValueError('An explicit agent endpoint and model are required')

    def plan(self, *, instruction, observation, memory, previous_plan, compositions=None):
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

        context = dict(instruction=instruction, observation=encode(observation),
                       task_specific_policy_memory=memory, previous_plan=previous_plan)
        if compositions is not None:
            context['available_compositions'] = compositions
        messages = [
            {'role': 'system', 'content': 'Interpret the instruction, decompose the next subgoal, and replan using current observations and the previous plan. Provide an advisory subgoal for the execution trace; it will not replace the policy instruction. Return only JSON with one nonempty string field: subgoal. Do not choose models, checkpoints, or independent task policies.'},
            {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(context, allow_nan=False)}] + images},
        ]
        if compositions is not None:
            messages[0]['content'] = (
                'Choose one of the supplied executable skill compositions using the instruction, '
                'observations, and memory. Return JSON with exactly composition and rationale '
                'as nonempty strings. composition must be a supplied name. Do not invent skills.'
            )
        body = json.dumps(dict(model=self.model, temperature=0, messages=messages,
                               response_format={'type': 'json_object'})).encode()
        request = urllib.request.Request(self.endpoint, data=body, headers={
            'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
        plan = json.loads(result['choices'][0]['message']['content'])
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
