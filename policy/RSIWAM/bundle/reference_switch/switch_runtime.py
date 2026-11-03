"""Optional adapter switching for a future service; no live r1 integration.

The service must register every episode before accepting its observations, drain
all episodes before apply_pending(), and supply real same-runtime CUDA acceptance.
This module never constructs a base policy or reloads its large checkpoints.
"""
import copy
import continuation_contract as continuation
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
import time

BASE = 'ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba'
REGISTRATION = '21e0e15f3026e6dfac8b44d3716185fe1fb7937101209958dfdbe2ed5654d860'
HELPER = '946f25497835c4a18a74731058b7dec25fa1acaf2aad4a4b4f78814cca2a735d'
SOURCE = '95feb351fc1211318d7edec9e3814679835029b7ba4314d04d25464b6a4691f0'

ONLINE_TRAINER = '808bc58925fa39e580809569c031a79b8347a984c63d0d2b4831a15b525f321b'
ONLINE_ENCODER = '7adb4deb019c849731d086d0885d727e363008e2cc443f5e3fa5fb73c1b21ba4'
ONLINE_SCOPES = {'recoverwam_success133_state_anchor_v2_parent20261032_lora20261033': '0acc3dfdd8e6ac5db922ecd78176fcacb9d39906ff5d6c9487061c4dcc5c764c', 'recoverwam_success133_original_parent20261032_lora20261033': 'f68004790c2b25fac3940234ebd1ba8c4885f4ab2277df0e809b88df8033d22f'}

def online_identity(candidate_id):
    require(candidate_id in ONLINE_SCOPES, 'Only the two registered online candidates are admitted')
    return dict(trainer_sha256=ONLINE_TRAINER, producer_mode='same_runtime_online_official_encoders',
        online_feature_module_sha256=ONLINE_ENCODER, online_scope_sha256=ONLINE_SCOPES[candidate_id],
        feature_cache_sha256=None, compatibility_sha256=None, shared_merged_cache_accepted=False,
        initial_lora_reference_mode='same_device_official_helper_reference')

def require_online_identity(identity, candidate_id):
    expected = online_identity(candidate_id)
    require(all(key in identity and type(identity[key]) is type(value) and identity[key] == value for key,value in expected.items()),
            'Online checkpoint provenance differs or claims an accepted merged cache')
    require(identity.get('format') == 'recoverwam-success133-action-lora-training-v1',
            'Unknown training identity format')
    initial = identity.get('initial_lora_sha256')
    require(isinstance(initial,str) and len(initial)==64 and all(c in '0123456789abcdef' for c in initial),
            'Missing real initializer tensor identity')

TRAINERS = {'808bc58925fa39e580809569c031a79b8347a984c63d0d2b4831a15b525f321b'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(binding):
    path = Path(binding['path'])
    require(path.is_absolute() and path.is_file() and sha(path) == binding['sha256'],
            'Artifact is missing or changed: ' + str(path))
    return path


def verify_candidate(binding, registration_path, admission_manifest_binding=None):
    require(admission_manifest_binding is not None, 'Frozen continuation admission manifest is required')
    admission = continuation.admitted_row(admission_manifest_binding, binding['candidate_id'])
    require(sha(registration_path) == REGISTRATION, 'The frozen new8 registration changed')
    candidates = read(registration_path)['candidates']
    candidate = next((c for c in candidates if c['candidate_id'] == binding['candidate_id']), None)
    require(candidate is not None, 'The candidate is not registered')
    checkpoint_path = checked(binding['checkpoint'])
    require(checkpoint_path.name == 'checkpoint.json' and checkpoint_path.parent.name == 'step000400',
            'Only a committed step000400 checkpoint may be evaluated')
    commit = read(checkpoint_path)
    require(type(commit.get('step')) is int and commit['step'] == 400, 'The checkpoint has not completed 400 optimizer updates')
    expected = dict(candidate_id=candidate['candidate_id'], base_identity=BASE,
        parent_checkpoint_sha256=candidate['parent_bridge']['sha256'],
        parent_recipe_sha256=candidate['parent_bridge']['recipe_sha256'],
        parent_bridge_seed=candidate['parent_bridge_seed'], recovery_recipe=candidate['recovery_recipe'],
        lora_seed=candidate['lora_seed'], init_seed=candidate['init_seed'], registration_sha256=REGISTRATION,
        helper_sha256=HELPER, consumer_source_tree_sha256=SOURCE,
        schedule_sha256=candidate['draw_schedule_binding']['sha256'],
        ordinary_config_sha256=candidate['ordinary_config_binding']['sha256'])
    identity = commit['identity']
    require(all(identity.get(key) == value for key, value in expected.items()),
            'Checkpoint candidate, seed, parent, or training schedule differs')
    continuation.validate_bundle(checkpoint_path.parent, admission, commit)
    require(str(checkpoint_path) == str(Path(admission['checkpoint_destination'])/'checkpoint.json'),
            'Switch checkpoint outside independently registered continuation import')
    require(binding['parent']['path'] == str(Path(admission['parent_destination'])/'adapter_final.pt')
            and binding['parent_recipe']['path'] == str(Path(admission['parent_destination'])/'recipe.json'),
            'Switch parent paths differ from the frozen continuation admission')
    require(binding['parent']['sha256'] == expected['parent_checkpoint_sha256']
            and binding['parent_recipe']['sha256'] == expected['parent_recipe_sha256'], 'Wrong frozen parent')
    parent, recipe = checked(binding['parent']), checked(binding['parent_recipe'])
    adapter, state = checkpoint_path.parent/'adapter.pt', checkpoint_path.parent/'training_state.pt'
    require(sha(adapter) == commit['adapter_sha256'] and sha(state) == commit['training_state_sha256'],
            'Committed adapter or optimizer-state bytes differ')
    return dict(candidate_id=candidate['candidate_id'], adapter=str(adapter), parent=str(parent),
        parent_recipe=str(recipe), parent_sha256=expected['parent_checkpoint_sha256'],
        commit=commit, checkpoint_sha256=binding['checkpoint']['sha256'])


def clear_all_history(wrapper):
    # wrapper.reset() intentionally keeps other clients alive, so a drained
    # candidate transition must explicitly clear all client and policy state.
    wrapper.model.reset()
    for name in ('_virtual_env_ids', '_client_last_use', '_last_infer_step', '_last_obs_by_client',
                 '_last_instruction_by_client', '_last_virtual_by_client', '_infer_count_by_virtual',
                 '_request_context'):
        getattr(wrapper, name).clear()
    wrapper._active_client = None
    wrapper._next_virtual_env_id = 0
    wrapper._client_use_counter = 0
    wrapper._seen_client_ids = False
    wrapper.last_obs = None
    wrapper.last_instruction = wrapper.default_instruction
    wrapper.history_guard_duplicates = 0
    wrapper.history_guard_mismatches = 0
    wrapper.model._request_context.clear()


def remove_adapters(policy, helper):
    model = policy.model
    for name, module in list(helper.lora.lora_modules(policy)):
        parent_name, _, leaf = name.rpartition('.')
        parent = model.action_expert.get_submodule(parent_name) if parent_name else model.action_expert
        setattr(parent, leaf, module.base)
    for name in ('recovery_bridge', '_action_lora_config', '_recovery_action_lora_config',
                 '_frozen_recovery_parent_identity', 'recovery_mode'):
        if hasattr(model, name):
            delattr(model, name)
    if hasattr(policy, 'recovery_adapter_metadata'):
        delattr(policy, 'recovery_adapter_metadata')
    model.requires_grad_(False)
    model.eval()
    model.action_expert.use_gradient_checkpointing = False
    model.mot.mot_checkpoint_mixed_attn = False


def load_candidate(policy, artifacts, helper):
    helper.load_frozen_parent(policy, artifacts['parent'],
        expected_sha256=artifacts['parent_sha256'], expected_base_identity=BASE)
    metadata = helper.load_adapter(artifacts['adapter'], policy,
        expected_parent_sha256=artifacts['parent_sha256'], expected_base_identity=BASE)
    require(metadata.get('steps') == 400 and metadata.get('candidate_id') == artifacts['candidate_id']
            and metadata.get('training_identity') == artifacts['commit']['identity'],
            'Loaded adapter metadata differs from the committed training identity')
    policy.recovery_adapter_metadata = metadata
    policy.model.recovery_mode = 'auto'
    policy.model.requires_grad_(False)
    policy.model.eval()
    policy.model.action_expert.use_gradient_checkpointing = False
    policy.model.mot.mot_checkpoint_mixed_attn = False
    return metadata


class SharedModelAdapterSwitch:
    def __init__(self, wrapper, helper, *, registration_path, accept_candidate, clear_extra_state,
                 admission_manifest_binding=None, before_model_change=None):
        require(sha(helper.__file__) == HELPER, 'The registered adapter helper changed')
        require(sha(registration_path) == REGISTRATION, 'The frozen new8 registration changed')
        self.wrapper, self.helper = wrapper, helper
        self.registration_path = Path(registration_path)
        self.accept_candidate, self.clear_extra_state = accept_candidate, clear_extra_state
        self.before_model_change = before_model_change or (lambda: None)
        self.admission_manifest_binding = admission_manifest_binding
        self.lock = threading.RLock()
        self.active = set()
        self.inflight = 0
        self.pending = None
        self.draining = False
        self.paused = False
        self.generation = 0
        self.candidate_id = 'official_baseline'
        self.model = wrapper.model.model
        require(not hasattr(self.model, 'recovery_bridge') and not hasattr(self.model, '_action_lora_config'),
                'Attach the switch controller to an accepted unadapted base')
        self.base = [(p, id(p), p.data_ptr(), p._version, p.dtype, p.device) for p in self.model.parameters()]
        self._check_base()

    def _check_base(self):
        require(self.wrapper.model.model is self.model, 'The base model object was replaced')
        present = {id(p) for p in self.model.parameters()}
        for parameter, identity, pointer, version, dtype, device in self.base:
            require(identity in present and id(parameter) == identity and parameter.data_ptr() == pointer
                    and parameter._version == version and parameter.dtype == dtype and parameter.device == device
                    and not parameter.requires_grad and parameter.grad is None, 'Frozen base parameter changed')
        require(not any(p.requires_grad for p in self.model.parameters()), 'Inference parameters must be frozen')

    def begin_episode(self, client_id, candidate_id, generation):
        with self.lock:
            require(not self.draining and not self.paused, 'The service is draining or paused')
            require(candidate_id == self.candidate_id and generation == self.generation, 'Stale candidate generation')
            require(isinstance(client_id, str) and client_id and client_id not in self.active, 'Repeated client episode')
            self.active.add(client_id)

    def end_episode(self, client_id, generation):
        with self.lock:
            require(generation == self.generation and client_id in self.active, 'Unknown or stale episode completion')
            self.wrapper.reset_client(dict(_recoverwam_client_id=client_id))
            self.active.remove(client_id)

    @contextmanager
    def prediction(self, client_id, generation):
        with self.lock:
            require(not self.paused and generation == self.generation and client_id in self.active,
                    'Inference requires an active episode of the current candidate')
            self.inflight += 1
            try:
                yield self.wrapper
            finally:
                self.inflight -= 1

    def request_switch(self, request_id, binding=None):
        with self.lock:
            require(isinstance(request_id, str) and request_id, 'A switch request identity is required')
            request = dict(request_id=request_id, binding=copy.deepcopy(binding))
            require(self.pending is None or self.pending == request, 'Another candidate switch is already pending')
            # Invalid requests must not interrupt admission for a healthy model.
            if binding is not None:
                verify_candidate(request['binding'], self.registration_path, self.admission_manifest_binding)
            self.pending = request
            self.draining = True
            return dict(status='WAITING_EPISODES_TO_DRAIN' if self.active else 'DRAINED_READY_TO_SWITCH',
                        active_episodes=len(self.active), generation=self.generation)

    def cancel_switch(self, request_id):
        with self.lock:
            require(self.pending is not None and self.pending['request_id'] == request_id,
                    'Only the matching pending request may be cancelled')
            self.pending = None
            self.draining = False

    def apply_pending(self):
        with self.lock:
            require(self.pending is not None and self.draining, 'No candidate switch is pending')
            require(not self.active and self.inflight == 0, 'Never switch an adapter during a live episode')
            request = self.pending
            try:
                artifacts = (verify_candidate(request['binding'], self.registration_path, self.admission_manifest_binding)
                             if request['binding'] is not None else None)
            except Exception:
                self.pending = None
                self.draining = False
                raise
            candidate_id = artifacts['candidate_id'] if artifacts else 'official_baseline'
            try:
                self._check_base()
                self.before_model_change()
                remove_adapters(self.wrapper.model, self.helper)
                clear_all_history(self.wrapper)
                self.clear_extra_state()
                if artifacts:
                    load_candidate(self.wrapper.model, artifacts, self.helper)
                self._check_base()
                acceptance = self.accept_candidate(self.wrapper, artifacts)
                for key in ('same_runtime_base_golden_exact', 'candidate_client_isolation_exact',
                            'native_parameters_exact', 'all_parameters_frozen'):
                    require(acceptance.get(key) is True, 'Candidate acceptance failed: ' + key)
                clear_all_history(self.wrapper)
                self.clear_extra_state()
                self._check_base()
            except Exception:
                self.pending = None
                self.draining = False
                self.paused = True
                self.candidate_id = 'unverified_after_switch_failure'
                self.before_model_change()
                remove_adapters(self.wrapper.model, self.helper)
                clear_all_history(self.wrapper)
                self.clear_extra_state()
                self._check_base()
                raise
            self.generation += 1
            self.candidate_id = candidate_id
            self.pending = None
            self.draining = False
            self.paused = False
            return dict(status='SHARED_BASE_CANDIDATE_ACCEPTED', at=time.time(),
                request_id=request['request_id'], candidate_id=candidate_id, generation=self.generation,
                optimizer_steps=400 if artifacts else None, base_model_reloaded=False,
                backbone_parameter_objects_and_storage_preserved=True, cleared_all_client_history=True,
                acceptance=acceptance)
