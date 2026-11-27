"""C05 native FP32 policy, with a small XPolicyLab observation/action boundary."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info, get_robot_state_schema
from .observation_contract import action_chunk, native_observation, validate_schema


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        if model_cfg['action_type'] != 'joint':
            raise ValueError('C05 supports the original dual-arm joint policy only')
        if model_cfg['env_cfg_type'] != 'aloha_agilex':
            raise ValueError('C05 is qualified for RoboTwin aloha_agilex, not arbitrary 14D robots')
        policy_dir = Path(__file__).resolve().parent
        bundle = resolve_checkpoint_root(model_cfg, policy_dir / 'checkpoints', policy_dir=policy_dir)
        contract = json.loads((bundle / 'checkpoint_contract.json').read_text())
        if contract['checkpoint_id'] != '96a254571814d58607b421f407cb2d1cabeae6d86da438b622e2d4eed3b93c30':
            raise ValueError('This adapter requires the C05@5000 release')
        self.schema = get_robot_state_schema('joint', get_robot_action_dim_info(model_cfg['env_cfg_type']))
        validate_schema(self.schema, contract)
        self.execute_steps = int(model_cfg.get('action_chunk', 32))
        if not 1 <= self.execute_steps <= contract['prediction_horizon']:
            raise ValueError('action_chunk exceeds the native prediction horizon')
        self.seed = int(model_cfg.get('seed') or 0)
        self.episode = 0
        self.runtime = bundle / 'runtime'
        self._invocation = tempfile.TemporaryDirectory(prefix='c05-inference-')
        invocation = Path(self._invocation.name)
        # The frozen upstream loader expects a training YAML three levels above hf_ckpt.
        checkpoint = invocation / 'checkpoints' / 'selected' / 'hf_ckpt'
        checkpoint.mkdir(parents=True)
        source = bundle / 'checkpoint'
        for name, size in contract['weights_bytes'].items():
            if (source / name).stat().st_size != size:
                raise ValueError('Weight size mismatch: ' + name)
        for path in source.iterdir():
            if path.is_file():
                (checkpoint / path.name).symlink_to(path.resolve())
        robot_dir = invocation / 'robot_configs'
        robot_dir.mkdir()
        shutil.copyfile(self.runtime / 'parking_robot_configs/robotwin.yaml', robot_dir / 'robotwin.yaml')
        template = (bundle / 'inference_template.json').read_text()
        rendered = template.replace('${RELEASE_RUNTIME}', str(self.runtime)).replace('${INVOCATION}', str(invocation))
        if '${' in rendered:
            raise ValueError('Unresolved inference configuration placeholder')
        # JSON is YAML; preserve numeric/boolean configuration values exactly.
        (invocation / 'lingbotvla_cli.yaml').write_text(rendered)
        os.environ['PARKING_ROOT'] = str(self.runtime)
        os.environ['PARKING_ROBOT_CONFIG_ROOT'] = str(robot_dir)
        os.environ['QWEN3VL_PATH'] = str(self.runtime / 'models/qwen3_vl_tokenizer')
        for directory in (self.runtime / 'lingbot-vla-v2', self.runtime / 'parking_runtime', self.runtime / 'submission_setup'):
            sys.path.insert(0, str(directory))
        from inference_runtime import install
        install()
        from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server, set_seed_everywhere
        from inference_precision import cpu_embedding_policy_class, fp32_loading_policy_class, configure_fp32_compute
        from reliable_protocol import install_kernel
        self._set_seed = set_seed_everywhere
        self.kernel_state, self.verify_kernel = install_kernel()
        policy_class = cpu_embedding_policy_class(fp32_loading_policy_class(LingbotVLAv2Server))
        with working_directory(self.runtime / 'lingbot-vla-v2'):
            self.policy = policy_class(str(checkpoint), use_length=self.execute_steps,
                                       chunk_ret=True, use_bf16=False, use_fp32=True, use_compile=False)
            self.policy.reset('robotwin')
        if self.policy.vla.model.config.num_steps != 10:
            raise ValueError('Unexpected native denoising schedule')
        self.compute_profile = configure_fp32_compute(self.policy)
        self.observations = []
        self._set_seed(self.seed)
        print('[C05] exact FP32 loading verified:', self.policy._competition_loaded_tensor_count,
              'tensors; CPU embedding; fixed-route MoE; action_chunk=', self.execute_steps, flush=True)

    def update_obs(self, obs):
        self.observations = [native_observation(obs, self.schema)]

    def update_obs_batch(self, obs_list):
        self.observations = [native_observation(obs, self.schema) for obs in obs_list]

    def _action(self, obs):
        from inference_precision import verify_fp32_compute
        verify_fp32_compute(self.policy)
        before = self.verify_kernel()
        result = self.policy.infer(obs)
        if self.verify_kernel() <= before:
            raise RuntimeError('Fixed-route MoE was not executed')
        return action_chunk(result, self.schema, self.execute_steps)

    def get_action(self):
        if len(self.observations) != 1:
            raise ValueError('Call update_obs with one observation first')
        return self._action(self.observations[0])

    def get_action_batch(self, env_idx_list=None):
        if not self.observations:
            raise ValueError('Call update_obs_batch first')
        if env_idx_list is not None and len(env_idx_list) != len(self.observations):
            raise ValueError('Active indices must align with the received observation batch')
        # The environment has already selected active observations, in active-index order.
        return [self._action(obs) for obs in self.observations]

    def reset(self):
        self.observations = []
        with working_directory(self.runtime / 'lingbot-vla-v2'):
            self.policy.reset('robotwin')
        self._set_seed(self.seed + self.episode)
        self.episode += 1

    def close(self):
        self.observations = []
        self._invocation.cleanup()
