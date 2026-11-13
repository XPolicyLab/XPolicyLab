"""Use the upstream RGB/EEF executor with a persistent Cypher controller."""
from dataclasses import replace
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import deploy as eef_deploy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect import deploy as inspect_deploy
from .client import ContinualClient
from .remote import RemoteContinualClient
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import InfrastructureFailure


class _InfrastructureGuard:
    """Transport outages stop the run instead of becoming a zero-scored episode."""
    def __init__(self, client):
        self.client = client

    def complete(self, messages, tools):
        try:
            return self.client.complete(messages, tools)
        except (OSError, TimeoutError) as error:
            raise InfrastructureFailure('Cypher controller transport failed; inspect private transport logs.') from error

    def finish(self):
        try:
            self.client.finish()
        except (OSError, TimeoutError) as error:
            raise InfrastructureFailure('Cypher episode boundary could not be recorded.') from error


def _policy_factory(*, action_spec, env, task_env):
    local = dict(env)
    local['L4_INSPECT_MODEL'] = local.get('CYPHER_TEMPRE_MODEL', 'gpt-6.1-sol')
    local['L4_INSPECT_DEPTH'] = 'off'
    try:
        client = RemoteContinualClient(local) if local.get('CYPHER_TEMPRE_BROKER_URL') else ContinualClient(local)
    except (OSError, KeyError, ValueError) as error:
        raise InfrastructureFailure('Cypher controller configuration or existing memory is unavailable.') from error
    return eef_deploy.EefAgentPolicy(
        action_spec=replace(action_spec, docs=eef_deploy.eef_docs_from_joint_docs(action_spec.docs)),
        env=local, planner=eef_deploy._planner(task_env),
        client=_InfrastructureGuard(client))


def eval_one_episode(TASK_ENV, model_client):
    # A new observation session does not reinitialize or reset persistent memory.
    clients = []
    def factory(**kwargs):
        policy = _policy_factory(**kwargs)
        clients.append(policy._client)
        return policy
    try:
        return inspect_deploy.eval_one_episode(TASK_ENV, model_client, policy_factory=factory)
    finally:
        for client in clients:
            client.finish()


def eval_one_episode_batch(TASK_ENV, model_client):
    if int(getattr(TASK_ENV, 'num_envs', 1) or 1) != 1:
        raise ValueError('Cypher_Tempre requires a single sequential evaluation stream.')
    return eval_one_episode(TASK_ENV, model_client)
