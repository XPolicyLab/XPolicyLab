"""Compose a model service and environment into a traceable bounded skill."""

import uuid

from PhysicalRSI_core.contracts import Contract, Operation


def policy_skill(name, revision, *, environment, model, recorder, max_chunks=4):
    if type(max_chunks) is not int or max_chunks < 1:
        raise ValueError("Positive action-chunk budget required")

    def execute(goal, context):
        recorder.begin_primitive(name)
        success = False
        count = 0
        try:
            for _ in range(max_chunks):
                context.check()
                observation = environment.observe()
                actions = model.predict(uuid.uuid4().hex, observation, {"goal": goal})
                proposal = recorder.add_proposal(str(goal), actions)
                for index, action in enumerate(actions):
                    context.check()  # cancellation boundary before each physical action
                    receipt = environment.step(uuid.uuid4().hex, action)
                    recorder.add_transition(
                        action,
                        receipt["observation"],
                        receipt["reward"],
                        receipt["terminated"],
                        receipt["truncated"],
                        vla_id=proposal,
                        proposal_index=index,
                    )
                    count += 1
                    success = bool(receipt.get("success", False))
                    if receipt["terminated"] or receipt["truncated"]:
                        return {
                            "environment_success": success,
                            "executed_actions": count,
                        }
            return {"environment_success": success, "executed_actions": count}
        finally:
            recorder.end_primitive()
            recorder.finalize()  # failures and partial execution remain evidence

    return Operation(
        name,
        revision,
        Contract("goal"),
        Contract("episode-result"),
        execute,
        frozenset({"environment:" + str(id(environment))}),
    )
