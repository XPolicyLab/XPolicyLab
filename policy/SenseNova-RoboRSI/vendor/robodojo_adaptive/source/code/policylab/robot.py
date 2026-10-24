"""Agent-visible client. No simulator code or test metadata belongs here."""
import json
import os
import socket


def call(request):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1200)
        connection.connect(os.environ.get("POLICYLAB_SOCKET", "/bridge/robot.sock"))
        connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
        with connection.makefile("rb") as stream:
            line = stream.readline(16 * 1024 * 1024)
    if not line:
        raise RuntimeError("No response; do not blindly repeat a possibly executed action")
    result = json.loads(line)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def observe():
    return call({"op": "observe"})


def act(action, repeat=1):
    return call({"op": "act", "action": action, "repeat": repeat})


def review(analysis):
    """Submit an evidence review at a checkpoint or before stopping early."""
    return call({"op": "review", "analysis": analysis})


def reset():
    """Restart evaluation only when explicitly enabled; unavailable during exploration."""
    return call({"op": "reset"})


def ready():
    return call({"op": "ready"})


def stop(reason):
    """End evaluation after a review; only the native evaluator determines success."""
    return call({"op": "stop", "reason": reason})


if __name__ == "__main__":
    import sys
    print(json.dumps(call(json.loads(sys.argv[1])), indent=2))


def vla_propose(subgoal):
    """Return the configured number of EEF candidates without moving the robot."""
    return call({"op": "vla_propose", "subgoal": subgoal})


def vla_review(proposal_id, decision, reason, assessment, targets=None, approved_steps=5):
    """Review 2-50 targets: accept, modify or reject; omitted length defaults to 5."""
    return call({"op": "vla_review", "proposal_id": proposal_id,
                 "decision": decision, "reason": reason, "assessment": assessment,
                 "targets": targets, "approved_steps": approved_steps})


def vla_review_and_propose(proposal_id, decision, reason, assessment, next_subgoal, targets=None, approved_steps=5):
    """Batch the SAME reviewed 2-50 targets and next unmoved proposal; default 5.

    No motion can skip GPT review. The next proposal uses updated measured state
    and cameras, and still requires a NEW GPT accept/modify/reject decision.
    """
    transition = vla_review(proposal_id, decision, reason, assessment, targets, approved_steps)
    if transition.get('done'):
        return {'transition': transition, 'proposal': None}
    return {'transition': transition, 'proposal': vla_propose(next_subgoal)}
