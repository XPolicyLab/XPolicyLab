"""Primary-LLM task plans: durable tree, sequential queues, no perception or model calls."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path


SCHEMA = "oeh.semmem.task-tree.v3"
STATUSES = {"not_started", "in_progress", "achieved"}


def policy_name(value):
    name = (value or {}).get("policy") if isinstance(value, dict) else None
    return "hierarchical-policy" if name == "task-tree" else name


def validate_config(cfg):
    given = cfg["semmem"]
    name = policy_name(given)
    allowed = {"policy", "max_depth"} if name == "hierarchical-policy" else {"policy"}
    if (not isinstance(given, dict) or name not in {"base-policy", "hierarchical-policy"}
            or cfg.get("mode") != "review-stmem" or set(given) - allowed):
        raise ValueError("SemMem supports base-policy or hierarchical-policy in review-stmem")
    if name == "base-policy":
        return {"policy": name}
    depth = given.get("max_depth", 5)
    if type(depth) is not int or depth < 1:
        raise ValueError("hierarchical max_depth must be a positive benchmark hyperparameter")
    return {"policy": name, "max_depth": depth}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {name} required")
    return value


def _strings(value, name, *, nonempty=False):
    if not isinstance(value, list) or nonempty and not value:
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}list")
    for item in value:
        _text(item, name)
    return copy.deepcopy(value)


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError("invalid fields; required: " + ", ".join(required))


class TaskTree:
    """Single runtime writer; atomic checkpoints include request deduplication and attempts."""

    def __init__(self, directory, *, episode_id, instruction, task_spec=None, max_depth=5):
        self.path = Path(directory) / "task-tree.json"
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("invalid max_depth")
        _text(instruction, "instruction")
        self.state = {"schema": SCHEMA, "episode_id": episode_id, "instruction": instruction,
                      "task_spec": copy.deepcopy(task_spec), "max_depth": max_depth, "revision": 0,
                      "root_id": "task-0", "next_id": 1, "nodes": {},
                      "queues": [{"parent_id": None, "depth": 0, "node_ids": ["task-0"], "plan_node_ids": ["task-0"]}],
                      "updates": {}, "attempts": {}, "last_departure_time": None}
        self.state["nodes"]["task-0"] = self._task("task-0", None, 0, 1, {
            "subgoal": instruction, "is_leaf": None, "achievement_condition": None,
            "required_evidence": [], "spec_refs": []})
        self.state["nodes"]["task-0"]["status"] = "in_progress"
        self.state["nodes"]["task-0"]["source"] = {"kind": "instruction", "episode_id": episode_id}
        self.restored = self.path.exists()
        if self.restored:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for key in ("schema", "episode_id", "instruction", "task_spec", "max_depth"):
                if saved.get(key) != self.state[key]:
                    raise ValueError("task-tree checkpoint identity mismatch: " + key)
            self.state = saved
            self._check()
        else:
            self._commit(self.state)

    @staticmethod
    def _task(nid, parent, depth, ordinal, spec):
        _fields(spec, ("subgoal", "is_leaf", "achievement_condition", "required_evidence", "spec_refs"))
        _text(spec["subgoal"], "subgoal")
        if spec["is_leaf"] is not None and type(spec["is_leaf"]) is not bool:
            raise ValueError("is_leaf must be boolean or null")
        if spec["achievement_condition"] is not None:
            _text(spec["achievement_condition"], "achievement_condition")
        _strings(spec["required_evidence"], "required_evidence")
        _strings(spec["spec_refs"], "spec_refs")
        if spec["is_leaf"] is True and not spec["achievement_condition"]:
            raise ValueError("an executable leaf needs an achievement condition")
        return {"node_id": nid, "kind": "task", "parent_id": parent, "depth": depth,
                "ordinal": ordinal, **copy.deepcopy(spec), "status": "not_started", "children": [], "child_count": 0, "attempt_ids": []}

    @property
    def revision(self):
        return self.state["revision"]

    @property
    def current_id(self):
        if self.state["queues"]:
            return self.state["queues"][-1]["node_ids"][0]
        return None

    def context(self, *, full=False, include_partial=True):
        """Ancestors as singletons, current-level siblings complete; queues only on recovery."""
        nodes = self.state["nodes"]
        current = self.current_id
        # Finished tasks have no live node; keep the original instruction visible.
        chain, nid = [], current or self.state["root_id"]
        while nid is not None:
            chain.append(nid)
            nid = nodes[nid]["parent_id"]
        chain.reverse()
        rows = [[nid] for nid in chain[:-1]]
        parent = nodes[chain[-1]]["parent_id"]
        rows.append(list(nodes[parent]["children"]) if parent else [chain[-1]])
        positions = {nid: i + 1 for queue in self.state["queues"] for i, nid in enumerate(queue["node_ids"])}
        def project(nid):
            fields = ("node_id", "kind", "parent_id", "depth", "ordinal", "child_count", "subgoal",
                      "is_leaf", "achievement_condition", "required_evidence", "spec_refs", "status",
                      "failed_node_id", "history", "reason")
            node = {k: copy.deepcopy(nodes[nid][k]) for k in fields if k in nodes[nid]}
            node["queue_position"] = positions.get(nid)
            node["is_current"] = nid == current
            return node
        result = {"schema": SCHEMA, "episode_id": self.state["episode_id"], "revision": self.revision,
                  "root_id": self.state["root_id"], "max_depth": self.state["max_depth"],
                  "current_node": nodes[current]["subgoal"] if current else None,
                  "current_node_id": current}
        result["current_task"] = project(current) if current else None
        result["last_departure_time"] = self.state["last_departure_time"]
        if include_partial or full:
            result["partial_tree"] = [[project(nid) for nid in row] for row in rows]
        if full:
            result["TaskTree"] = self.model_tree_rows()
            result["TaskQueues"] = self.model_queue_rows()
        return result

    def model_node(self, nid):
        """Public planning view. Sources and per-action audits remain invisible in checkpoints."""
        raw = self.state["nodes"][nid]
        fields = ("node_id", "kind", "parent_id", "depth", "ordinal", "child_count", "children",
                  "subgoal", "is_leaf", "achievement_condition", "required_evidence", "spec_refs",
                  "status", "failed_node_id", "history", "reason")
        node = {k: copy.deepcopy(raw[k]) for k in fields if k in raw}
        for key, allowed in {
            "completion": ("judge", "verified", "evidence_refs", "reason", "required_children", "evidence", "failed_replan_ids"),
            "departure": ("outcome", "evidence_provided_at", "evidence_refs", "evidence_valid")}.items():
            if key in raw:
                node[key] = {k: copy.deepcopy(raw[key][k]) for k in allowed if k in raw[key]}
        return node

    def model_tree_rows(self):
        rows, pending = [], [self.state["root_id"]]
        while pending:
            rows.append([self.model_node(nid) for nid in pending])
            pending = [child for nid in pending for child in self.state["nodes"][nid]["children"]]
        return rows

    def model_queue_rows(self):
        return [[self.model_node(nid) for nid in q["node_ids"]] for q in self.state["queues"]]

    def tree_rows(self):
        rows, pending = [], [self.state["root_id"]]
        while pending:
            rows.append([copy.deepcopy(self.state["nodes"][nid]) for nid in pending])
            pending = [child for nid in pending for child in self.state["nodes"][nid]["children"]]
        return rows

    def queue_rows(self):
        return [[copy.deepcopy(self.state["nodes"][nid]) for nid in queue["node_ids"]]
                for queue in self.state["queues"]]

    def _commit(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pending = self.path.with_suffix(".tmp")
        try:
            with pending.open("w", encoding="utf-8") as stream:
                json.dump(state, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            pending.replace(self.path)
        finally:
            pending.unlink(missing_ok=True)
        self.state = state

    def _id(self):
        nid = "task-" + str(self.state["next_id"])
        self.state["next_id"] += 1
        return nid

    def _expand(self, children, source):
        parent = self.state["nodes"][self.current_id]
        if not isinstance(children, list) or not children:
            raise ValueError("decompose requires nonempty children")
        depth = parent["depth"] + 1
        if depth > self.state["max_depth"]:
            raise ValueError("max_depth exceeded")
        ids = []
        for spec in children:
            nid = self._id()
            node = self._task(nid, parent["node_id"], depth, len(parent["children"]) + 1, spec)
            node["source"] = copy.deepcopy(source)
            self.state["nodes"][nid] = node
            parent["children"].append(nid)
            parent["child_count"] = len(parent["children"])
            ids.append(nid)
        parent["is_leaf"] = False
        self.state["nodes"][ids[0]]["status"] = "in_progress"
        self.state["queues"].append({"parent_id": parent["node_id"], "depth": depth, "node_ids": ids, "plan_node_ids": list(ids)})

    def _depart(self, source, outcome, evidence_refs=()):
        node = self.state["nodes"][self.current_id]
        if source.get("evidence_owner", self.current_id) != self.current_id:
            raise ValueError("evidence belongs to the previous leaf; observe the new current leaf first")
        timestamp = source.get("evidence_time", source.get("sim_time_s"))
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("finite framework evidence delivery time required")
        last = self.state["last_departure_time"]
        if last is not None and timestamp <= last:
            raise ValueError("leaf evidence time conflict: departures must strictly increase")
        refs = _strings(list(evidence_refs), "evidence_refs")
        if len(refs) > 1:
            raise ValueError("only one current evidence reference may be cited")
        if refs and "available_evidence_refs" in source and refs[0] not in source["available_evidence_refs"]:
            raise ValueError("evidence reference is not in the supplied STMem evidence slot")
        node["departure"] = {"outcome": outcome, "evidence_provided_at": timestamp,
                             "evidence_refs": refs, "evidence_valid": False, "source": copy.deepcopy(source)}
        self.state["last_departure_time"] = timestamp

    def _complete(self, operation, source):
        node = self.state["nodes"][self.current_id]
        if node["is_leaf"] is not True:
            raise ValueError("only a prepared current leaf may be completed")
        evidence = _strings(operation["evidence_refs"], "evidence_refs")
        self._depart(source, "achieved", evidence)
        reason = _text(operation["reason"], "completion reason")
        record = {"judge": "primary_llm", "verified": False, "evidence_refs": evidence,
                  "reason": reason, "source": copy.deepcopy(source)}
        node["status"], node["completion"] = "achieved", record
        while self.state["queues"]:
            queue = self.state["queues"][-1]
            queue["node_ids"].pop(0)
            if queue["node_ids"]:
                self.state["nodes"][queue["node_ids"][0]]["status"] = "in_progress"
                break
            self.state["queues"].pop()
            if queue["parent_id"] is None:
                break  # The instruction itself has left depth-0's queue.
            parent = self.state["nodes"][queue["parent_id"]]
            parent["status"] = "achieved"
            parent["completion"] = {"judge": "structural_child_completion", "verified": False,
                                     "source": copy.deepcopy(source), "required_children": list(queue["plan_node_ids"]),
                                     "evidence": {"kind": "all_required_children_achieved", "last_leaf_time": self.state["last_departure_time"]},
                                     "failed_replan_ids": list(dict.fromkeys(
                                         [nid for nid in parent["children"] if self.state["nodes"][nid]["kind"] == "replan"] +
                                         [rid for nid in queue["plan_node_ids"] for rid in self.state["nodes"][nid].get("completion", {}).get("failed_replan_ids", [])]))}

    def _replan(self, operation, source):
        levels, histories = operation["levels"], operation["histories"]
        if type(levels) is not int or not 1 <= levels < len(self.state["queues"]):
            raise ValueError("replan levels must select existing bottom queues and preserve depth-0 instruction")
        if not isinstance(histories, list) or len(histories) != levels:
            raise ValueError("one history per destroyed queue, deepest first, required")
        departing_leaf = self.current_id
        self._depart(source, "failed_replan")
        for i, history in enumerate(histories):
            _fields(history, ("history", "reason"))
            _text(history["history"], "subtree history")
            if i == 0 or history["reason"] is not None:
                _text(history["reason"], "replan reason")
            queue = self.state["queues"].pop()
            parent = self.state["nodes"][queue["parent_id"]]
            nid = self._id()
            failed = self.state["nodes"][queue["node_ids"][0]]
            self.state["nodes"][nid] = {"node_id": nid, "kind": "replan", "parent_id": parent["node_id"],
                "depth": queue["depth"], "ordinal": len(parent["children"]) + 1,
                "subgoal": failed["subgoal"], "status": "failed_replan", "failed_node_id": failed["node_id"],
                "children": [], "child_count": 0, "history": history["history"], "reason": history["reason"],
                "evidence": {"kind": "leaf_failed_replan" if i == 0 else "subtree_failed_replan",
                             "leaf_node_id": departing_leaf, "provided_at": self.state["last_departure_time"]},
                "source": copy.deepcopy(source), "removed_queue": copy.deepcopy(queue["node_ids"])}
            parent["children"].append(nid)
            parent["child_count"] = len(parent["children"])
            failed["failure"] = {"replan_node_id": nid, "leaf_node_id": departing_leaf,
                                 "evidence_provided_at": self.state["last_departure_time"]}
        self.state["nodes"][self.current_id]["is_leaf"] = None
        if operation["children"]:
            self._expand(operation["children"], source)
        elif not isinstance(operation["children"], list):
            raise ValueError("replacement children must be a list; [] clears without completing")

    def update(self, value, *, source):
        _fields(value, ("update_id", "expected_revision", "operations"))
        _text(value["update_id"], "update_id")
        fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
        cached = self.state["updates"].get(value["update_id"])
        if cached is not None:
            if cached["fingerprint"] != fingerprint:
                raise ValueError("update_id reused with different content")
            return copy.deepcopy(cached["result"])
        if type(value["expected_revision"]) is not int or value["expected_revision"] != self.revision:
            raise ValueError("stale task-tree revision")
        operations = value["operations"]
        if not isinstance(operations, list) or not operations:
            raise ValueError("nonempty operations required")
        # Local validation/atomic mutation only; no rollout of future branches.
        draft = copy.copy(self)
        draft.state = copy.deepcopy(self.state)
        completes = 0
        source = copy.deepcopy(source)
        source.setdefault("evidence_owner", self.current_id)
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError("task operation must be an object")
            kind = op.get("operation")
            if draft.current_id is None:
                raise ValueError("task already reported achieved")
            _text(op.get("node_id"), "node_id")
            if op["node_id"] not in ("current", draft.current_id):
                raise ValueError("only the current queue head may be changed")
            if kind == "replan":
                _fields(op, ("operation", "node_id", "levels", "histories", "children"))
                draft._replan(op, source)
            elif kind == "decompose":
                _fields(op, ("operation", "node_id", "children"))
                draft._expand(op["children"], source)
            elif kind == "prepare":
                _fields(op, ("operation", "node_id", "achievement_condition", "required_evidence"))
                node = draft.state["nodes"][draft.current_id]
                node["achievement_condition"] = _text(op["achievement_condition"], "achievement_condition")
                node["required_evidence"] = _strings(op["required_evidence"], "required_evidence")
                node["is_leaf"] = True
                node["preparation_source"] = copy.deepcopy(source)
            elif kind == "complete":
                _fields(op, ("operation", "node_id", "reason", "evidence_refs"))
                completes += 1
                if completes > 1:
                    raise ValueError("only one explicit completion per response")
                draft._complete(op, source)
            else:
                raise ValueError("unknown task operation")
        draft.state["revision"] += 1
        draft._check()
        result = {"status": "stored", "update_id": value["update_id"], "revision": draft.revision,
                  "current_node_id": draft.current_id}
        draft.state["updates"][value["update_id"]] = {"fingerprint": fingerprint, "result": copy.deepcopy(result)}
        self._commit(draft.state)
        return result

    def query(self, operation, node_id=None):
        if operation in {"full_tree", "all_queues", "queue"}:
            if node_id not in (None, self.current_id):
                raise ValueError("global/current queue query uses null or current node_id")
            result = {"schema": SCHEMA, "revision": self.revision, "operation": operation,
                      "current_node_id": self.current_id}
            if operation == "full_tree":
                result["TaskTree"] = self.model_tree_rows()
            elif operation == "all_queues": result["TaskQueues"] = self.model_queue_rows()
            else: result["TaskQueue"] = self.model_queue_rows()[-1] if self.state["queues"] else []
            return result
        if node_id not in self.state["nodes"]:
            raise ValueError("unknown task node_id")
        nodes = self.state["nodes"]
        if operation == "children":
            ids = list(nodes[node_id].get("children", []))
        elif operation == "subtree":
            ids, pending = [], [node_id]
            while pending:
                nid = pending.pop()
                ids.append(nid)
                pending.extend(reversed(nodes[nid].get("children", [])))
        elif operation in ("chain", "chain-span"):
            chain, nid = [], node_id
            while nid is not None:
                chain.append(nid)
                nid = nodes[nid]["parent_id"]
            ids = []
            for nid in reversed(chain):
                parent = nodes[nid]["parent_id"]
                ids.extend(nodes[parent]["children"] if operation == "chain-span" and parent else [nid])
            ids = list(dict.fromkeys(ids))
        else:
            raise ValueError("query must be subtree, chain, chain-span or children")
        return {"schema": SCHEMA, "revision": self.revision, "operation": operation, "node_id": node_id,
                "nodes": [self.model_node(nid) for nid in ids],
                "current_node_id": self.current_id}

    def check_binding(self, binding):
        _fields(binding, ("node_id", "revision"))
        if type(binding["revision"]) is not int or binding["revision"] != self.revision or binding["node_id"] != self.current_id:
            raise ValueError("action binding does not match the final task-tree revision/current leaf")
        node = self.state["nodes"].get(self.current_id)
        if not node or node["is_leaf"] is not True or not node["achievement_condition"]:
            raise ValueError("prepare or decompose the current task before motion")

    def record_attempt(self, attempt_id, binding, *, source, action):
        self.check_binding(binding)
        if attempt_id in self.state["attempts"]:
            raise ValueError("task action attempt already submitted; inspect its actual result")
        state = copy.deepcopy(self.state)
        state["nodes"][binding["node_id"]]["attempt_ids"].append(attempt_id)
        state["attempts"][attempt_id] = {"binding": copy.deepcopy(binding), "source": copy.deepcopy(source),
            "action": copy.deepcopy(action), "status": "result_unknown"}
        self._commit(state)

    def record_result(self, attempt_id, result):
        state = copy.deepcopy(self.state)
        state["attempts"][attempt_id]["result"] = copy.deepcopy(result)
        state["attempts"][attempt_id]["status"] = "result_recorded"
        self._commit(state)

    def _check(self):
        state, nodes = self.state, self.state["nodes"]
        if type(state["revision"]) is not int or state["revision"] < 0:
            raise ValueError("invalid checkpoint revision")
        seen, pending = set(), [(state["root_id"], None, 0)]
        while pending:
            nid, parent, depth = pending.pop()
            if nid in seen or nid not in nodes:
                raise ValueError("invalid task-tree edges")
            seen.add(nid)
            node = nodes[nid]
            if node["node_id"] != nid or node["parent_id"] != parent or node["depth"] != depth or depth > state["max_depth"]:
                raise ValueError("invalid task-tree hierarchy")
            if node.get("child_count") != len(node.get("children", [])):
                raise ValueError("invalid child_count")
            if node["kind"] == "task":
                if node["status"] not in STATUSES:
                    raise ValueError("invalid task status")
                for i, child in enumerate(node["children"]):
                    if nodes[child]["ordinal"] != i + 1:
                        raise ValueError("invalid task-tree order")
                    pending.append((child, nid, depth + 1))
            elif (node["kind"] != "replan" or not node.get("subgoal")
                  or node.get("status") != "failed_replan" or node["child_count"]):
                raise ValueError("invalid failed replan record")
        if seen != set(nodes):
            raise ValueError("unreachable task nodes")
        root_id = state["root_id"]
        root_achieved = nodes[root_id]["status"] == "achieved"
        if not state["queues"] and not root_achieved:
            raise ValueError("unfinished instruction must remain in depth-0 queue")
        if root_achieved and state["queues"]:
            raise ValueError("achieved root cannot have active queues")
        parent = None
        for depth, queue in enumerate(state["queues"]):
            plan = queue.get("plan_node_ids")
            if not isinstance(plan, list) or not plan or len(plan) != len(set(plan)):
                raise ValueError("invalid queue plan")
            if any(nid not in nodes or nodes[nid]["kind"] != "task" or nodes[nid]["parent_id"] != parent for nid in plan):
                raise ValueError("invalid queue plan members")
            if queue["node_ids"] != plan[-len(queue["node_ids"]):]:
                raise ValueError("remaining queue must be a plan suffix")
            if any(nodes[nid]["status"] != "achieved" for nid in plan[:len(plan)-len(queue["node_ids"])]):
                raise ValueError("queue prefix must have achieved tasks")
            if queue["parent_id"] != parent or queue["depth"] != depth or not queue["node_ids"]:
                raise ValueError("invalid active TaskQueues path")
            if depth == 0 and queue["node_ids"] != [root_id]:
                raise ValueError("depth-0 queue must contain only the instruction")
            for i, nid in enumerate(queue["node_ids"]):
                node = nodes[nid]
                if node["parent_id"] != parent or node["kind"] != "task" or node["status"] != ("not_started" if i else "in_progress"):
                    raise ValueError("invalid queue task")
            parent = queue["node_ids"][0]
        if type(state["next_id"]) is not int or state["next_id"] <= max(int(n.split("-")[-1]) for n in nodes):
            raise ValueError("invalid task ID sequence")
