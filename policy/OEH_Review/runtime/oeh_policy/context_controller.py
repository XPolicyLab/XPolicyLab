"""Base context organization: read state/memory and deliver it, without proactive resets.

Teacher owns transport, calls and budgets; memory owns facts and plan mutations.
This controller owns delivery bookkeeping and presentation, not task truth.
"""
import copy
import json

from . import semmem_context, state_context
from .stmem_events import SpatialEvents
from .context_views import scene_frame_model_view


def _reference_prompt(runtime, prompt):
    from .llm_contract import has_v2_experience, reference_section
    if has_v2_experience(runtime) or runtime.cfg["review"].get("exploration_probe"):
        prompt += "\n" + reference_section("q2-experience") + "\n"
    if runtime.cfg["review"].get("digest") == "event_summary":
        prompt += "\n" + reference_section("q3-interval") + "\n"
    return prompt


def initial_prompt(runtime) -> str:
    """The developer prompt of a runtime's teacher: its benchmark's prompt set, under its schedule."""
    from .teacher import execution_contract_prompt, prompt_text
    from .task_tree import policy_name
    if runtime.cfg.get("mode") != "review-stmem":
        # The declared context manifest (review.context, G3) shapes the prompt; absent, today's prompt.
        prompt = prompt_text(runtime.cfg["review"].get("schedule"), runtime.contract.benchmark,
                             context=runtime.cfg["review"].get("context"))
        return _reference_prompt(runtime, prompt)
    is_tree = policy_name(runtime.cfg.get("semmem")) == "hierarchical-policy"
    input_context = runtime.cfg["review"].get("teacher", {}).get("input_context")
    if input_context and input_context.get("enforce_budget", True) and not runtime.cfg["review"].get("llm_token_budget"):
        from .input_context import PROMPT
        from .llm_contract import interaction_contract
        prompt = PROMPT.replace(interaction_contract(), interaction_contract(task_tree=True)) if is_tree else PROMPT
        return _reference_prompt(runtime, execution_contract_prompt(prompt,runtime))
    prompt = prompt_text(runtime.cfg["review"].get("schedule"), runtime.contract.benchmark)
    if runtime.cfg.get("mode") == "review-stmem":
        from .llm_contract import interaction_contract
        prompt += "\n" + interaction_contract(task_tree=is_tree) + "\n"
    if runtime.cfg.get("semmem") or runtime.cfg.get("stmem", {}).get("context_max_entities"):
        prompt = prompt.replace("current spatial states of all tracked entities", "a bounded subset of current spatial states; other objects are queryable")
    budget=runtime.cfg["review"].get("llm_token_budget") or {}
    if budget.get("scaling"):
        basis=budget["scaling"]
        prompt += (f"\nEpisode token allowance: {budget['limit']} input+output tokens for "
                   f"{basis['native_step_limit']} allowed native steps; reference "
                   f"{basis['base_limit']} tokens / {basis['reference_steps']} steps.\n")
    return _reference_prompt(runtime, execution_contract_prompt(prompt,runtime))


class BaseContextController:
    policy = "base-policy"

    def __init__(self, runtime):
        self.runtime = runtime
        self.spatial_events = SpatialEvents()
        self.replies = []
        self.evidence = None
        self.delivered_revision = None
        self.requires_recovery = False
        self.discard_response = False
        self.thread_id = None
        self.prepared_delivery = None
        self.recovery_notice = None

    def add_observation(self, brain, packet, images, intro=None):
        """Preserve the existing projection; separate routing/images from execution state."""
        memory_mode = getattr(self.runtime, "stmem", None) is not None or self.runtime.cfg.get("semmem")
        if memory_mode:
            visible = state_context.observation_packet(packet)
        elif brain.cfg.get("input_context"):
            from .input_context import packet as project_packet
            visible = project_packet(packet, self.runtime.method.limits["student_max_steps"])
        else:
            visible = packet
        # These opt-in evidence packages must survive bounded/memory projections. Keep legacy
        # packet projections unchanged, and never reconstruct a removed ablation's evidence.
        from .llm_contract import has_v2_experience
        extra = []
        if has_v2_experience(self.runtime):
            extra.append("task_notes")
        if self.runtime.cfg.get("review", {}).get("exploration_probe"):
            extra.append("exploration_probe")
        if self.runtime.cfg.get("review", {}).get("digest") == "event_summary":
            extra.append("interval_digest")
        if extra:
            visible = dict(visible)
            for key in extra:
                if key in packet:
                    visible[key] = copy.deepcopy(packet[key])
        if packet.get("scene_frame"):
            visible = {k: v for k, v in visible.items() if k not in {"images", "scene_frame"}}
            visible["stmem_scene_frame"] = scene_frame_model_view(packet["scene_frame"])
        brain.conversation.add_packet(json.dumps(visible, separators=(",", ":")), images, intro)

    def prepare_turn(self, brain, packet, *, pruned=None):
        """Append notifications, queued replies, independent state and semantic plan once per call.

        Failed calls retain these items in the transport conversation. Only a successful
        create advances the delivered tree revision. This never clears conversation history.
        """
        runtime = self.runtime
        tree = getattr(runtime, "task_tree", None)
        memory_mode = getattr(runtime, "stmem", None) is not None or runtime.cfg.get("semmem")
        full = bool(memory_mode and (self.requires_recovery or pruned or
            (tree and getattr(tree, "restored", False) and self.delivered_revision is None)))
        self.discard_response = False
        self.prepared_delivery = {"full": full, "revision": tree.revision if tree else None}
        if full:
            runtime.context_recovery_pending = True
            self.spatial_events = SpatialEvents()  # A compacted model needs unchanged current entities again.
        notification = None
        if getattr(runtime, "stmem", None) is not None:
            raw = getattr(runtime.stmem, "last_context", None) or packet["spatial_context"]
            notification = self.spatial_events.notify(raw, brain.working_set,
                packet.get("observation_id", packet["request_id"]))
            if notification is not None:
                brain.conversation.add_packet(json.dumps({"stmem_notification": notification}, separators=(",", ":")), [])
                runtime.emit("stmem_notification", notification)
        for response, images in self.replies:
            brain.conversation.add_packet(json.dumps(response), images)
        self.replies = []
        if memory_mode:
            # A restored tree may retain the original public TaskSpec when this packet omits it.
            # The state adapter itself does not depend on SemMem or infer any task rules.
            state_packet = packet
            if packet.get("task_spec") is None and tree is not None:
                state_packet = {**packet, "task_spec": copy.deepcopy(tree.state["task_spec"])}
            package = {"state_context": state_context.model_context(runtime, state_packet)}
            if full:
                package["context_recovery"] = copy.deepcopy(self.recovery_notice) if self.recovery_notice else {
                    "reason": "checkpoint_or_requested_recovery", "previous_response_not_executed": False}
            if runtime.cfg.get("semmem"):
                context = semmem_context.model_context(runtime, packet, full=full,
                    include_partial=tree is not None and self.delivered_revision != tree.revision)
                self.evidence = copy.deepcopy(context["evidence"])
                package["semantic_context"] = context
                runtime.emit("task_tree_context", {"revision": tree.revision if tree else None,
                    "full_tree": full, "partial_tree": "partial_tree" in context})
            brain.conversation.add_packet(json.dumps(package, separators=(",", ":")),
                runtime.review.images if full else [])
        return notification

    def add_feedback(self, brain, value):
        """Present runtime validation/admission feedback in the same conversation."""
        brain.conversation.add_error(json.dumps(value))

    def add_legacy_query_reply(self, brain, packet, result, images, projection):
        """Preserve the legacy spatial-query reply and size bound without an extra call."""
        if projection:
            from .memory_context import spatial_context
            if "spatial_context" in result:
                packet["spatial_context"] = spatial_context(result.pop("spatial_context"),
                    max_entities=projection["context_max_entities"], max_chars=projection["context_max_chars"])
            reply = json.dumps({"memory_result": result, "no_execution": True})
            if len(reply) > projection["context_max_chars"]:
                reply = json.dumps({"memory_result": {"status": "result_too_large",
                    "hint": "Use a smaller snapshot limit and paginate next_offset, or query one entity"}, "no_execution": True})
                images = []
        else:
            reply = json.dumps({"spatial_context": packet["spatial_context"], "memory_result": result, "no_execution": True})
        if images:
            brain.conversation.add_packet(reply, images)
        else:
            brain.conversation.add_error(reply)

    def on_delivery(self, data):
        """A successful transport call acknowledges delivery; context loss gates its action."""
        tree = getattr(self.runtime, "task_tree", None)
        memory_mode = getattr(self.runtime, "stmem", None) is not None or self.runtime.cfg.get("semmem")
        if not memory_mode:
            return
        self.requires_recovery = False
        self.runtime.context_recovery_pending = False
        self.recovery_notice = None
        if self.prepared_delivery:
            self.delivered_revision = self.prepared_delivery["revision"]
            if self.prepared_delivery["full"]:
                self.runtime.emit("context_recovery_delivered", dict(self.prepared_delivery))
        codex = data.get("codex") or {}
        thread_id = codex.get("thread_id")
        lost = any(item.get("event") == "thread/compacted" or item.get("item") == "contextCompaction"
                   for item in codex.get("native_items", []) if isinstance(item, dict))
        lost = lost or bool(self.thread_id and thread_id and self.thread_id != thread_id)
        if thread_id:
            self.thread_id = thread_id
        if lost:
            self.requires_recovery = self.discard_response = True
            self.runtime.context_recovery_pending = True
            self.recovery_notice = {"reason": "provider_context_changed", "previous_response_not_executed": True}
            event = {"revision": tree.revision if tree else None, "reason": "provider_context_changed"}
            self.runtime.emit("context_recovery_required", event)
            if tree:
                self.runtime.emit("task_tree_recovery_required", event)
