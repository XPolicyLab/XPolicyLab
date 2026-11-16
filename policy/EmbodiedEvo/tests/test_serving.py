"""Real WebSocket routing, independent layouts and bounded agent lifecycle."""

import asyncio
import hashlib
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from client_server.ws.model_server import PolicyServerConfig
from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
from client_server.ws.protocol.exceptions import ServerRestartedError, WsError
from client_server.ws.protocol.codec import decode_envelope
from client_server.ws.protocol.messages import MessageType
from XPolicyLab.policy.EmbodiedEvo.runtime.serving import TaskRoutingServer, close_models, prepare_models, selected_tasks, server_settings
from XPolicyLab.policy.EmbodiedEvo.runtime.workspace import POLICY_DIR
from XPolicyLab.utils.process_data import encode_image_bit


def observation(value, *, encoded=False):
    rgb = np.full((16, 16, 3), [value % 255, 20, 30], dtype=np.uint8)
    return {"instruction": f"task marker {value}",
            "vision": {"cam_head": {"color": encode_image_bit(rgb) if encoded else rgb}},
            "state": {"arm_joint_state": np.full(6, float(value)), "ee_joint_state": np.zeros(1),
                      "ee_pose": np.array([float(value), 0, 0, 1, 0, 0, 0])}}


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
        cfg.update(diagnostic=True, eval_batch=True, server_tasks=["stack_bowls", "push_T"])
        with patch.dict(os.environ, EMBODIEDEVO_RUN_DIR=str(self.root / "server")), \
             patch("XPolicyLab.policy.EmbodiedEvo.model.get_robot_action_dim_info", return_value={"arm_dim": [6], "ee_dim": [1]}):
            self.models, run = prepare_models(cfg)
        self.server = TaskRoutingServer(self.models, PolicyServerConfig(host="127.0.0.1", port=0),
                                        run_dir=run, disconnect_grace_s=1, poll_interval_s=0.01)
        await self.server.start()
        self.clients = []

    async def asyncTearDown(self):
        await asyncio.gather(*(client.close() for client in self.clients), return_exceptions=True)
        await self.server.stop()
        self.assertFalse(self.server.sessions)
        self.assertFalse(self.server.capacity.reserved)
        self.assertFalse(self.server.scheduler.entries)
        close_models(self.models)

    async def client(self, evaluation):
        client = PolicyEvalClient(PolicyEvalClientConfig(url=self.server.url, evaluation_id=evaluation,
            request_timeout_s=40, max_connect_attempts=1, proxy=None))
        self.clients.append(client)
        await client.connect()
        return client

    def model(self, client, case="stack_bowls_case"):
        return self.server.sessions[(client.config.evaluation_id, case)].model

    async def begin(self, client, case="stack_bowls_case", *, layout=0, environments=None):
        await client.reset(trial_id="trial", action_case_id=case)
        await client.prepare_case(case, case_meta={"native_step_limit": 2, "diagnostic_environment": True,
            "environments": environments or [{"env_idx": 0, "layout_id": layout}]})

    async def act(self, client, case, obs):
        await client.call("update_obs", obs, action_case_id=case)
        async with asyncio.timeout(25):
            while True:
                response = await client.call("get_action", action_case_id=case)
                if response.payload["result"]:
                    return response.payload["result"][0]

    async def finish(self, client, case, obs):
        await client.call("update_obs", obs, action_case_id=case)
        await client.trial_end(trial_id="trial", action_case_id=case,
                               result={"success": False, "terminated": False, "truncated": True})

    async def test_same_task_distinct_layouts_have_identical_initial_knowledge_and_independent_workers(self):
        first, second = await self.client("first"), await self.client("second")
        case = "stack_bowls_case"
        self.assertFalse(self.server.sessions)
        await asyncio.gather(self.begin(first, layout=7), self.begin(second, layout=19))
        models = [self.model(first), self.model(second)]
        self.assertNotEqual(models[0].run, models[1].run)
        for model in models:
            self.assertIsNone(model.contexts[0].worker)
            self.assertFalse((model.run / "episodes").exists())
        markers = [observation(7), observation(19, encoded=True)]
        for key in ("arm_joint_state", "ee_pose"):
            actions = await asyncio.gather(*(self.act(c, case, obs) for c, obs in zip([first, second], markers)))
            self.assertEqual([a[key][0] for a in actions], [7, 19])
        workers = [model.contexts[0].worker for model in models]
        self.assertNotEqual(workers[0].pid, workers[1].pid)
        workspaces = [m.run / "episodes/episode_0000/workspace" for m in models]
        for component in ("skills", "lessons", "instruction.md", "primitives", "scripts"):
            def hashes(workspace):
                base = workspace / component
                files = [base] if base.is_file() else sorted(base.rglob("*"))
                return {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files if p.is_file()}
            self.assertEqual(hashes(workspaces[0]), hashes(workspaces[1]))
        (workspaces[0] / "scratch/private.txt").write_text("first only")
        self.assertFalse((workspaces[1] / "scratch/private.txt").exists())
        for model, layout in zip(models, (7, 19)):
            self.assertEqual(json.loads((model.run / "episode-map.jsonl").read_text())["layout_id"], layout)
        await asyncio.gather(*(self.finish(c, case, obs) for c, obs in zip([first, second], markers)))
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(self.server.capacity.total, 0)
        # Another layout in the SAME client starts fresh Python globals/workspace.
        await self.begin(first, layout=31)
        action = await self.act(first, case, observation(31))
        self.assertEqual(action["arm_joint_state"][0], 31)
        self.assertTrue((models[0].run / "episodes/episode_0001/workspace").is_dir())

    async def test_same_layout_in_different_evaluations_is_also_isolated(self):
        first, second = await self.client("a"), await self.client("b")
        await asyncio.gather(self.begin(first, layout=5), self.begin(second, layout=5))
        self.assertIsNot(self.model(first).contexts[0], self.model(second).contexts[0])
        self.assertEqual(self.server.capacity.total, 2)

    async def test_unknown_and_conflicting_task_metadata_cannot_touch_a_session(self):
        client = await self.client("metadata")
        await self.begin(client)
        context = self.model(client).contexts[0]
        with self.assertRaisesRegex(WsError, "not served"):
            await client.reset(trial_id="trial", action_case_id="missing_case")
        with self.assertRaisesRegex(WsError, "conflicts"):
            await client.request(MessageType.RESET, {"task_name": "push_T"}, action_case_id="stack_bowls_case")
        self.assertIs(self.model(client).contexts[0], context)

    async def test_explicit_task_binding_random_variant_and_legacy_no_layout(self):
        client = await self.client("explicit")
        await client.request(MessageType.RESET, {"task_name": "push_T"}, action_case_id="custom_case")
        await client.prepare_case("custom_case", case_meta={"native_step_limit": 2, "diagnostic_environment": True})
        action = await self.act(client, "custom_case", observation(12))
        self.assertEqual(action["arm_joint_state"][0], 12)
        await self.begin(client, "stack_bowls_random_case")
        self.assertEqual(self.model(client, "stack_bowls_random_case").cfg["task_name"], "stack_bowls")

    async def test_duplicate_reset_replay_and_message_ids_are_scoped_to_case(self):
        first, second = await self.client("first"), await self.client("second")
        await first.request(MessageType.RESET, {}, action_case_id="stack_bowls_case", _request_id="same-id")
        await first.prepare_case("stack_bowls_case", case_meta={"native_step_limit": 2, "diagnostic_environment": True})
        context = self.model(first).contexts[0]
        await first.request(MessageType.RESET, {}, action_case_id="stack_bowls_case", _request_id="same-id")
        self.assertIs(self.model(first).contexts[0], context)
        await second.request(MessageType.RESET, {}, action_case_id="stack_bowls_case", _request_id="same-id")
        self.assertIsNot(self.model(first), self.model(second))

    async def test_close_releases_only_that_layout_client(self):
        first, second = await self.client("layout-a"), await self.client("layout-b")
        await asyncio.gather(self.begin(first), self.begin(second))
        await asyncio.gather(self.act(first, "stack_bowls_case", observation(7)), self.act(second, "stack_bowls_case", observation(19)))
        workers = [self.model(c).contexts[0].worker for c in [first, second]]
        await first.close()
        async with asyncio.timeout(10):
            while ("layout-a", "stack_bowls_case") in self.server.sessions:
                await asyncio.sleep(0.01)
        self.assertFalse(workers[0].is_alive())
        self.assertTrue(workers[1].is_alive())
        self.assertEqual(self.server.capacity.total, 1)

    async def test_reconnect_preserves_worker_and_fences_old_peer(self):
        old = await self.client("resume")
        await self.begin(old)
        await self.act(old, "stack_bowls_case", observation(7))
        worker = self.model(old).contexts[0].worker
        new = await self.client("resume")
        await self.act(new, "stack_bowls_case", observation(7))
        with self.assertRaisesRegex(WsError, "Stale connection"):
            await old.call("get_action", action_case_id="stack_bowls_case")
        await old.close()
        self.assertIs(self.model(new).contexts[0].worker, worker)
        self.assertTrue(worker.is_alive())
        await self.finish(new, "stack_bowls_case", observation(7))

    async def test_disconnect_expiry_cleans_resources_and_rejects_resume(self):
        self.server.disconnect_grace_s = 0.08
        client = await self.client("expired")
        await self.begin(client)
        await self.act(client, "stack_bowls_case", observation(7))
        worker = self.model(client).contexts[0].worker
        await client._reset_connection_state()
        async with asyncio.timeout(10):
            while ("expired", "stack_bowls_case") in self.server.sessions:
                await asyncio.sleep(0.02)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.server.capacity.total, 0)
        await client.connect()
        with self.assertRaisesRegex(WsError, "Session expired"):
            await client.call("get_action", action_case_id="stack_bowls_case")

    async def test_invalid_prepare_does_not_leak_capacity_or_partial_contexts(self):
        client = await self.client("invalid")
        with self.assertRaisesRegex(WsError, "positive action limit"):
            await client.prepare_case("stack_bowls_case", case_meta={"native_step_limit": 0, "diagnostic_environment": True})
        self.assertFalse(self.model(client).contexts)
        self.assertEqual(self.server.capacity.total, 0)
        await self.begin(client)

    async def test_capacity_is_atomic_across_layouts_and_released_on_reset(self):
        self.server.capacity.limit = 1
        clients = [await self.client("capacity-a"), await self.client("capacity-b")]
        results = await asyncio.gather(*(self.begin(c, layout=i) for i, c in enumerate(clients)), return_exceptions=True)
        failures = [i for i, result in enumerate(results) if isinstance(result, WsError)]
        self.assertEqual(len(failures), 1)
        self.assertIn("capacity exhausted", str(results[failures[0]]))
        self.assertEqual(self.server.capacity.total, 1)
        await clients[1-failures[0]].reset(trial_id="trial", action_case_id="stack_bowls_case")
        await self.begin(clients[failures[0]])
        self.assertEqual(self.server.capacity.total, 1)

    async def test_partial_batch_completion_keeps_other_layout_alive(self):
        client = await self.client("batch")
        await self.begin(client, environments=[{"env_idx": 2, "layout_id": 12}, {"env_idx": 5, "layout_id": 15}])
        model = self.model(client)
        observations = [{**observation(v, encoded=True), "env_idx": i} for i, v in [(2, 7), (5, 19)]]
        for key in ("arm_joint_state", "ee_pose"):
            await client.call("update_obs_batch", observations, action_case_id="stack_bowls_case")
            async with asyncio.timeout(25):
                while True:
                    response = await client.call("get_action_batch", [2, 5], action_case_id="stack_bowls_case")
                    chunks = response.payload["result"]
                    if all(chunks): break
            self.assertEqual([chunk[0][key][0] for chunk in chunks], [7, 19])
        workers = {i: c.worker for i, c in model.contexts.items()}
        for index in (2, 5):
            await client.call("update_obs_batch", [o for o in observations if o["env_idx"] == index], action_case_id="stack_bowls_case")
            await client.trial_end(trial_id="trial", action_case_id="stack_bowls_case", result={
                "results": [{"env_idx": index, "success": False, "truncated": True}]})
            self.assertFalse(workers[index].is_alive())
            self.assertEqual(self.server.capacity.total, int(index == 2))
            if index == 2: self.assertTrue(workers[5].is_alive())

    async def test_lost_inflight_response_replays_exactly_once(self):
        client = await self.client("inflight")
        await self.begin(client)
        entered, release = asyncio.Event(), asyncio.Event()
        count = 0
        async def slow_action():
            nonlocal count
            count += 1
            entered.set()
            await release.wait()
            return [{"marker": 7}]
        with patch.object(self.server.scheduler, "ready", return_value=True), \
             patch.object(self.model(client), "get_action", new=slow_action):
            request = asyncio.create_task(client.request(MessageType.CALL, {"func_name": "get_action"},
                action_case_id="stack_bowls_case", _request_id="lost-inflight"))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                generation = self.server.next_generation
                for connection in self.server._server.connections: connection.transport.abort()
                async with asyncio.timeout(10):
                    while self.server.case_peers[("inflight", "stack_bowls_case")].generation <= generation:
                        await asyncio.sleep(0.01)
            finally:
                release.set()
            result = await request
            replay = await client.request(MessageType.CALL, {"func_name": "get_action"}, action_case_id="stack_bowls_case", _request_id="lost-inflight")
            self.assertEqual(replay.payload, result.payload)
            self.assertEqual(count, 1)

    async def test_lost_completed_reply_does_not_execute_another_action(self):
        client = await self.client("lost-reply")
        await self.begin(client)
        connection = next(iter(self.server._server.connections))
        send = connection.send
        async def lose_response(data):
            if decode_envelope(data).request_id == "lost-response":
                connection.transport.abort()
                return
            await send(data)
        with patch.object(self.server.scheduler, "ready", return_value=True), \
             patch.object(self.model(client), "get_action", return_value=[{"marker": 8}]) as action, \
             patch.object(connection, "send", new=lose_response):
            result = await client.request(MessageType.CALL, {"func_name": "get_action"}, action_case_id="stack_bowls_case", _request_id="lost-response")
            self.assertEqual(result.payload["result"], [{"marker": 8}])
            action.assert_called_once()

    async def test_overlapping_calls_rejected_before_mutating_state(self):
        client = await self.client("serialized")
        await self.begin(client)
        entered, release = asyncio.Event(), asyncio.Event()
        async def action():
            entered.set()
            await release.wait()
            return []
        model = self.model(client)
        with patch.object(self.server.scheduler, "ready", return_value=True), \
             patch.object(model, "get_action", new=action), patch.object(model, "update_obs") as update:
            pending = asyncio.create_task(client.call("get_action", action_case_id="stack_bowls_case"))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                with self.assertRaisesRegex(WsError, "request in flight"):
                    await client.call("update_obs", observation(7), action_case_id="stack_bowls_case")
                update.assert_not_called()
            finally:
                release.set()
                await pending

    async def test_startup_queue_is_bounded_and_close_cancels_queued_worker(self):
        self.server.scheduler.semaphore = asyncio.Semaphore(1)
        first, second = await self.client("starting"), await self.client("queued")
        await asyncio.gather(self.begin(first), self.begin(second))
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.server.scheduler.run_model
        async def blocked_start(method, *args):
            if method.__self__ is self.model(first).contexts[0]:
                entered.set()
                await release.wait()
            return await original(method, *args)
        with patch.object(self.server.scheduler, "run_model", side_effect=blocked_start):
            await first.call("update_obs", observation(7), action_case_id="stack_bowls_case")
            await first.call("get_action", action_case_id="stack_bowls_case")
            await asyncio.wait_for(entered.wait(), 5)
            await second.call("update_obs", observation(19), action_case_id="stack_bowls_case")
            await second.call("get_action", action_case_id="stack_bowls_case")
            model = self.model(second)
            self.assertIsNone(model.contexts[0].worker)
            self.assertEqual(self.server.scheduler.queued, 1)
            await second.close()
            async with asyncio.timeout(5):
                while ("queued", "stack_bowls_case") in self.server.sessions: await asyncio.sleep(0.01)
            self.assertFalse((model.run / "episodes").exists())
            release.set()
            await self.act(first, "stack_bowls_case", observation(7))
            self.assertEqual(self.server.scheduler.peak_starting, 1)

    async def test_idle_connected_case_expires_without_starting_worker(self):
        self.server.idle_session_s = 0.08
        client = await self.client("idle")
        await self.begin(client)
        async with asyncio.timeout(5):
            while ("idle", "stack_bowls_case") in self.server.sessions: await asyncio.sleep(0.02)
        self.assertEqual(self.server.capacity.total, 0)
        self.assertFalse(list(self.root.rglob("episodes")))

    async def test_startup_failure_retires_case_and_releases_capacity(self):
        client = await self.client("bad-start")
        await self.begin(client)
        context = self.model(client).contexts[0]
        with patch.object(context, "start_worker", side_effect=RuntimeError("cannot launch")):
            await client.call("update_obs", observation(7), action_case_id="stack_bowls_case")
            await client.call("get_action", action_case_id="stack_bowls_case")
            async with asyncio.timeout(5):
                while ("bad-start", "stack_bowls_case") in self.server.sessions: await asyncio.sleep(0.02)
        self.assertEqual(self.server.capacity.total, 0)

    async def test_shutdown_cleans_live_workers(self):
        client = await self.client("shutdown")
        await self.begin(client)
        await self.act(client, "stack_bowls_case", observation(7))
        worker = self.model(client).contexts[0].worker
        await self.server.stop()
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.server.sessions)

    async def test_dead_worker_is_reaped_even_if_client_stops_polling(self):
        client = await self.client("dead-worker")
        await self.begin(client)
        await self.act(client, "stack_bowls_case", observation(7))
        worker = self.model(client).contexts[0].worker
        worker.kill()
        async with asyncio.timeout(5):
            while ("dead-worker", "stack_bowls_case") in self.server.sessions:
                await asyncio.sleep(0.02)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.server.capacity.total, 0)

    async def test_failed_cleanup_retains_capacity_until_retry_succeeds(self):
        client = await self.client("cleanup-retry")
        await self.begin(client)
        slot = self.server.sessions[("cleanup-retry", "stack_bowls_case")]
        with patch.object(slot.model, "close", side_effect=RuntimeError("Docker unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Docker unavailable"):
                await self.server._retire(slot.identity, reason="test cleanup")
            self.assertTrue(slot.retiring)
            self.assertEqual(self.server.capacity.total, 1)
            self.assertIn(slot.identity, self.server.sessions)
        async with asyncio.timeout(5):
            while slot.identity in self.server.sessions:
                await asyncio.sleep(0.02)
        self.assertEqual(self.server.capacity.total, 0)

    async def test_client_rejects_restarted_server(self):
        client = await self.client("restart")
        await client._reset_connection_state()
        await self.server.stop()
        self.server = TaskRoutingServer(self.models, PolicyServerConfig(host="127.0.0.1", port=0), run_dir=self.root / "server")
        await self.server.start()
        client.config.url = self.server.url
        with self.assertRaises(ServerRestartedError): await client.connect()


class ConfigurationTests(unittest.TestCase):
    def test_task_catalog_and_capacity_defaults(self):
        cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
        manifest = json.loads((POLICY_DIR / "Workspace/experience-manifest.json").read_text())
        self.assertEqual(selected_tasks(cfg), sorted(manifest["tasks"]))
        self.assertGreaterEqual(server_settings(cfg)["max_active_agents"], 200)
        with self.assertRaisesRegex(ValueError, "Unknown"): selected_tasks({**cfg, "server_tasks": ["missing"]})
        with self.assertRaises(ValueError): server_settings({**cfg, "server_max_sessions": True})

    def test_failed_snapshot_initialization_closes_prior_templates(self):
        cfg = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
        cfg.update(diagnostic=True, server_tasks=["stack_bowls", "push_T"])
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, EMBODIEDEVO_RUN_DIR=str(Path(root) / "run")), \
             patch("XPolicyLab.policy.EmbodiedEvo.runtime.serving.Model") as factory:
            model = factory.return_value
            factory.side_effect = [model, RuntimeError("missing task assets")]
            with self.assertRaisesRegex(RuntimeError, "missing task assets"): prepare_models(cfg)
            model.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
