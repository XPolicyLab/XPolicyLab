"""The episode core works for any embodiment kind: bounded executions, guards, the shield's
verdicts and scopes on a session that is not a robot's; skills, embodiment files, agent
resolution and the driver keyed by the embodiment kind, with robot results as before."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from harness_policy.config.agent import resolve_agent  # noqa: E402
from harness_policy.core.contracts import Observation  # noqa: E402
from harness_policy.embodiments import KINDS, load_embodiment, profile  # noqa: E402
from harness_policy.runtime import episode as episode_module  # noqa: E402
from harness_policy.runtime import events as ev  # noqa: E402
from harness_policy.runtime.chat import ScriptedChat  # noqa: E402
from harness_policy.runtime.driver import TaskConfig, run_program_episode  # noqa: E402
from harness_policy.runtime.events import Escalation  # noqa: E402
from harness_policy.runtime.skills import load_skills, mixins, resolve, skills_for_layers  # noqa: E402

from helpers import TEST_DEPLOY  # noqa: E402

PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
EVERY_SERVICE = {"segmenter", "pointer", "vlm", "verifier", "policy", "world_model", "mcp"}


class ScreenWorker:
    """A worker for a computer-like embodiment: a command is an input event, the state
    has the focused window, and the shield rejects deleting anything."""

    def __init__(self, title="Editor", switch_at=None):
        self.state = {"step": 0, "ended": False, "title": title}
        self.switch_at = switch_at
        self.commands = []

    def call(self, name, **arguments):
        if name == "profile":
            return {"kind": "desktop", "control_hz": 2.0, "screen": [1280, 720]}
        if name == "state":
            return dict(self.state)
        if name == "tick":
            command = arguments["command"]
            self.commands.append(command)
            if command.get("key") == "delete":
                return {"executed": False, "state": dict(self.state),
                        "shield": {"verdict": "rejected", "command": None, "reasons": [{"check": "irreversible"}]}}
            self.state["step"] += 1
            if self.switch_at is not None and self.state["step"] >= self.switch_at:
                self.state["title"] = "Terminal"
            return {"executed": True, "state": dict(self.state),
                    "shield": {"verdict": "pass", "command": command, "reasons": []}}
        raise AssertionError(name)

    def close(self):
        pass


class ScreenSession(episode_module.EpisodeCore):
    """The episode core with a computer's commands: nothing of the robot's session."""

    embodiment = "desktop"

    def _base_command(self, state):
        return {"type": "noop"}

    def _tick(self, command, cameras=None):
        result = self.env.call("tick", episode_id=self.episode_id, step=self.state["step"], command=command)
        self.state = result["state"]
        return result

    def _reactive_view(self, state, tick):
        return {"step": state["step"], "t": tick / self.hz, "title": state["title"]}

    def _outside(self, region, state):
        return state["title"] != region["window"]


def screen_session(worker):
    with mock.patch.object(episode_module, "RemoteEnvironment", lambda *args, **kwargs: worker):
        return ScreenSession({"endpoint": "http://127.0.0.1:1", "episode_id": "e", "skills": [],
                              "reactive_budget_ms": 50})


def typing(count):
    def command(state, tick):
        return {"type": "text", "text": "a"} if tick < count else None
    return command


class EpisodeCoreTests(unittest.TestCase):
    def test_a_bounded_execution_on_another_embodiment(self):
        worker = ScreenWorker()
        session = screen_session(worker)
        trace = session._execute("type_text", {}, typing(3), max_seconds=10, region=None, until=None)
        self.assertEqual((trace.event.kind, trace.ticks), (ev.COMPLETED, 3))
        self.assertEqual(trace.samples, [{"step": 1}])
        self.assertEqual(session.namespace, {"__name__": "__program__"})
        with self.assertRaises(Escalation) as raised:
            session._execute("type_text", {}, typing(100), max_seconds=1, region=None, until=None)
        self.assertEqual((raised.exception.trace.event.kind, raised.exception.trace.ticks), (ev.TIMEOUT, 2))
        trace = session._execute("type_text", {}, typing(100), max_seconds=10, region=None,
                                 until=lambda view: view["step"] >= 7)
        self.assertEqual((trace.event.kind, worker.state["step"]), (ev.PREDICATE_SATISFIED, 7))

    def test_guards_take_the_kinds_actions(self):
        """stop and hold are every embodiment's; open_gripper is the robot's, so on another
        embodiment it is an event the model gets."""
        worker = ScreenWorker()
        session = screen_session(worker)
        session.reactive.when(lambda view: view["step"] >= 2, "hold", name="settle")
        with self.assertRaises(Escalation) as raised:
            session._execute("type_text", {}, typing(10), max_seconds=10, region=None, until=None)
        event = raised.exception.trace.event
        self.assertEqual((event.kind, event.detail["action"]), (ev.GUARD_TRIGGERED, "hold"))
        self.assertEqual(worker.commands[-1], {"type": "noop"})
        session.reactive.clear()
        session.reactive.when(lambda view: True, "open_gripper", name="grip")
        with self.assertRaises(Escalation) as raised:
            session._execute("type_text", {}, typing(10), max_seconds=10, region=None, until=None)
        self.assertEqual(raised.exception.trace.event.kind, "open_gripper")
        self.assertEqual(session.guard_actions, ("stop", "hold"))

    def test_shield_rejection_escalates_unless_handled(self):
        worker = ScreenWorker()
        session = screen_session(worker)

        def delete(state, tick):
            return {"type": "key", "key": "delete"}
        with self.assertRaises(Escalation) as raised:
            session._execute("press_key", {}, delete, max_seconds=5, region=None, until=None)
        self.assertEqual(raised.exception.trace.event.kind, ev.SHIELD_REJECTED)
        session.handlers[ev.SHIELD_REJECTED] = lambda trace: True
        trace = session._execute("press_key", {}, delete, max_seconds=5, region=None, until=None)
        self.assertEqual(trace.event.kind, ev.SHIELD_REJECTED)
        self.assertTrue(session.drain()["events"][-1]["handled"])

    def test_leaving_the_scope_ends_the_execution(self):
        worker = ScreenWorker(switch_at=2)
        session = screen_session(worker)
        with self.assertRaises(Escalation) as raised:
            session._execute("type_text", {}, typing(10), max_seconds=10, region={"window": "Editor"}, until=None)
        self.assertEqual(raised.exception.trace.event.kind, ev.REGION_EXIT)
        self.assertEqual(worker.state["step"], 2)

    def test_sessions_exist_per_kind(self):
        robot = episode_module.session_class("robot")
        self.assertIs(robot, episode_module.session_class())
        self.assertEqual(robot.embodiment, "robot")
        self.assertEqual(robot.guard_actions, ("stop", "hold", "open_gripper"))
        with self.assertRaisesRegex(ValueError, "No session"):
            episode_module.session_class("desktop")


class SkillKindTests(unittest.TestCase):
    def test_robot_skills_are_as_before(self):
        layers = {"learned": True, "reactive": True, "handlers": True}
        self.assertEqual(skills_for_layers(layers), skills_for_layers(layers, embodiment="robot"))
        self.assertTrue(all(skill.embodiments == ("robot",) for skill in load_skills().values()))
        self.assertEqual(skills_for_layers(layers, embodiment="desktop"), [])
        self.assertEqual(mixins(embodiment="desktop"), [])

    def test_a_skill_for_another_embodiment_is_dropped(self):
        resolution = resolve(["core", "motion"], EVERY_SERVICE, embodiment="desktop")
        self.assertEqual(resolution.enabled, [])
        self.assertEqual(resolution.dropped["motion"], "does not run on the desktop embodiment")
        with self.assertRaisesRegex(ValueError, "motion"):
            resolve(["core", "motion"], EVERY_SERVICE, embodiment="desktop", strict=True)
        with self.assertRaisesRegex(ValueError, "kind"):
            resolve(["core"], EVERY_SERVICE, embodiment="boat")

    def test_skills_for_different_embodiments_may_share_names(self):
        def write(root, name, kinds):
            (root / name).mkdir()
            (root / name / "skill.toml").write_text(
                f'name = "{name}"\nmixin = "m:C"\nexports = ["finish"]\nembodiments = {json.dumps(kinds)}\n')

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, "arm_core", ["robot"])
            write(root, "screen_core", ["desktop"])
            self.assertEqual(sorted(load_skills(root)), ["arm_core", "screen_core"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, "arm_core", ["robot"])
            write(root, "both_core", ["robot", "desktop"])
            with self.assertRaisesRegex(ValueError, "more than one skill"):
                load_skills(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, "boat_core", ["boat"])
            with self.assertRaisesRegex(ValueError, "embodiments"):
                load_skills(root)


class EmbodimentKindTests(unittest.TestCase):
    def test_each_kind_has_its_keys(self):
        self.assertEqual(sorted(KINDS), ["desktop", "robot"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embodiment.toml"
            path.write_text('name = "ubuntu"\nkind = "desktop"\ncontrol_hz = 2.0\nscreen = [1280, 720]\n')
            self.assertEqual(load_embodiment(path)["screen"], [1280, 720])
            with self.assertRaisesRegex(ValueError, "robots only"):
                profile(load_embodiment(path))
            path.write_text('name = "ubuntu"\nkind = "desktop"\ncontrol_hz = 2.0\n')
            with self.assertRaisesRegex(ValueError, "missing screen"):
                load_embodiment(path)
            path.write_text('name = "boat"\nkind = "boat"\n')
            with self.assertRaisesRegex(ValueError, "kind"):
                load_embodiment(path)
        self.assertNotIn("kind", load_embodiment("arx_x5"))


class AgentKindTests(unittest.TestCase):
    def test_a_robot_lock_names_no_kind(self):
        agent = resolve_agent(deploy=str(TEST_DEPLOY))
        self.assertEqual(agent.embodiment, "robot")
        self.assertNotIn("embodiment", agent.lock)

    def test_a_prompt_table_gives_the_kinds_prompt(self):
        from harness_policy.config import load_layered
        from harness_policy.config.agent import DEFAULT_AGENT

        tree = load_layered(DEFAULT_AGENT)
        default = resolve_agent(tree, deploy=str(TEST_DEPLOY), name="default")
        table = resolve_agent({**tree, "prompt": {"robot": tree["prompt"]}}, deploy=str(TEST_DEPLOY), name="default")
        self.assertEqual(table.identity, default.identity)
        with self.assertRaisesRegex(ValueError, "desktop"):
            resolve_agent({**tree, "prompt": {"robot": tree["prompt"]}}, deploy=str(TEST_DEPLOY), name="default",
                          embodiment="desktop")
        with tempfile.TemporaryDirectory() as directory:
            screen = Path(directory) / "screen.md"
            screen.write_text("You operate a computer.\n")
            desktop = resolve_agent({**tree, "prompt": {"robot": tree["prompt"], "desktop": str(screen)}},
                                    deploy=str(TEST_DEPLOY), name="default", embodiment="desktop")
        self.assertEqual((desktop.embodiment, desktop.lock["embodiment"]), ("desktop", "desktop"))
        self.assertEqual(desktop.runtime["skills"], [])
        self.assertEqual(desktop.system_prompt, "You operate a computer.\n")
        self.assertNotEqual(desktop.identity, default.identity)
        with self.assertRaisesRegex(ValueError, "kind"):
            resolve_agent(tree, deploy=str(TEST_DEPLOY), embodiment="boat")


class ScreenEnvironment:
    """The driver's calls to a computer worker; `kind` is what its profile reports."""

    def __init__(self, kind="desktop"):
        self.kind = kind

    def tools(self):
        return []

    def reset(self, task, seed):
        return Observation("episode-1", 0, "Rename the file.", {"step": 0, "active_window": "Files"},
                           {"screen": PIXEL})

    def call(self, name, **arguments):
        if name == "profile":
            return {"kind": self.kind, "control_hz": 2.0}
        if name == "claim_outcome":
            return {"token": "secret"}
        if name == "observe":
            return {"state": {"step": 0, "ended": False, "active_window": "Files"}, "images": {"screen": PIXEL}}
        if name == "outcome":
            return {"reward": 1.0, "success": True, "process_score": 100.0, "steps": 1,
                    "terminated": True, "truncated": False}
        raise AssertionError(name)


class Recording(ScriptedChat):
    def __init__(self, replies):
        super().__init__(replies)
        self.systems, self.requests = [], []

    def complete(self, system, messages):
        self.systems.append(system)
        self.requests.append(json.loads(json.dumps(messages)))
        return super().complete(system, messages)


def screen_config(embodiment="desktop"):
    return TaskConfig(
        path=Path("."), environment={"endpoint": "http://127.0.0.1:1"}, main={"model": "scripted"},
        runtime={"bootstrap": "harness_policy.runtime.kernel:PlainSession", "isolation": False, "skills": ["core"]},
        system_prompt="system", task="rename_file", seeds=[0], max_model_calls=4, max_wall_seconds=60,
        embodiment=embodiment)


class DriverKindTests(unittest.TestCase):
    def test_the_model_reads_about_the_embodiment(self):
        chat = Recording(["<verify>the file is renamed</verify>", "<finish>done</finish>"])
        verifier = Recording(["YES the name changed"])
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(screen_config(), 0, Path(root) / "seed-0", chat=chat, verifier=verifier,
                                          environment=ScreenEnvironment())
        self.assertEqual(summary["status"], "finished")
        first = chat.requests[0][0]["content"]
        self.assertTrue(first[0]["text"].endswith('Computer state: {"step": 0, "active_window": "Files"}'))
        self.assertEqual([part["text"] for part in first if part["type"] == "input_text"][1:], ["Camera: screen"])
        self.assertIn("computer", verifier.systems[0])
        self.assertIn("Computer state:", chat.requests[1][-1]["content"][0]["text"])
        self.assertEqual(screen_config().snapshot()["embodiment"], "desktop")
        self.assertNotIn("embodiment", screen_config("robot").snapshot())

    def test_a_worker_of_another_kind_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            summary = run_program_episode(screen_config("robot"), 0, Path(root) / "seed-0",
                                          chat=ScriptedChat(["<finish>done</finish>"]), verifier=ScriptedChat([]),
                                          environment=ScreenEnvironment("desktop"))
        self.assertEqual(summary["status"], "error")
        self.assertIn("drives a desktop embodiment", summary["error"])


if __name__ == "__main__":
    unittest.main()
