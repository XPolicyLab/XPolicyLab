"""Configuration: layered TOML, --set, the agent, model files and experiments, locks and
identities, and the agent's defaults (the leaderboard protocol)."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from helpers import TEST_DEPLOY, agent_file, load_test_experiment  # noqa: E402
from harness_policy import AGENT_VERSION  # noqa: E402
from harness_policy.config import apply_sets, deep_merge, load_layered, parse_set  # noqa: E402
from harness_policy.config.agent import DEFAULT_AGENT, model_names, resolve_agent  # noqa: E402


class LayeredTomlTests(unittest.TestCase):
    def test_includes_merge_and_paths_follow_their_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base").mkdir()
            (root / "base" / "a.toml").write_text('prompt = "p"\n[runtime]\nx = 1\ny = { a = 1, b = 2 }\n')
            (root / "b.toml").write_text('include = ["base/a.toml"]\n[runtime]\ny = { b = 3 }\n')
            tree = load_layered(root / "b.toml")
        self.assertEqual(tree["runtime"], {"x": 1, "y": {"a": 1, "b": 3}})
        self.assertEqual(Path(tree["prompt"]), (root / "base" / "p").resolve())

    def test_include_cycles_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a.toml").write_text('include = ["b.toml"]\n')
            (Path(directory) / "b.toml").write_text('include = ["a.toml"]\n')
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_layered(Path(directory) / "a.toml")

    def test_set_values_are_toml(self):
        self.assertEqual(parse_set("a.b=3"), (["a", "b"], 3))
        self.assertEqual(parse_set('m="x/y"'), (["m"], "x/y"))
        self.assertEqual(parse_set("m=x/y"), (["m"], "x/y"))
        self.assertEqual(parse_set("l=[1, 2]"), (["l"], [1, 2]))
        self.assertEqual(apply_sets({"a": {"b": 1}}, ["a.c=true"]), {"a": {"b": 1, "c": True}})
        self.assertEqual(deep_merge({"a": [1]}, {"a": [2]}), {"a": [2]})


class AgentTests(unittest.TestCase):
    def test_the_default_agent_and_its_model(self):
        agent = resolve_agent(deploy=str(TEST_DEPLOY))
        self.assertEqual(agent.name, "default")
        self.assertEqual(agent.path, DEFAULT_AGENT)
        self.assertEqual(agent.lock["agent"], "default")
        self.assertEqual(agent.lock["agent_version"], AGENT_VERSION)
        self.assertEqual(agent.main["model"], load_layered(ROOT / "configs" / "models" /
                                                           f"{load_layered(DEFAULT_AGENT)['model']}.toml")
                         ["main"]["model"])
        self.assertEqual(set(agent.runtime["layers"].values()), {True})

    def test_the_default_follows_the_leaderboard_protocol(self):
        """What RoboDojo's official client hands a policy and no more: the XPolicyLab
        contract, no task descriptions, 30 model calls, every layer and the verifier; the
        submitted configuration: SAM 3 and Simate, no pointer role, no world model."""
        agent = resolve_agent(deploy=str(TEST_DEPLOY))
        self.assertEqual(agent.runtime["contract"], "xpolicylab")
        self.assertNotIn("task_descriptions", agent.runtime)
        self.assertEqual(agent.max_model_calls, 30)
        self.assertTrue(agent.runtime["verifier"])
        self.assertEqual(agent.lock["services"], {"segmenter": "sam3", "policies": ["simate"]})
        self.assertNotIn("pointer", agent.lock["roles"])
        self.assertNotIn("pointer", agent.runtime)
        for skill in ("vla", "reactive", "handlers", "verify", "eef", "scene", "pointing"):
            self.assertIn(skill, agent.runtime["skills"])
        self.assertNotIn("wam", agent.runtime["skills"])

    def test_task_descriptions_are_an_ablation(self):
        run = load_test_experiment("with-task-description")
        config = run.task_config()
        self.assertTrue(config.task_description.startswith("#"))
        self.assertEqual(run.agent.runtime["contract"], "xpolicylab")
        self.assertEqual(load_test_experiment().task_config().task_description, "")

    def test_a_model_by_name_or_file(self):
        by_name = resolve_agent(model="grok-low", deploy=str(TEST_DEPLOY))
        by_file = resolve_agent(model=str(ROOT / "configs" / "models" / "grok-low.toml"), deploy=str(TEST_DEPLOY))
        self.assertEqual(by_name.lock, by_file.lock)
        self.assertEqual(by_name.main["reasoning_effort"], "low")
        self.assertIn("grok-low", model_names())
        with self.assertRaisesRegex(ValueError, "models: "):
            resolve_agent(model="no-such-model", deploy=str(TEST_DEPLOY))

    def test_ablations_change_only_what_they_ablate(self):
        default = resolve_agent(deploy=str(TEST_DEPLOY)).lock
        for name, changed in {"no-handlers": {"skills"}, "no-verifier": set(), "r3": {"skills"},
                              # R2's learned layer is off, so the agent's world-model skill is not
                              # requested there rather than dropped for want of a service.
                              "r2": {"skills", "skills_dropped", "services"}, "with-task-description": set()}.items():
            with self.subTest(ablation=name):
                lock = resolve_agent(agent_file(name), deploy=str(TEST_DEPLOY)).lock
                differs = {key for key in default if key not in ("agent", "prompt_sha256", "runtime", "service_manifests")
                           and lock.get(key) != default.get(key)}
                self.assertEqual(differs, changed)
                runtime = {key for key in default["runtime"] if lock["runtime"].get(key) != default["runtime"][key]}
                self.assertLessEqual(runtime, {"layers", "skills", "skills_dropped", "verifier"}
                                     | ({"task_descriptions"} if name == "with-task-description" else set()))

    def test_identity_ignores_where_services_run(self):
        a = load_test_experiment().agent
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "deploy.toml"
            moved.write_text(TEST_DEPLOY.read_text().replace("[services.sam3]\ngpu = 0",
                                                             '[services.sam3]\nmode = "external"\nurl = "http://10.1.2.3:9000"'))
            b = load_test_experiment(deploy=moved).agent
        self.assertEqual(a.runtime["segmenter"]["url"], "http://127.0.0.1:8810")
        self.assertEqual(b.runtime["segmenter"]["url"], "http://10.1.2.3:9000")
        self.assertEqual(a.identity, b.identity)
        self.assertNotEqual(a.identity, load_test_experiment("r3").agent.identity)
        self.assertNotEqual(a.identity, load_test_experiment(model="grok-low").agent.identity)

    def test_the_version_is_part_of_the_identity(self):
        a = resolve_agent(deploy=str(TEST_DEPLOY))
        with mock.patch("harness_policy.config.agent.AGENT_VERSION", "0.0.1"):
            b = resolve_agent(deploy=str(TEST_DEPLOY))
        self.assertNotEqual(a.identity, b.identity)

    def test_a_service_the_deploy_does_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deploy.toml"
            path.write_text(TEST_DEPLOY.read_text().replace("[services.simate]\ngpu = 4", ""))
            run = load_test_experiment(deploy=path)
            self.assertNotIn("vla", run.agent.runtime["skills"])
            self.assertEqual(run.agent.runtime["skills_dropped"]["vla"], "no policy service")
            self.assertNotIn("run_policy(", run.agent.system_prompt)
            with self.assertRaisesRegex(ValueError, "simate"):
                load_test_experiment(deploy=path, strict=True)

    def test_strict_resolution_accepts_skills_left_out_by_design(self):
        # R0 has no vision roles: its dropped skills are the configuration's own choice on
        # every machine, not something this machine lacks.
        for agent in ("r0", "r1"):
            with self.subTest(agent=agent):
                run = load_test_experiment(agent, strict=True)
                self.assertIn("core", run.agent.runtime["skills"])
                self.assertEqual(run.agent.lock, load_test_experiment(agent, strict=False).agent.lock)

    def test_strict_resolution_refuses_a_named_skill_that_cannot_run(self):
        tree = {"prompt": str(ROOT / "prompts" / "robodojo"), "model": {"main": {"model": "m"}},
                "skills": {"names": ["core", "segmentation"]}}
        with self.assertRaisesRegex(ValueError, "skills it names cannot run"):
            resolve_agent(tree, deploy=str(TEST_DEPLOY), strict=True, name="t")
        self.assertNotIn("segmentation", resolve_agent(tree, deploy=str(TEST_DEPLOY), name="t").runtime["skills"])

    def test_overrides_reach_the_agent_and_model(self):
        run = load_test_experiment(model="grok-low", sets=["agent.runtime.keep_images=1",
                                                           'model.main.model="z-ai/glm-5"', "budget.max_cost_usd=2"])
        config = run.task_config()
        self.assertEqual(config.runtime["keep_images"], 1)
        self.assertEqual(config.main["model"], "z-ai/glm-5")
        self.assertEqual(config.main["reasoning_effort"], "low")
        self.assertEqual(config.max_cost_usd, 2)
        self.assertEqual(config.agent_id, run.agent.identity)
        self.assertEqual(config.snapshot()["agent_version"], AGENT_VERSION)

    def test_model_on_the_command_line_wins(self):
        from harness_policy.config.experiments import load_experiment

        tree = {"model": "grok-medium", "tasks": ["general_pickup"]}
        run = load_experiment(tree=tree, deploy=str(TEST_DEPLOY), model="grok-low")
        self.assertEqual(run.agent.main["reasoning_effort"], "low")
        run = load_experiment(tree=tree, deploy=str(TEST_DEPLOY), sets=["model=grok-low"])
        self.assertEqual(run.agent.main["reasoning_effort"], "low")

    def test_an_experiment_without_a_file(self):
        from harness_policy.config.experiments import load_experiment

        run = load_experiment(deploy=str(TEST_DEPLOY), tasks=["general_pickup"])
        self.assertEqual((run.agent.name, run.bench, run.layouts, run.seeds), ("default", "robodojo", "procedural", [0]))
        with self.assertRaisesRegex(ValueError, "names no tasks"):
            load_experiment(deploy=str(TEST_DEPLOY))

    def test_unknown_keys_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.toml"
            path.write_text('prompt = "x"\nsystem = "m"\n')
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                resolve_agent(path, model={"main": {"model": "m"}}, deploy=str(TEST_DEPLOY))

    def test_every_config_file_resolves(self):
        from harness_policy.config.experiments import load_experiment

        resolve_agent(deploy=str(TEST_DEPLOY))
        for path in sorted((ROOT / "configs" / "ablations").glob("*.toml")):
            with self.subTest(ablation=path.name):
                resolve_agent(path, deploy=str(TEST_DEPLOY))
        for path in sorted((ROOT / "configs" / "models").glob("*.toml")):
            with self.subTest(model=path.name):
                resolve_agent(model=path.stem, deploy=str(TEST_DEPLOY))
        for path in sorted((ROOT / "configs" / "experiments").glob("*.toml")):
            with self.subTest(experiment=path.name):
                if "robotwin" in path.name or "eval" in path.name:
                    continue  # task lists come from the benchmark checkouts
                load_experiment(path, deploy=str(TEST_DEPLOY))
        for path in sorted((ROOT / "configs" / "deploy").glob("*.toml")):
            with self.subTest(deploy=path.name):
                from harness_policy.services.deploy import load_deploy
                load_deploy(path)


if __name__ == "__main__":
    unittest.main()
