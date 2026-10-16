"""Exercise the hyphenated policy package and its factory/serve entry points."""

from importlib import import_module
from pathlib import Path
import runpy
import signal
import sys
from types import ModuleType
from unittest import TestCase
from unittest.mock import Mock, patch

import yaml


POLICY_DIR = Path(__file__).resolve().parents[1]
PACKAGE = "XPolicyLab.policy.SenseNova-RoboRSI"


class TestPolicyName(TestCase):
    def test_deploy_name_loads_the_renamed_package(self):
        config = yaml.safe_load((POLICY_DIR / "deploy.yml").read_text())
        self.assertEqual(config["policy_name"], POLICY_DIR.name)
        module = import_module(f"XPolicyLab.policy.{config['policy_name']}")
        self.assertEqual(Path(module.__file__).resolve().parent, POLICY_DIR)

    def test_factory_uses_the_canonical_package_module(self):
        preset = yaml.safe_load((POLICY_DIR / "agent.yaml").read_text())
        module_name, function_name = preset["factory"].split(":")
        self.assertTrue(all(part.isidentifier() for part in module_name.split(".")))
        self.assertTrue(function_name.isidentifier())
        source_root = (POLICY_DIR / preset["source_root"]).resolve()
        hybrid = ModuleType(f"{PACKAGE}.hybrid_agent")
        hybrid.make_agent = Mock(return_value=object())
        case, config = object(), object()
        with patch.object(sys, "path", [str(source_root), *sys.path]):
            factory_module = import_module(module_name)
        self.assertIn(
            Path(factory_module.__file__).resolve(),
            [(POLICY_DIR / name).resolve() for name in preset["code"]],
        )
        with patch.dict(sys.modules, {hybrid.__name__: hybrid}):
            result = getattr(factory_module, function_name)(case, config=config)
        self.assertIs(result, hybrid.make_agent.return_value)
        hybrid.make_agent.assert_called_once_with(case, config=config)

    def test_standalone_serve_shutdown_imports_the_renamed_model(self):
        namespace = runpy.run_path(str(POLICY_DIR / "serve.py"))
        model = ModuleType(f"{PACKAGE}.model")
        model.close_models = Mock()
        with patch.dict(sys.modules, {model.__name__: model}):
            with self.assertRaises(SystemExit) as stopped:
                namespace["_terminate"](signal.SIGTERM, None)
        self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
        model.close_models.assert_called_once_with()
