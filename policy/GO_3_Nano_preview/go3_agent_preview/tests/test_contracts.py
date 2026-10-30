"""Task loading and native transport contracts; no weights or simulator required."""
import sys
from pathlib import Path
import pytest
import yaml
_XPL_ROOT = Path(__file__).resolve().parents[3]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))
from XPolicyLab.policy.go3_agent_preview.gm import model as gm_module
from support import cfg, DIMS
ADAPTER = Path(__file__).resolve().parents[1]
ROBODOJO_TASKS = (
    "align_blocks arrange_largest_number arrange_largest_number_random build_tower "
    "classify_objects classify_objects_by_language cover_blocks deposit_coin "
    "fasten_screws fill_egg_holder fill_pen_holder fold_clothes fold_clothes_random "
    "general_pickup hang_mugs hang_mugs_random imitate_sorting_sequence insert_key "
    "insert_tubes make_kong make_toast make_toast_random match_and_pick_from_conveyor "
    "organize_table pack_objects_into_box pack_objects_into_box_random "
    "pick_from_conveyor_by_image play_Xylophone play_stacking_toy play_tic_tac_toe "
    "plug_in_charger pour_balls_into_vase pour_by_language pour_liquid_into_cup "
    "pour_liquid_into_cup_random press_by_number push_T push_T_random "
    "put_bottles_into_dustbin solve_equation sort_nesting_dolls_by_size "
    "sort_nesting_dolls_by_size_random stack_blocks stack_blocks_by_language "
    "stack_blocks_random stack_bowls stack_bowls_random store_laptop_and_headphones "
    "store_laptop_and_headphones_random store_tools_in_toolbox swap_T swap_blocks "
    "sweep_blocks sweep_blocks_random"
).split()

def test_random_variants_run_the_base_program(monkeypatch):
    # GmModel loads a program by task name and falls back to the base task's
    # program when no `<task>_random.py` exists. A `<task>_random.py` that only
    # re-exports its base's `run` looks like an unused import to a linter;
    # losing that line leaves a module the loader finds but cannot run.
    monkeypatch.setattr(gm_module,"get_robot_action_dim_info",lambda _:DIMS)
    variants = [task for task in ROBODOJO_TASKS if task.endswith("_random")]
    assert variants
    for name in variants:
        model = gm_module.GmModel(cfg(name))
        run = model._load(name).run
        base = name.removesuffix("_random")
        assert run.__module__.split(".")[-1] == base, (name, run.__module__)
        model.close()


# --- env-client request budget ------------------------------------------------
#
# The budget belongs to the adapter, not to shared transport code: RoboDojo hands
# `deploy.py` the client it built, and the code branch can outlast the protocol
# default. These tests cover the adapter-side override only.


class _FakeClient:
    """A client shaped like WsModelClient: a `_client.config` with both budgets."""

    def __init__(self, request=120.0, handshake=60.0, with_config=True):
        self.calls = []
        config = type("Config", (), {"request_timeout_s": request,
                                     "handshake_timeout_s": handshake})()
        self._client = type("Inner", (), {"config": config})() if with_config else object()

    def call(self, func_name=None, obs=None, **kwargs):
        self.calls.append(func_name)
        if func_name == "reset":
            return None
        if func_name == "get_action":
            return [[]]
        return None


def test_positive_seconds_reads_values_and_ignores_blank(monkeypatch):
    from XPolicyLab.policy.go3_agent_preview import deploy

    monkeypatch.delenv("XPL_TEST_BUDGET", raising=False)
    assert deploy._positive_seconds("XPL_TEST_BUDGET") is None
    monkeypatch.setenv("XPL_TEST_BUDGET", "  ")
    assert deploy._positive_seconds("XPL_TEST_BUDGET") is None
    monkeypatch.setenv("XPL_TEST_BUDGET", "900")
    assert deploy._positive_seconds("XPL_TEST_BUDGET") == 900.0


@pytest.mark.parametrize("raw", ["nope", "0", "-5"])
def test_positive_seconds_rejects_values_it_cannot_use(monkeypatch, raw):
    from XPolicyLab.policy.go3_agent_preview import deploy

    monkeypatch.setenv("XPL_TEST_BUDGET", raw)
    with pytest.raises(ValueError, match="XPL_TEST_BUDGET"):
        deploy._positive_seconds("XPL_TEST_BUDGET")


def test_deploy_raises_the_request_budget_from_the_environment(monkeypatch):
    from XPolicyLab.policy.go3_agent_preview import deploy

    monkeypatch.setenv("XPL_REQUEST_TIMEOUT_S", "900")
    monkeypatch.setenv("XPL_HANDSHAKE_TIMEOUT_S", "120")
    client = _FakeClient()
    deploy._apply_budget(client)
    assert client._client.config.request_timeout_s == 900.0
    assert client._client.config.handshake_timeout_s == 120.0


def test_deploy_leaves_the_budget_alone_when_the_environment_is_unset(monkeypatch):
    from XPolicyLab.policy.go3_agent_preview import deploy

    monkeypatch.delenv("XPL_REQUEST_TIMEOUT_S", raising=False)
    monkeypatch.delenv("XPL_HANDSHAKE_TIMEOUT_S", raising=False)
    client = _FakeClient()
    deploy._apply_budget(client)
    assert client._client.config.request_timeout_s == 120.0
    assert client._client.config.handshake_timeout_s == 60.0


def test_deploy_tolerates_a_client_without_the_expected_shape(monkeypatch):
    """A transport double keeps its own defaults instead of raising."""
    from XPolicyLab.policy.go3_agent_preview import deploy

    monkeypatch.setenv("XPL_REQUEST_TIMEOUT_S", "900")
    deploy._apply_budget(_FakeClient(with_config=False))


def test_deploy_applies_the_budget_before_the_first_call(monkeypatch):
    from XPolicyLab.policy.go3_agent_preview import deploy

    monkeypatch.setenv("XPL_REQUEST_TIMEOUT_S", "777")
    client = _FakeClient()

    class _Env:
        def is_episode_end(self):
            return True

    deploy.eval_one_episode(_Env(), client)
    assert client._client.config.request_timeout_s == 777.0
    assert client.calls[0] == "reset"


def test_all_runnable_tasks_load_their_own_program(monkeypatch):
    monkeypatch.setattr(gm_module,"get_robot_action_dim_info",lambda _:DIMS)
    for task in ROBODOJO_TASKS:
        model=gm_module.GmModel(cfg(task))
        assert callable(model._load(model.program_name()).run), task
        model.close()


def test_shipped_deploy_has_only_code_trace_config():
    shipped=yaml.safe_load((ADAPTER/"deploy.yml").read_text())
    assert shipped["code"]=={"out_dir":None,"worker_startup_timeout_s":30}
    assert "gm" not in shipped
    assert set(shipped).isdisjoint({"routing","handoff","skill_switch","router"})


def test_pen_fill_generic_exception_has_no_switch_checkpoint():
    import ast
    from XPolicyLab.policy.go3_agent_preview.gm.runtime.session import SWITCH_SOURCES
    tree=ast.parse((ADAPTER/"gm/programs/fill_pen_holder.py").read_text())
    run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="run")
    generic=[node for node in ast.walk(run) if isinstance(node,ast.ExceptHandler)
             and isinstance(node.type,ast.Name) and node.type.id=="Exception"]
    assert generic
    assert all(not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                   and node.func.attr=="switch_to_wam") for handler in generic for node in ast.walk(handler))
    assert "fill_pen_holder.fill_abandoned" not in SWITCH_SOURCES["fill_pen_holder"]


def test_42_base_tasks_have_registered_explicit_caller_coverage():
    from XPolicyLab.policy.go3_agent_preview.gm.runtime.session import Ctx, SwitchReason, SWITCH_SOURCES
    base={task for task in ROBODOJO_TASKS if not task.endswith("_random")}
    assert len(base)==42
    assert base.issubset(SWITCH_SOURCES)
    for task in base:
        assert SWITCH_SOURCES[task]
        source=sorted(SWITCH_SOURCES[task])[0]
        request=Ctx(task,None).switch_to_wam(SwitchReason.CODE_ABORTED,detail="synthetic allowlist validation",source=source)
        assert request.source==source
