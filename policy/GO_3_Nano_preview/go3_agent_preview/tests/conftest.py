from types import SimpleNamespace
import pytest
from support import cfg, DIMS, Backend
from XPolicyLab.policy.go3_agent_preview.gm import model as gm_module
from XPolicyLab.policy.go3_agent_preview import model as model_module
from XPolicyLab.policy.go3_agent_preview import skill_switch
from XPolicyLab.policy.go3_agent_preview.wam import model as wam_module
@pytest.fixture
def setup_model(monkeypatch,tmp_path):
    monkeypatch.setattr(gm_module,"get_robot_action_dim_info",lambda _:DIMS)
    monkeypatch.setattr(wam_module,"get_robot_action_dim_info",lambda _:DIMS)
    monkeypatch.setattr(skill_switch,"get_robot_action_dim_info",lambda _:DIMS)
    # Direct code for deterministic CPU tests; isolation has separate process tests.
    monkeypatch.setattr(model_module,"IsolatedGm",gm_module.GmModel)
    real_wam=wam_module.Model
    built=[]
    def make(program,*,backend=None,extra=None):
        backend=backend or Backend()
        def factory(c):built.append(backend);return real_wam(c,client=backend)
        monkeypatch.setattr(wam_module,"Model",factory)
        monkeypatch.setattr(gm_module.GmModel,"_load",lambda self,name:SimpleNamespace(run=program))
        config={**cfg(),"seed":3,"code":{"out_dir":str(tmp_path)},**(extra or {})}
        model=model_module.Model(config)
        model.test_wam_builds=built
        return model,backend
    return make
