import ast
from pathlib import Path


POLICY_DIR = Path(__file__).resolve().parents[1]


def test_model_template_contract_without_importing_heavy_dependencies():
    source = (POLICY_DIR / "model.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    methods = {node.name for node in model.body if isinstance(node, ast.FunctionDef)}
    assert {"__init__", "update_obs", "update_obs_batch", "get_action", "get_action_batch", "reset"} <= methods
    assert "ModelTemplate" in {ast.unparse(base) for base in model.bases}
    assert "get_robot_action_dim_info" in source
    for forbidden in ("cv2.imdecode", "np.frombuffer", "Image.open"):
        assert forbidden not in source
