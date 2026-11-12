"""CPU coverage of native SFT, launcher validation and SFT -> RL checkpoint IO."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from XPolicyLab.policy.Pi_05.rlinf import sft
from XPolicyLab.policy.Pi_05.rl import Pi05RLConfig, load_pi05_rl_policy
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models import get_model
from rlinf.models.embodiment.openpi.modules import gemma, siglip
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("PI05_SFT_ACTION_DIM", "14")
    return sft.compose_config("RoboDojo", "stack_bowls", "arx_x5", "joint", 0, "0")


@pytest.mark.parametrize(
    "override,match",
    [
        ("actor.model.openpi.task=rl", "native"),
        ("actor.fsdp_config.sharding_strategy=invalid", "no_shard"),
        ("actor.fsdp_config.strategy=fsdp2", "FSDP1"),
        ("actor.model.precision=bf16", "FP32"),
        ("actor.global_batch_size=3", "divisible"),
        ("runner.val_check_interval=1", "validation loop"),
        ("actor.model.openpi.action_horizon=4", "horizon"),
    ],
)
def test_invalid_training_configs(monkeypatch, override, match):
    monkeypatch.setenv("PI05_SFT_ACTION_DIM", "14")
    with pytest.raises(ValueError, match=match):
        sft.compose_config(
            "RoboTwin", "task", "aloha_agilex", "joint", 0, "0,1", [override]
        )


def test_shell_dry_run_without_model_dependencies(tmp_path):
    env = dict(os.environ, RL_PYTHON=sys.executable, HF_HOME=str(tmp_path))
    result = subprocess.run(
        [
            "bash",
            str(sft.ADAPTER_DIR / "scripts" / "train_pi05_sft.sh"),
            "RoboDojo",
            "stack_bowls",
            "arx_x5",
            "joint",
            "7",
            "0,2",
            "--dry-run",
            "actor.global_batch_size=4",
            "runner.max_steps=10",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "model_type: openpi" in result.stdout
    assert "actor: 0-1" in result.stdout
    assert "max_steps: 10" in result.stdout
    assert "experiment_name: RoboDojo-stack_bowls-arx_x5-joint-7" in result.stdout
    assert not list(tmp_path.iterdir())


def test_checkpoints_do_not_overlap_official_run(cfg, tmp_path):
    output = Path(cfg.runner.logger.log_path) / cfg.runner.logger.experiment_name
    jax_output = sft.POLICY_DIR / "checkpoints" / "RoboDojo-stack_bowls-arx_x5-joint-0"
    assert not output.is_relative_to(jax_output)
    stats = tmp_path / "input" / "norm_stats.json"
    stats.parent.mkdir()
    stats.write_text('{"norm_stats": {}}')
    cfg.actor.model.openpi_data.norm_stats_path = str(stats)
    sft.save_metadata(cfg, tmp_path / "actor")
    assert (tmp_path / "actor" / "norm_stats.json").read_bytes() == stats.read_bytes()
    meta = json.loads((tmp_path / "actor" / "sft_metadata.json").read_text())
    assert meta["rlinf_revision"] == sft.checked_rlinf_revision()
    assert meta["action_horizon"] == 50


def test_preflight_rejects_jax_weights_and_accidental_overwrite(
    cfg, tmp_path, monkeypatch
):
    monkeypatch.delenv("POLICY_MODEL_PATH", raising=False)
    with pytest.raises(ValueError, match="POLICY_MODEL_PATH"):
        sft.preflight(cfg)
    cfg.actor.model.model_path = str(tmp_path / "base")
    (tmp_path / "base" / "params").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="JAX"):
        sft.preflight(cfg)
    save_file(
        {"action_out_proj.weight": torch.ones(2, 2)},
        tmp_path / "base" / "model.safetensors",
    )
    monkeypatch.setattr(sft, "build_train_config", lambda *args: None)
    cfg.runner.logger.log_path = str(tmp_path / "runs")
    output = sft.preflight(cfg)
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("existing run")
    with pytest.raises(FileExistsError):
        sft.preflight(cfg)
    resume = output / "checkpoints" / "global_step_10"
    cfg.runner.resume_dir = str(resume)
    with pytest.raises(FileNotFoundError, match="dcp_checkpoint"):
        sft.preflight(cfg)
    (resume / "actor" / "dcp_checkpoint").mkdir(parents=True)
    assert sft.preflight(cfg) == output
    cfg.runner.max_steps = 10
    with pytest.raises(ValueError, match="exceed"):
        sft.preflight(cfg)


def small_architecture(monkeypatch):
    """Only shrink dimensions: all vision, attention and flow-loss code is native."""
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 256)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(
            width=32, depth=1, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16
        ),
    )
    decode = siglip._decode_variant
    monkeypatch.setattr(siglip, "_decode_variant", lambda _: decode("mu/14"))
    set_torch_compile(False)
    return Pi0Config(
        pi05=True,
        action_dim=4,
        action_horizon=3,
        max_token_len=4,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
    )


def test_native_sft_optimizer_checkpoint_and_rl_loading(cfg, monkeypatch, tmp_path):
    from XPolicyLab.infra.RLinf.register import register
    from XPolicyLab.policy.Pi_05.rlinf.sft_worker import Pi05SftWorker

    torch.manual_seed(12)
    shape = small_architecture(monkeypatch)
    source = Pi0(shape)
    # Fresh Pi0 starts with zero vision projection/adaRMS gates. Simulate
    # nonzero pretrained weights so a single full SFT update reaches every trunk.
    torch.nn.init.normal_(source.img.head.weight, std=0.02)
    for module in source.modules():
        if isinstance(module, gemma.RMSNorm) and module.adaptive:
            torch.nn.init.normal_(module.ada_modulation.weight, std=0.02)
    save_file(source.state_dict(), tmp_path / "model.safetensors")
    model_cfg = cfg.actor.model
    model_cfg.model_path = str(tmp_path)
    model_cfg.openpi.action_horizon = 3
    model_cfg.openpi.model_action_dim = 4
    model_cfg.openpi.max_token_len = 4
    model_cfg.openpi.paligemma_variant = "dummy"
    model_cfg.openpi.action_expert_variant = "dummy"
    model_cfg.openpi.embed_dtype = "fp32"  # CPU numerical test, production uses BF16
    model_cfg.num_action_chunks = 3
    model_cfg.action_dim = 4
    register()
    model = get_model(model_cfg)
    assert type(model) is Pi0
    for a, b in zip(model.parameters(), source.parameters()):
        torch.testing.assert_close(a, b)
    worker = object.__new__(Pi05SftWorker)
    worker.model = model
    worker.amp_context = nullcontext()
    observation = shape.fake_obs(1)
    observation.tokenized_prompt.fill_(1)
    actions = torch.randn(1, 3, 4)
    loss, metrics = worker.get_train_model_output((observation, actions))
    assert torch.isfinite(loss) and metrics["loss"] > 0
    loss.backward()
    # Full SFT must propagate through vision and language as well as action heads.
    for parameter in (
        model.img.stem.weight,
        model.llm.embedder.embedding.weight,
        model.action_out_proj.weight,
    ):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    before = model.action_out_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    assert not torch.equal(before, model.action_out_proj.weight)
    actor = tmp_path / "actor"
    (actor / "model_state_dict").mkdir(parents=True)
    torch.save(model.state_dict(), actor / "model_state_dict" / "full_weights.pt")
    # The real RL bridge must accept the SFT layout without a JAX round trip.
    rl = load_pi05_rl_policy(
        shape,
        actor,
        Pi05RLConfig.from_dict({"action_chunk": 3, "action_env_dim": 4}),
        output_transform=lambda x: x,
        precision="fp32",
        embed_dtype="bf16",
    )
    assert rl.embed_dtype == torch.bfloat16
    assert rl.img.stem.weight.dtype == torch.float32
    for key, value in model.state_dict().items():
        torch.testing.assert_close(rl.state_dict()[key], value)
    assert model.forward.__func__ is Pi0.forward
    with pytest.raises(ValueError, match="padded"):
        model(forward_type=ForwardType.SFT, data=(observation, torch.ones(1, 3, 2)))
