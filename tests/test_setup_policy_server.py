import sys

from setup_policy_server import parse_args_and_config


def test_parse_args_accepts_robodojo_ws_cli_options(tmp_path, monkeypatch):
    config_path = tmp_path / "deploy.yml"
    config_path.write_text(
        "policy_name: demo_policy\nport: 9999\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup_policy_server.py",
            "--config-path",
            str(config_path),
            "--protocol",
            "robodojo_ws",
            "--host",
            "0.0.0.0",
            "--port",
            "19000",
            "--relay-url",
            "ws://relay.example",
        ],
    )

    cfg = parse_args_and_config()

    assert cfg["policy_name"] == "demo_policy"
    assert cfg["protocol"] == "robodojo_ws"
    assert cfg["host"] == "0.0.0.0"
    assert cfg["port"] == 19000
    assert cfg["relay_url"] == "ws://relay.example"
