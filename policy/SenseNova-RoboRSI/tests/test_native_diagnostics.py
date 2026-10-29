"""Native diagnostic opt-in and cleanup, without simulator or watchdog threads."""

from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest


native = import_module("XPolicyLab.policy.SenseNova-RoboRSI.native_entry")


@pytest.fixture
def launch(tmp_path, monkeypatch):
    for key in os.environ:
        if key.startswith(("ROBORSI_", "ROBODOJO_")):
            monkeypatch.delenv(key)
    state = tmp_path / "state"
    sink = state / "native-infrastructure"
    sink.mkdir(parents=True)
    monkeypatch.setenv("ROBORSI_RUNTIME_STATE_DIR", str(state))
    monkeypatch.setenv("ROBORSI_NATIVE_DIAGNOSTICS_DIR", str(sink))
    entry = tmp_path / "main.py"
    entry.write_text("# Official entry fixture\n")
    run = Mock()
    dump = Mock()
    cancel = Mock()
    monkeypatch.setattr(native.runpy, "run_path", run)
    monkeypatch.setattr(native.faulthandler, "dump_traceback_later", dump)
    monkeypatch.setattr(native.faulthandler, "cancel_dump_traceback_later", cancel)
    arguments = ["--official-entry", str(entry), "--", "--task_name", "fasten_screws", "--seed", "3"]
    return arguments, entry, sink, run, dump, cancel


@pytest.mark.parametrize("switch", [None, "0"])
def test_directory_alone_never_starts_periodic_dumps(launch, monkeypatch, switch):
    arguments, entry, sink, run, dump, cancel = launch
    if switch is not None:
        monkeypatch.setenv("ROBORSI_NATIVE_PERIODIC_TRACEBACK", switch)
    native.main(arguments)
    run.assert_called_once_with(str(entry), run_name="__main__")
    dump.assert_not_called()
    cancel.assert_not_called()
    assert list(sink.iterdir()) == []


@pytest.mark.parametrize("switch", ["", "true", "false", "2"])
def test_invalid_periodic_switch_fails_before_official_entry(launch, monkeypatch, switch):
    arguments, _, sink, run, dump, cancel = launch
    monkeypatch.setenv("ROBORSI_NATIVE_PERIODIC_TRACEBACK", switch)
    with pytest.raises(ValueError, match="ROBORSI_NATIVE_PERIODIC_TRACEBACK must be 0 or 1"):
        native.main(arguments)
    run.assert_not_called()
    dump.assert_not_called()
    cancel.assert_not_called()
    assert list(sink.iterdir()) == []


def test_explicit_periodic_mode_requires_output_directory(launch, monkeypatch):
    arguments, _, _, run, dump, cancel = launch
    monkeypatch.setenv("ROBORSI_NATIVE_PERIODIC_TRACEBACK", "1")
    monkeypatch.delenv("ROBORSI_NATIVE_DIAGNOSTICS_DIR")
    with pytest.raises(ValueError, match="require ROBORSI_NATIVE_DIAGNOSTICS_DIR"):
        native.main(arguments)
    run.assert_not_called()
    dump.assert_not_called()
    cancel.assert_not_called()


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_periodic_mode_cancels_before_closing_file(launch, monkeypatch, failure):
    arguments, _, sink, run, dump, cancel = launch
    monkeypatch.setenv("ROBORSI_NATIVE_PERIODIC_TRACEBACK", "1")
    streams = []
    events = []

    def register(timeout, *, repeat, file, exit):
        assert (timeout, repeat, exit) == (120, True, False)
        assert not file.closed
        assert os.fstat(file.fileno()).st_mode & 0o777 == 0o600
        streams.append(file)
        events.append("register")

    def execute(*args, **kwargs):
        assert not streams[0].closed
        events.append("official_entry")
        if failure:
            raise RuntimeError("fixture failure")

    def stop():
        assert not streams[0].closed
        events.append("cancel")

    dump.side_effect = register
    run.side_effect = execute
    cancel.side_effect = stop
    before = (sys.argv, list(sys.path), list(sys.meta_path), os.execv)
    if failure:
        with pytest.raises(RuntimeError, match="fixture failure"):
            native.main(arguments)
    else:
        native.main(arguments)
    assert events == ["register", "official_entry", "cancel"]
    assert streams[0].closed
    assert len(list(sink.iterdir())) == 1
    assert (sys.argv, sys.path, sys.meta_path, os.execv) == before


def test_default_mode_preserves_official_fatal_handler(tmp_path):
    """Run in a child so the test runner's own signal handlers are untouched."""
    sink = tmp_path / "state/native-infrastructure"
    sink.mkdir(parents=True)
    marker = tmp_path / "official-entry-ran"
    entry = tmp_path / "main.py"
    entry.write_text(
        "import faulthandler\n"
        "from pathlib import Path\n"
        "faulthandler.enable(all_threads=True)\n"
        f"Path({str(marker)!r}).write_text('ready')\n"
    )
    program = (
        "import faulthandler, sys\n"
        "from unittest.mock import patch\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from native_entry import main\n"
        "with patch.object(faulthandler, 'dump_traceback_later') as dump, "
        "patch.object(faulthandler, 'disable') as disable:\n"
        "    main(['--official-entry', sys.argv[2], '--', '--task_name', 'fasten_screws', '--seed', '3'])\n"
        "    dump.assert_not_called()\n"
        "    disable.assert_not_called()\n"
        "    assert faulthandler.is_enabled()\n"
    )
    environ = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("ROBORSI_", "ROBODOJO_"))
    }
    environ.update(ROBORSI_RUNTIME_STATE_DIR=str(sink.parent),
                   ROBORSI_NATIVE_DIAGNOSTICS_DIR=str(sink))
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(Path(native.__file__).parent), str(entry)],
        env=environ, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "ready"
    assert list(sink.iterdir()) == []
