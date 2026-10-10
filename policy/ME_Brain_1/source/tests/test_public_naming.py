"""Check the release namespace."""

import sys
import tomllib
from pathlib import Path

import pytest

from focus_vlwa import FocusVLWAConfig
from focus_vlwa.cli import main
from focus_vlwa.inference import FocusVLWAPolicy
from focus_vlwa.model.focus_vlwa import FocusVLWA

ROOT = Path(__file__).parents[1]


def test_package_and_public_names():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert project['project']['name'] == 'focus-vlwa'
    assert project['project']['scripts'] == {'focus-vlwa': 'focus_vlwa.cli:main'}
    assert project['tool']['hatch']['build']['targets']['wheel']['packages'] == ['src/focus_vlwa']
    assert FocusVLWAConfig.__name__ == 'FocusVLWAConfig'
    assert FocusVLWAPolicy.__name__ == 'FocusVLWAPolicy'
    assert FocusVLWA.__name__ == 'FocusVLWA'


def test_cli_uses_release_name(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['focus-vlwa', '--help'])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 0
    assert 'usage: focus-vlwa' in capsys.readouterr().out
