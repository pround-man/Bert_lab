from types import SimpleNamespace

import pytest

from watch_intents import cli
from watch_intents.models import DatasetError


def test_help_does_not_load_credentials_or_call_models(monkeypatch, capsys):
    monkeypatch.setattr(cli, "LLMProvider", lambda *a, **kw: pytest.fail("API constructed for help"))
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    assert "smoke" in capsys.readouterr().out


def test_error_is_sanitized_and_nonzero(monkeypatch, capsys):
    def bad_config(*args, **kwargs):
        raise ValueError("SECRET_OR_REJECTED_TEXT")
    monkeypatch.setattr(cli, "load_settings", bad_config)
    assert cli.main(["check"]) == 1
    assert "SECRET_OR_REJECTED_TEXT" not in capsys.readouterr().err


@pytest.mark.parametrize("command", ["verify", "report"])
def test_offline_commands_never_construct_api_provider(tmp_path, monkeypatch, command):
    monkeypatch.setattr(cli, "load_settings", lambda *a: SimpleNamespace(root=tmp_path, model_dir=tmp_path, max_tokens=512))
    monkeypatch.setattr(cli, "load_catalog", lambda *a: {"labels": []})
    monkeypatch.setattr(cli, "TextInspector", lambda *a: object())
    monkeypatch.setattr(cli, "LLMProvider", lambda *a, **kw: pytest.fail("Offline command accessed API"))
    if command == "verify":
        monkeypatch.setattr(cli, "verify_dataset", lambda *a: {"stage": "pilot", "ok": False, "accepted": 0, "expected": 300, "errors": [1]})
    else:
        monkeypatch.setattr(cli, "plan_slots", lambda *a: [])
        monkeypatch.setattr(cli, "Store", lambda *a, **kw: (_ for _ in ()).throw(DatasetError("No dataset")))
    monkeypatch.setattr(cli, "state_directory", lambda *a: tmp_path)
    assert cli.main([command, "--stage", "pilot"]) in (1, 2)


def test_smoke_disables_network_retries_and_has_no_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_settings", lambda *a: SimpleNamespace(root=tmp_path, model_dir=tmp_path, max_tokens=512))
    monkeypatch.setattr(cli, "load_catalog", lambda *a: {"labels": []})
    monkeypatch.setattr(cli, "TextInspector", lambda *a: object())
    options = {}
    def provider(*args, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli, "LLMProvider", provider)
    def run(*args, **kwargs):
        assert args[-1] == "smoke" and kwargs["resume"] is False
        return {"stage": "smoke", "accepted": 8, "expected": 9, "missing": 1, "complete": False}
    monkeypatch.setattr(cli, "run_build", run)
    monkeypatch.setattr(cli, "state_directory", lambda *a: tmp_path)
    assert cli.main(["smoke"]) == 2
    assert options["max_retries"] == 0


@pytest.mark.parametrize("bounded", [False, True])
def test_build_defaults_to_continuous_and_supports_explicit_bounded(tmp_path, monkeypatch, bounded):
    monkeypatch.setattr(cli, "load_settings", lambda *a: SimpleNamespace(root=tmp_path, model_dir=tmp_path, max_tokens=512))
    monkeypatch.setattr(cli, "load_catalog", lambda *a: {"labels": []})
    monkeypatch.setattr(cli, "TextInspector", lambda *a: object())
    monkeypatch.setattr(cli, "LLMProvider", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    def run(*args, **kwargs):
        assert kwargs["resume"] is True
        assert kwargs["until_complete"] is not bounded
        return {"stage": "pilot", "accepted": 300, "expected": 300, "missing": 0, "complete": True}
    monkeypatch.setattr(cli, "run_build", run)
    monkeypatch.setattr(cli, "state_directory", lambda *a: tmp_path)
    argv = ["build", "--stage", "pilot", "--resume"] + (["--bounded"] if bounded else [])
    assert cli.main(argv) == 0
