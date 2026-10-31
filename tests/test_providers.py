"""Provider contract completeness, CLI persistence and R3 admission."""
import json

import pytest
from sqlmodel import Session

from kognita.cli import main
from kognita.db import create_all, make_engine
from kognita.providers import ProviderRecord, _DETAILS, _register, _registered, _view


def _details():
    return {**{key: "Recorded contract terms" for key in _DETAILS},
            "use_cases": ["COLLABORATION"], "sub_processors": [], "training_use": False,
            "incident_history": [{"date": "2026-01-01", "summary": "Resolved outage"}],
            "regulatory_cooperation": "Cooperate with supervisory requests"}


def test_contract_and_incident_history_round_trip(tmp_path):
    path = tmp_path / "providers.db"
    engine = make_engine(path)
    create_all(engine)
    with Session(engine) as session:
        _register(session, "api.openai.com", _details())
        session.commit()
    engine.dispose()
    engine = make_engine(path)
    with Session(engine) as session:
        assert _view(session.get(ProviderRecord, "api.openai.com")) == {
            "provider": "api.openai.com", **_details()}
        assert _registered(session, "api.openai.com")
        assert not _registered(session, "other")
    engine.dispose()


@pytest.mark.parametrize("field", [*_DETAILS, "incident_history", "regulatory_cooperation"])
def test_incomplete_record_refused(session, field):
    details = _details()
    del details[field]
    with pytest.raises(ValueError, match="R3-UNREGISTERED"):
        _register(session, "provider", details)
    assert not _registered(session, "provider")


def test_duplicate_and_malformed_record_refused(session):
    _register(session, "provider", _details())
    with pytest.raises(ValueError, match="R3-UNREGISTERED"):
        _register(session, "provider", _details())
    session.add(ProviderRecord(provider="malformed", _record={}))
    session.flush()
    assert not _registered(session, "malformed")


def test_cli_register_list_show(tmp_path, capsys):
    path = tmp_path / "provider.json"
    record = {"provider": "api.openai.com", **_details()}
    path.write_text(json.dumps(record))
    db = str(tmp_path / "cli.db")
    assert main(["provider", "register", str(path), "--db", db]) == 0
    assert json.loads(capsys.readouterr().out) == record
    assert main(["provider", "list", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out) == [record]
    assert main(["provider", "show", "api.openai.com", "--db", db]) == 0
    assert json.loads(capsys.readouterr().out) == record
    assert main(["provider", "show", "unknown", "--db", db]) == 1
    assert "R3-UNREGISTERED" in capsys.readouterr().err
