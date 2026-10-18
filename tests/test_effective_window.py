"""Closed history is not what a decision loads.

``in_force`` uses the same half-open window as ``is_effective``. A row that
has closed, including one whose ``effective_to`` is exactly ``at``, is absent.
A row that is open at ``at`` is present. The decision loaders use that read.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from kognita.breaker import load_drill_dues
from kognita.db import create_all, make_engine
from kognita.governance import load_snapshot
from kognita.models import (
    BreakerThreshold,
    DrillDue,
    ModelCard,
    Policy,
    Revocation,
    RiskAppetite,
    UseCase,
    in_force,
)
from kognita.revocation import load_revocations
from kognita.usecases import load_register

AT = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def _windows() -> list[tuple[str, datetime, datetime | None]]:
    return [
        ("open", AT - timedelta(days=10), None),
        ("closed-before", AT - timedelta(days=10), AT - timedelta(days=1)),
        ("closes-at", AT - timedelta(days=10), AT),
        ("closes-after", AT - timedelta(days=10), AT + timedelta(days=1)),
        ("starts-after", AT + timedelta(seconds=1), None),
        ("starts-at", AT, None),
    ]


def test_in_force_matches_is_effective_on_every_dated_table() -> None:
    engine = make_engine(":memory:")
    create_all(engine)
    with Session(engine) as session:
        for name, start, end in _windows():
            session.add(
                Policy(
                    regime="INTERNAL",
                    rule_type="PROHIBITED",
                    citation=name,
                    effective_from=start,
                    effective_to=end,
                )
            )
            session.add(
                Revocation(
                    dimension="model",
                    value=name,
                    citation=name,
                    effective_from=start,
                    effective_to=end,
                )
            )
            session.add(
                DrillDue(
                    usecase_id=name,
                    next_due=AT,
                    effective_from=start,
                    effective_to=end,
                )
            )
            session.add(
                UseCase(
                    usecase_id=name,
                    version=1,
                    record={"usecase_id": name},
                    effective_from=start,
                    effective_to=end,
                )
            )
            session.add(
                ModelCard(
                    card_id=name,
                    version=1,
                    record={"card_id": name},
                    effective_from=start,
                    effective_to=end,
                )
            )
            session.add(
                RiskAppetite(
                    version=1,
                    record={"version": 1},
                    effective_from=start,
                    effective_to=end,
                )
            )
            session.add(
                BreakerThreshold(
                    dimension="model",
                    value=name,
                    threshold=1,
                    window_seconds=60,
                    citation=name,
                    effective_from=start,
                    effective_to=end,
                )
            )
        session.commit()
        for model in (
            Policy,
            Revocation,
            DrillDue,
            UseCase,
            ModelCard,
            RiskAppetite,
            BreakerThreshold,
        ):
            loaded = in_force(session, model, AT)
            expected = [
                row
                for row in session.exec(select(model)).all()
                if row.is_effective(AT)
            ]
            assert {row.id for row in loaded} == {row.id for row in expected}
            assert {row.id for row in loaded} == {
                row.id
                for row in session.exec(select(model)).all()
                if row.is_effective(AT)
                and getattr(row, "citation", None) != "closes-at"
            }
            # The boundary row is closed. It must not be in the loaded set.
            assert all(
                getattr(row, "citation", None) != "closes-at"
                and getattr(row, "value", None) != "closes-at"
                and getattr(row, "usecase_id", None) != "closes-at"
                and getattr(row, "card_id", None) != "closes-at"
                for row in loaded
            )


def test_decision_loaders_skip_closed_rows() -> None:
    """An open revocation is loaded. A closed breaker, drill, and use case are not."""
    engine = make_engine(":memory:")
    create_all(engine)
    with Session(engine) as session:
        session.add(
            Revocation(
                dimension="model",
                value="gateway-model",
                citation="open",
                actor_id="owner",
                accountable_owners=["owner"],
                effective_from=AT - timedelta(days=1),
            )
        )
        session.add(
            Revocation(
                dimension="model",
                value="gateway-model",
                citation="closed",
                actor_id="owner",
                accountable_owners=["owner"],
                effective_from=AT - timedelta(days=10),
                effective_to=AT - timedelta(days=1),
            )
        )
        session.add(
            Policy(
                regime="INTERNAL",
                rule_type="CIRCUIT_BREAKER",
                rule={"dimension": "model", "value": "gateway-model", "on_violation": "fail"},
                citation="closed breaker",
                effective_from=AT - timedelta(days=10),
                effective_to=AT,
            )
        )
        session.add(
            DrillDue(
                usecase_id="COLLABORATION",
                next_due=AT - timedelta(days=1),
                effective_from=AT - timedelta(days=10),
                effective_to=AT,
            )
        )
        session.add(
            UseCase(
                usecase_id="COLLABORATION",
                version=1,
                record={"usecase_id": "COLLABORATION", "status": "retired"},
                effective_from=AT - timedelta(days=10),
                effective_to=AT,
            )
        )
        session.commit()
        revocations = load_revocations(session, AT)
        assert len(revocations) == 1
        assert revocations[0].citation == "open"
        snapshot = load_snapshot(session, as_of=AT)
        assert snapshot.policies == ()
        assert snapshot.drill_dues == ()
        assert snapshot.revocations == revocations
        assert load_register(session, as_of=AT).entries == ()
