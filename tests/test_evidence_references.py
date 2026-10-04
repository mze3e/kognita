"""Evidence payload links are foreign keys, and SQLite enforces them.

``approvals.decision_id`` and ``entity_edges.from_entity_id`` /
``to_entity_id`` already reference their parents. Those constraints are left
as they are. The ids an evidence payload cites are the ones this file adds:
a dangling insert is rejected on the engine ``make_engine`` builds, which is
the connection the app uses.
"""
from __future__ import annotations

import sqlalchemy
import pytest
from sqlalchemy import inspect
from sqlmodel import Session, select

from fixtures import demo_pack as dp

from kognita.evidence import EvidenceWriter, hashes_only, verify_chain
from kognita.models import (
    Approval,
    Continuation,
    Entity,
    EntityEdge,
    EvidenceCheck,
    EvidenceEvent,
    EvidenceItem,
    GovernanceDecision,
    KnowledgeItem,
    Policy,
    RunRecord,
)
from kognita.registry import register
from kognita.testing.harness import FROZEN_NOW, Harness
from kognita.vocabulary import EventType, Outcome


@pytest.fixture
def harness():
    def seed(session):
        dp.seed_policies(session, now=FROZEN_NOW)
        register(session, name="eligibility-assistant", owner_exec="Head of Data Governance")

    return Harness(pack=dp.DemoPack(), purposes=dp.PURPOSES, seed=seed)


def _pragma(session: Session) -> int:
    return session.connection().exec_driver_sql("PRAGMA foreign_keys").scalar()


def test_the_app_connection_enforces_foreign_keys(engine):
    """``make_engine`` is the connection the library opens. Enforcement is on."""
    with Session(engine) as session:
        assert _pragma(session) == 1


def test_existing_reference_constraints_are_not_duplicated(engine):
    """The columns that already had a foreign key still have exactly one."""
    inspector = inspect(engine)
    approvals = inspector.get_foreign_keys("approvals")
    edges = inspector.get_foreign_keys("entity_edges")
    assert [fk["constrained_columns"] for fk in approvals].count(["decision_id"]) == 1
    assert [fk["constrained_columns"] for fk in approvals].count(["proposal_id"]) == 1
    assert [fk["constrained_columns"] for fk in edges].count(["from_entity_id"]) == 1
    assert [fk["constrained_columns"] for fk in edges].count(["to_entity_id"]) == 1


def test_dangling_decision_and_entity_references_are_rejected(engine):
    """A missing parent for the constraints that already existed is still an error."""
    with Session(engine) as session:
        session.add(Approval(decision_id=999_999, envelope_hash="missing"))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()

    with Session(engine) as session:
        session.add(EntityEdge(from_entity_id=1, to_entity_id=2, type="RELATED"))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()

    with Session(engine) as session:
        origin = Entity(type="subject", ref_id="1", label="one")
        session.add(origin)
        session.flush()
        session.add(
            EntityEdge(from_entity_id=origin.id, to_entity_id=999_999, type="RELATED")
        )
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()

    with Session(engine) as session:
        origin = Entity(type="subject", ref_id="1", label="one")
        target = Entity(type="subject", ref_id="2", label="two")
        session.add(origin)
        session.add(target)
        session.flush()
        session.add(
            EntityEdge(
                from_entity_id=origin.id,
                to_entity_id=target.id,
                type="RELATED",
            )
        )
        session.commit()
        stored = session.exec(select(EntityEdge)).one()
        assert stored.from_entity_id == origin.id
        assert stored.to_entity_id == target.id


def test_inserting_a_dangling_evidence_column_is_rejected(engine):
    """Declaring the column is not the check. The insert itself has to fail."""
    with Session(engine) as session:
        assert _pragma(session) == 1
        session.add(
            EvidenceEvent(
                sequence=1,
                correlation_id="dangling-column",
                event_type=EventType.APPROVAL,
                approval_id=999_999,
            )
        )
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()

    with Session(engine) as session:
        session.add(RunRecord(id="run-missing", continuation_hash="ab" * 32))
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            session.commit()


def test_writer_rejects_a_dangling_payload_reference(engine):
    """The ids ``emit`` copies out of the payload are enforced by SQLite."""
    writer = EvidenceWriter(engine)

    def rejected(payload):
        with Session(engine) as session:
            with pytest.raises(sqlalchemy.exc.IntegrityError):
                writer.emit(
                    session,
                    correlation_id="dangling",
                    event_type=EventType.POLICY_DECISION,
                    payload=payload,
                )

    rejected({"approval_id": 999_999})
    rejected({"policy_id": 999_999})
    rejected({"successor_id": 999_999})
    rejected({"checks": [{"policy_id": 999_999}]})
    rejected({"checks": [{"policy_id": None}, {"policy_id": 999_999}]})
    rejected({"returned_ids": [999_999]})
    rejected({"items": [{"id": 999_999}]})
    rejected({"run_id": "no-such-run"})
    rejected({"continuation_hash": "cd" * 32})


def test_writer_stores_payload_links_that_exist(engine):
    """A cited row is kept, and a null policy id in a check is not a citation."""
    writer = EvidenceWriter(engine)
    with Session(engine) as session:
        policy = Policy(regime="INTERNAL", rule_type="ALLOW", citation="cited")
        other = Policy(regime="INTERNAL", rule_type="DENY", citation="other")
        decision = GovernanceDecision(
            request_id="req-1",
            outcome=Outcome.ALLOW,
            envelope_hash="env",
        )
        item = KnowledgeItem(title="note", body="text")
        session.add(policy)
        session.add(other)
        session.add(decision)
        session.add(item)
        session.flush()
        approval = Approval(decision_id=decision.id, envelope_hash="env")
        digest = "ef" * 32
        session.add(Continuation(content_hash=digest, payload={"kept": True}))
        session.add(RunRecord(id="run-1"))
        session.add(approval)
        session.flush()
        run = session.get(RunRecord, "run-1")
        run.continuation_hash = digest
        session.add(run)
        session.flush()

        event = writer.emit(
            session,
            correlation_id="req-1",
            event_type=EventType.POLICY_DECISION,
            payload={
                "approval_id": approval.id,
                "policy_id": policy.id,
                "successor_id": other.id,
                "run_id": "run-1",
                "continuation_hash": digest,
                "checks": [
                    {"policy_id": policy.id},
                    {"policy_id": other.id},
                    {"policy_id": None},
                    {"policy_id": policy.id},
                ],
                "returned_ids": [item.id, item.id],
                "items": [{"id": item.id, "content_hash": "abc"}],
            },
        )
        session.commit()
        assert event.approval_id == approval.id
        assert event.policy_id == policy.id
        assert event.successor_id == other.id
        assert event.run_id == "run-1"
        assert event.continuation_hash == digest
        cited = {row.policy_id for row in session.exec(select(EvidenceCheck)).all()}
        assert cited == {policy.id, other.id}
        stored_items = session.exec(select(EvidenceItem)).all()
        assert [row.item_id for row in stored_items] == [item.id]
        assert verify_chain(session) == 1


def test_a_redacted_payload_does_not_treat_a_hash_as_a_row_id(engine):
    """``hashes_only`` replaces each value with a digest. That digest is not a key."""
    writer = EvidenceWriter(engine, redact_payload=hashes_only)
    with Session(engine) as session:
        event = writer.emit(
            session,
            correlation_id="redacted",
            event_type=EventType.APPROVAL,
            payload={"approval_id": 1, "action": "OPENED"},
        )
        session.commit()
        assert event.approval_id is None
        assert session.exec(select(EvidenceCheck)).all() == []


def test_deleting_an_event_removes_its_citations(engine):
    """The chain-break test deletes an event. The citation goes with it."""
    writer = EvidenceWriter(engine)
    with Session(engine) as session:
        policy = Policy(regime="INTERNAL", rule_type="ALLOW", citation="cited")
        session.add(policy)
        session.flush()
        event = writer.emit(
            session,
            correlation_id="gone",
            event_type=EventType.POLICY_DECISION,
            payload={"checks": [{"policy_id": policy.id}]},
        )
        session.commit()
        session.delete(event)
        session.commit()
        assert session.exec(select(EvidenceCheck)).all() == []
        assert session.exec(select(EvidenceEvent)).all() == []


def test_a_recorded_decision_cites_the_policies_it_names(harness):
    """The decision path writes the check ids, and SQLite accepts them."""
    envelope = dp.envelope("get_subject_profile", site="SG", subject="2")
    with harness.session() as session:
        evaluation = harness.decide_and_record(envelope, session)
        session.commit()
        cited = {row.policy_id for row in session.exec(select(EvidenceCheck)).all()}
        named = {
            check.policy_id
            for check in evaluation.checks
            if check.policy_id is not None
        }
        assert cited == named
        assert named
        assert verify_chain(session) >= 1
