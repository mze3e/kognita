"""Policy files, change deltas, and colocated scenarios.

The scenario text below is the shape in the 0.4 roadmap. The rule that makes
those two cases pass is an ATTRIBUTE_DENYLIST of the other client's id: the
current rule types have no "subject is the principal" condition.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kognita.cli import main
from kognita.governance import decide, load_snapshot
from kognita.envelope import Envelope
from kognita.policy import (
    PolicyLanguageError,
    apply_change,
    blocking_findings,
    compile_rules,
    diff_sets,
    explain,
    merge_changes,
    parse_change,
    read_change,
    read_policy_set,
    render_change,
    render_explanation,
    run_scenarios,
    validate,
)
from kognita.vocabulary import Outcome

AT = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
APPLIED = datetime(2026, 10, 18, 9, 0, tzinfo=timezone.utc)

ROADMAP_SCENARIOS = """\
rule: client-email-isolation
scenarios:
  - name: client can read own email
    given: { principal: "alice@corp", subject_type: client, subject_id: alice }
    when:  { tool: read_email }
    then:  { decision: ALLOW }

  - name: client cannot read another client's email
    given: { principal: "alice@corp", subject_type: client, subject_id: bob }
    when:  { tool: read_email }
    then:  { decision: DENY, cites: client-email-isolation }
"""

RULE = """\
id: client-email-isolation
regime: CLIENT
rule_type: ATTRIBUTE_DENYLIST
applies_to: client
citation: A client may read only their own mailbox
effective_from: "2020-01-01T00:00:00+00:00"
rule:
  deny:
    subject_id:
      - bob
  on_violation: fail
"""


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _policy_dir(
    tmp_path: Path, rule: str = RULE, scenarios: str | None = ROADMAP_SCENARIOS
) -> Path:
    directory = tmp_path / "policies"
    _write(directory, "client-email-isolation.yaml", rule)
    if scenarios is not None:
        _write(directory, "client-email-isolation.scenarios.yaml", scenarios)
    return directory


def _codes(findings, *, strict: bool) -> set[str]:
    return {item.code for item in blocking_findings(findings, strict=strict)}


def test_roadmap_scenario_shape_parses_and_passes(tmp_path):
    policies = read_policy_set(_policy_dir(tmp_path))
    assert [rule.id for rule in policies.rules] == ["client-email-isolation"]
    assert policies.scenarios[0].scenarios[0].given["principal"] == "alice@corp"
    assert policies.scenarios[0].scenarios[1].then["cites"] == "client-email-isolation"
    results = run_scenarios(policies, as_of=AT)
    assert [(item.name, item.passed, item.detail) for item in results] == [
        ("client can read own email", True, "ALLOW"),
        ("client cannot read another client's email", True, "DENY"),
    ]


def test_a_scenario_failure_names_the_outcome(tmp_path):
    text = ROADMAP_SCENARIOS.replace("decision: DENY", "decision: ALLOW", 1)
    results = run_scenarios(
        read_policy_set(_policy_dir(tmp_path, scenarios=text)), as_of=AT
    )
    failed = [item for item in results if not item.passed]
    assert len(failed) == 1
    assert "expected ALLOW, got DENY" in failed[0].detail


def test_a_decision_without_a_citation_fails_the_scenario(tmp_path):
    rule = RULE.replace(
        "citation: A client may read only their own mailbox",
        'citation: ""',
    )
    results = run_scenarios(read_policy_set(_policy_dir(tmp_path, rule=rule)), as_of=AT)
    assert results[1].passed is False
    assert "no citation" in results[1].detail


def test_cites_must_name_a_rule_in_the_basis(tmp_path):
    text = ROADMAP_SCENARIOS.replace(
        "cites: client-email-isolation", "cites: some-other-rule"
    )
    # The dangling id is a strict validation problem. The scenario runner still
    # reports that the denial did not cite it. Give the id a rule so validation
    # is not what this test is about: the cited rule passes, so it is not the basis.
    extra = RULE.replace("id: client-email-isolation", "id: some-other-rule").replace(
        "      - bob", "      - nobody"
    )
    directory = _policy_dir(tmp_path, scenarios=text)
    _write(directory, "some-other-rule.yaml", extra)
    _write(
        directory,
        "some-other-rule.scenarios.yaml",
        "rule: some-other-rule\nscenarios: []\n",
    )
    results = run_scenarios(read_policy_set(directory), as_of=AT)
    denial = next(item for item in results if item.name.startswith("client cannot"))
    assert denial.passed is False
    assert "cite some-other-rule" in denial.detail


def test_strict_fails_uncited_untested_dangling_and_clarification(tmp_path):
    rule = RULE.replace(
        "citation: A client may read only their own mailbox",
        'citation: ""',
    ).replace(
        "on_violation: fail",
        'on_violation: fail\n  description: "[NEEDS CLARIFICATION: which mailbox]"',
    )
    directory = _policy_dir(tmp_path, rule=rule, scenarios=None)
    _write(
        directory,
        "missing.scenarios.yaml",
        "rule: missing-rule\nscenarios:\n"
        "  - name: ghost\n"
        "    given: { principal: alice }\n"
        "    when: { tool: read_email }\n"
        "    then: { decision: DENY, cites: also-missing }\n",
    )
    findings = validate(read_policy_set(directory))
    assert _codes(findings, strict=False) == set()
    assert _codes(findings, strict=True) == {
        "uncited",
        "untested",
        "dangling",
        "clarification",
    }
    questions = [item.message for item in findings if item.code == "clarification"]
    assert any("which mailbox" in message for message in questions)


def test_an_unclosed_clarification_marker_is_unresolved(tmp_path):
    rule = RULE.replace(
        "citation: A client may read only their own mailbox",
        'citation: "before [NEEDS CLARIFICATION: forgotten"',
    )
    findings = validate(read_policy_set(_policy_dir(tmp_path, rule=rule)))
    assert "clarification" in _codes(findings, strict=True)


def test_modified_and_removed_must_name_a_current_rule(tmp_path):
    directory = _policy_dir(tmp_path)
    change = tmp_path / "change.md"
    change.write_text(
        "## MODIFIED\n\n- id: missing-rule\n  regime: CLIENT\n"
        "  rule_type: PROHIBITED\n  citation: gone\n"
        '  effective_from: "2020-01-01T00:00:00+00:00"\n  rule: {}\n\n'
        "## REMOVED\n\n- id: also-missing\n",
        encoding="utf-8",
    )
    findings = validate(read_policy_set(directory), read_change(change))
    assert _codes(findings, strict=False) == {"referential"}
    messages = " ".join(item.message for item in findings)
    assert "MODIFIED" in messages
    assert "REMOVED" in messages


def test_added_id_that_already_exists_is_a_referential_error(tmp_path):
    directory = _policy_dir(tmp_path)
    change = parse_change(
        "## ADDED\n\n" + RULE,
        source="change.md",
    )
    findings = validate(read_policy_set(directory), change)
    assert any(
        item.rule_id == "client-email-isolation" and item.code == "referential"
        for item in findings
    )


def test_unknown_rule_type_and_unknown_field_fail_closed(tmp_path):
    broken = RULE.replace("ATTRIBUTE_DENYLIST", "CUSTOM")
    with pytest.raises(PolicyLanguageError) as excinfo:
        read_policy_set(_policy_dir(tmp_path, rule=broken))
    assert any("unknown rule_type" in item.message for item in excinfo.value.findings)

    extra = RULE.replace("on_violation: fail", "on_violation: fail\n  surprise: 1")
    with pytest.raises(PolicyLanguageError) as excinfo:
        read_policy_set(_policy_dir(tmp_path, rule=extra))
    assert any("unknown rule field" in item.message for item in excinfo.value.findings)


def test_tabs_and_duplicate_keys_are_parse_errors(tmp_path):
    path = _write(tmp_path, "tabs.yaml", "id:\tbad\n")
    with pytest.raises(PolicyLanguageError) as excinfo:
        read_policy_set(path)
    assert excinfo.value.findings[0].code == "parse"

    path = _write(tmp_path, "dup.yaml", "id: one\nid: two\n")
    with pytest.raises(PolicyLanguageError) as excinfo:
        read_policy_set(path)
    assert "duplicate key" in excinfo.value.findings[0].message


def test_diff_renders_added_modified_and_removed(tmp_path):
    before = _policy_dir(tmp_path / "before")
    after_dir = _policy_dir(tmp_path / "after")
    updated = RULE.replace(
        "A client may read only their own mailbox",
        "A client may read only their own mailbox today",
    )
    _write(after_dir, "client-email-isolation.yaml", updated)
    _write(
        after_dir,
        "geo.yaml",
        """\
id: geo-fence
regime: CLIENT
rule_type: ATTRIBUTE_ALLOWLIST
citation: Staff act from SG or HK
effective_from: "2020-01-01T00:00:00+00:00"
rule:
  allow:
    actor_location:
      - SG
      - HK
  on_violation: fail
""",
    )
    # The original id is modified and a new id is added. Remove nothing from
    # `after`, so build the removed case by diffing the other way as well.
    change = diff_sets(read_policy_set(before), read_policy_set(after_dir))
    text = render_change(change)
    assert text.startswith("## ADDED\n")
    assert "\n## MODIFIED\n" in text
    assert "\n## REMOVED\n" in text
    assert "geo-fence" in text.split("## MODIFIED")[0]
    assert "today" in text.split("## MODIFIED")[1]

    removed = diff_sets(read_policy_set(after_dir), read_policy_set(before))
    removed_text = render_change(removed)
    assert "geo-fence" in removed_text.split("## REMOVED")[1]
    parsed = parse_change(removed_text, source="diff.md")
    assert parsed.removed == ("geo-fence",)


def test_applying_a_rendered_diff_reproduces_the_later_set(tmp_path):
    before = read_policy_set(_policy_dir(tmp_path / "before"))
    after_dir = _policy_dir(tmp_path / "after")
    _write(
        after_dir,
        "geo.yaml",
        """\
id: geo-fence
regime: CLIENT
rule_type: PROHIBITED
citation: Closed
effective_from: "2020-01-01T00:00:00+00:00"
rule: {}
""",
    )
    after = read_policy_set(after_dir)
    merged = apply_change(before, parse_change(render_change(diff_sets(before, after))))
    assert [rule.id for rule in merged.rules] == [rule.id for rule in after.rules]
    assert {rule.id: rule.citation for rule in merged.rules} == {
        rule.id: rule.citation for rule in after.rules
    }


def test_disjoint_changes_merge_and_overlapping_changes_conflict(tmp_path):
    base = read_policy_set(_policy_dir(tmp_path))
    added = parse_change(
        """\
## ADDED

- id: geo-fence
  regime: CLIENT
  rule_type: PROHIBITED
  citation: Closed
  effective_from: "2020-01-01T00:00:00+00:00"
  rule: {}
""",
        source="add.md",
    )
    modified = parse_change(
        "## MODIFIED\n\n"
        + RULE.replace(
            "A client may read only their own mailbox",
            "Mailbox isolation",
        ),
        source="modify.md",
    )
    merged = merge_changes(added, modified)
    once = apply_change(apply_change(base, added), modified)
    together = apply_change(base, merged)
    assert [rule.id for rule in together.rules] == [rule.id for rule in once.rules]
    assert {rule.id: rule.citation for rule in together.rules} == {
        rule.id: rule.citation for rule in once.rules
    }

    other = parse_change(
        "## MODIFIED\n\n"
        + RULE.replace(
            "A client may read only their own mailbox",
            "A different citation",
        ),
        source="other.md",
    )
    with pytest.raises(PolicyLanguageError) as excinfo:
        merge_changes(modified, other)
    assert excinfo.value.findings[0].code == "conflict"
    assert excinfo.value.findings[0].rule_id == "client-email-isolation"


def test_apply_archives_the_change_with_its_date_and_writes_the_rule(tmp_path):
    directory = _policy_dir(tmp_path)
    change = parse_change(
        """\
## ADDED

```yaml
- id: geo-fence
  regime: CLIENT
  rule_type: PROHIBITED
  citation: Closed
  effective_from: "2020-01-01T00:00:00+00:00"
  rule: {}
```
""",
        source=str(tmp_path / "open-geo.md"),
    )
    # The new rule needs a scenario before a strict load. apply_change itself
    # checks referential integrity. The archive is what this test is about.
    _write(
        directory,
        "geo-fence.scenarios.yaml",
        "rule: geo-fence\nscenarios:\n"
        "  - name: prohibited\n"
        "    given: { principal: ada }\n"
        "    when: { tool: read_email }\n"
        "    then: { decision: DENY, cites: geo-fence }\n",
    )
    current = read_policy_set(directory)
    apply_change(current, change, directory=directory, when=APPLIED)
    archived = list((directory / "archive").iterdir())
    assert len(archived) == 1
    assert archived[0].name.startswith("2026-10-18-")
    assert "geo-fence" in archived[0].read_text(encoding="utf-8")
    written = (directory / "geo-fence.yaml").read_text(encoding="utf-8")
    assert "geo-fence" in written
    assert "PROHIBITED" in written
    reread = read_policy_set(directory)
    assert {rule.id for rule in reread.rules} == {"client-email-isolation", "geo-fence"}


def test_explain_names_the_rule_and_its_citation(tmp_path):
    policies = read_policy_set(_policy_dir(tmp_path))
    envelope = Envelope(
        principal="alice@corp",
        purpose="READ",
        tool="read_email",
        subject_type="client",
        subject_id="bob",
    )
    report = explain(policies, envelope, purposes=("READ",), as_of=AT)
    text = render_explanation(report)
    assert report.outcome is Outcome.DENY
    assert "engaged client-email-isolation" in text
    assert "citation: A client may read only their own mailbox" in text
    assert "so decide() evaluated it" in text
    engaged = report.rules[0]
    assert engaged.checks
    assert all(check.citation.strip() for check in engaged.checks)
    assert all(check.citation.strip() for check in report.other_checks)


def test_explain_reports_a_rule_outside_its_window_as_not_engaged(tmp_path):
    rule = RULE.replace("2020-01-01T00:00:00+00:00", "2099-01-01T00:00:00+00:00")
    policies = read_policy_set(_policy_dir(tmp_path, rule=rule))
    envelope = Envelope(
        principal="alice@corp", purpose="READ", tool="read_email", subject_id="bob"
    )
    report = explain(policies, envelope, purposes=("READ",), as_of=AT)
    text = render_explanation(report)
    assert "not engaged client-email-isolation" in text
    assert "outside the effective window" in text
    assert report.outcome is Outcome.ALLOW


def test_applies_to_does_not_skip_a_rule_decide_would_evaluate(tmp_path):
    policies = read_policy_set(_policy_dir(tmp_path))
    envelope = Envelope(
        principal="alice@corp",
        purpose="READ",
        tool="read_email",
        subject_type="other",
        subject_id="bob",
    )
    report = explain(policies, envelope, purposes=("READ",), as_of=AT)
    assert report.rules[0].engaged is True
    assert report.rules[0].applies_to == "client"
    assert report.outcome is Outcome.DENY


def test_explain_fails_closed_when_the_purpose_list_is_empty(tmp_path):
    policies = read_policy_set(_policy_dir(tmp_path))
    envelope = Envelope(
        principal="alice@corp", purpose="READ", tool="read_email", subject_id="alice"
    )
    report = explain(policies, envelope, as_of=AT)
    assert report.outcome is Outcome.DENY
    purpose = next(check for check in report.other_checks if check.check == "PURPOSE")
    assert purpose.citation.strip()
    assert "Lawful-basis registry" in purpose.citation


def test_compile_inserts_rows_and_a_later_change_closes_the_old_row(tmp_path, session):
    policies = read_policy_set(_policy_dir(tmp_path))
    assert compile_rules(session, policies, at=AT) == 1
    assert compile_rules(session, policies, at=AT) == 0
    changed = read_policy_set(
        _policy_dir(
            tmp_path / "next",
            rule=RULE.replace(
                "A client may read only their own mailbox",
                "Mailbox isolation, revised",
            ),
        )
    )
    later = AT.replace(day=2)
    assert compile_rules(session, changed, at=later) == 1
    assert compile_rules(session, changed, at=later) == 0
    snapshot = load_snapshot(session, as_of=AT)
    before = decide(
        Envelope(principal="a", purpose="READ", tool="read_email", subject_id="bob"),
        snapshot,
        purposes=("READ",),
        as_of=AT,
        request_id="before",
    )
    after = decide(
        Envelope(principal="a", purpose="READ", tool="read_email", subject_id="bob"),
        load_snapshot(session, as_of=later),
        purposes=("READ",),
        as_of=later,
        request_id="after",
    )
    assert any(
        check.citation == "A client may read only their own mailbox"
        for check in before.basis()
    )
    assert any(
        check.citation == "Mailbox isolation, revised" for check in after.basis()
    )
    assert before.outcome is Outcome.DENY
    assert after.outcome is Outcome.DENY


def test_cli_validate_diff_explain_and_test(tmp_path, capsys):
    directory = _policy_dir(tmp_path)
    assert main(["policy", "validate", str(directory)]) == 0
    assert main(["policy", "validate", str(directory), "--strict"]) == 0
    assert main(["policy", "test", str(directory)]) == 0
    captured = capsys.readouterr()
    assert "PASS client-email-isolation: client can read own email" in captured.out
    assert (
        "PASS client-email-isolation: client cannot read another client's email"
        in captured.out
    )

    untested = _policy_dir(tmp_path / "bare", scenarios=None)
    assert main(["policy", "validate", str(untested)]) == 0
    assert main(["policy", "validate", str(untested), "--strict"]) == 1
    captured = capsys.readouterr()
    assert "untested" in captured.err

    other = _policy_dir(tmp_path / "other")
    _write(
        other,
        "client-email-isolation.yaml",
        RULE.replace("own mailbox", "own mailbox now"),
    )
    assert main(["policy", "diff", str(directory), str(other)]) == 0
    captured = capsys.readouterr()
    assert "## ADDED" in captured.out
    assert "## MODIFIED" in captured.out
    assert "## REMOVED" in captured.out
    assert "own mailbox now" in captured.out

    envelope = tmp_path / "envelope.json"
    envelope.write_text(
        json.dumps(
            {
                "principal": "alice@corp",
                "purpose": "READ",
                "tool": "read_email",
                "subject_type": "client",
                "subject_id": "bob",
                "purposes": ["READ"],
                "as_of": "2026-06-01T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "policy",
                "explain",
                "--envelope",
                str(envelope),
                "--policy",
                str(directory),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "outcome: DENY" in captured.out
    assert "engaged client-email-isolation" in captured.out
    assert "citation: A client may read only their own mailbox" in captured.out


def test_cli_load_compiles_rows_a_decision_can_cite(tmp_path, capsys):
    directory = _policy_dir(tmp_path)
    database = tmp_path / "store.db"
    assert main(["policy", "load", str(directory), "--db", str(database)]) == 0
    captured = capsys.readouterr()
    assert "compiled 1 rules" in captured.out

    from kognita.db import session_scope, make_engine

    engine = make_engine(database)
    with session_scope(engine) as session:
        evaluation = decide(
            Envelope(
                principal="alice@corp",
                purpose="READ",
                tool="read_email",
                subject_type="client",
                subject_id="bob",
            ),
            load_snapshot(session, as_of=AT),
            purposes=("READ",),
            as_of=AT,
            request_id="loaded",
        )
    assert evaluation.outcome is Outcome.DENY
    assert any(
        check.citation == "A client may read only their own mailbox"
        for check in evaluation.basis()
    )
    assert all(check.citation.strip() for check in evaluation.checks)


def test_cli_load_of_a_change_archives_and_refuses_an_untested_rule(tmp_path, capsys):
    directory = _policy_dir(tmp_path)
    database = tmp_path / "store.db"
    change = tmp_path / "open-geo.md"
    change.write_text(
        """\
## ADDED

- id: geo-fence
  regime: CLIENT
  rule_type: REQUIRES_HUMAN_APPROVAL
  citation: Sending mail needs a person
  effective_from: "2020-01-01T00:00:00+00:00"
  rule:
    tools:
      - send_email
""",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "policy",
                "load",
                str(change),
                "--policy",
                str(directory),
                "--db",
                str(database),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "untested" in captured.err
    assert not (directory / "geo-fence.yaml").exists()
    assert not (directory / "archive").exists()
    assert not database.exists()

    _write(
        directory,
        "geo-fence.scenarios.yaml",
        "rule: geo-fence\nscenarios:\n"
        "  - name: sending needs a person\n"
        "    given: { principal: ada }\n"
        "    when: { tool: send_email }\n"
        "    then: { decision: HUMAN_APPROVAL, cites: geo-fence }\n",
    )
    assert (
        main(
            [
                "policy",
                "load",
                str(change),
                "--policy",
                str(directory),
                "--db",
                str(database),
            ]
        )
        == 0
    )
    assert (directory / "geo-fence.yaml").is_file()
    archived = list((directory / "archive").iterdir())
    assert len(archived) == 1
    assert archived[0].name[:10] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    captured = capsys.readouterr()
    assert "applied" in captured.out
    assert main(["policy", "test", str(directory)]) == 0


def test_python_module_entry_reaches_the_cli():
    result = subprocess.run(
        [sys.executable, "-m", "kognita", "policy", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for name in ("load", "diff", "validate", "explain", "test"):
        assert name in result.stdout


def test_validate_of_a_change_needs_the_current_policy_set(tmp_path, capsys):
    change = tmp_path / "change.md"
    change.write_text("## REMOVED\n\n- id: client-email-isolation\n", encoding="utf-8")
    assert main(["policy", "validate", str(change)]) == 2
    captured = capsys.readouterr()
    assert "--policy" in captured.err
