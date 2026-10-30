"""The rule registry and the generic primitives it ships with.

A policy row names a ``rule_type``; the evaluator registered for that type reads
the row's JSON ``rule`` payload and returns checks. The core ships primitives
general enough to express most access rules — allowlists, denylists, required
flags, mandatory human review — and a domain pack registers evaluators for
anything its regimes need beyond them.

Every evaluator must be a pure function of ``(policy, context)``. No I/O, no
clock reads, no randomness: replaying a decision must give the same answer, or
the evidence plane is recording a story rather than a fact.
"""

from __future__ import annotations

from typing import Any, Callable

from kognita.envelope import Check, RuleContext
from kognita.vocabulary import CheckResult

Evaluator = Callable[[Any, RuleContext], list[Check]]

#: Evaluators shipped by the core, keyed by ``rule_type``.
CORE_RULES: dict[str, Evaluator] = {}


def rule(rule_type: str) -> Callable[[Evaluator], Evaluator]:
    """Register an evaluator for ``rule_type`` in the core registry."""

    def register(fn: Evaluator) -> Evaluator:
        CORE_RULES[rule_type] = fn
        return fn

    return register


def build_registry(*overlays: dict[str, Evaluator]) -> dict[str, Evaluator]:
    """Merge pack registries over the core primitives; later overlays win."""
    registry = dict(CORE_RULES)
    for overlay in overlays:
        registry.update(overlay)
    return registry


def _lookup(context: RuleContext, key: str) -> Any:
    """Read an attribute a rule refers to.

    Attributes resolved by the pack take precedence; envelope fields are the
    fallback so simple packs need not restate ``actor_location`` and friends.
    """
    if key in context.attributes:
        return context.attributes[key]
    return getattr(context.envelope, key, None)


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(v) for v in value}
    return {str(value)}


def _citation(policy: Any, context: RuleContext, *keys: str) -> str:
    """The policy rule, and the classifier label when the check acted on one.

    A check that did not read a classifier-filled attribute keeps the policy
    citation alone. Typed attributes are not a classifier label.
    """
    citation = policy.citation or ""
    record = context.attributes.get("classifier")
    if not isinstance(record, dict):
        return citation
    filled = {str(item) for item in record.get("filled") or []}
    if not filled.intersection(keys):
        return citation
    label = record.get("label")
    confidence = record.get("confidence")
    if label is None or confidence is None:
        return citation
    return f"{citation} — classified as {label} at {float(confidence):.2f}"


# ── Primitives ───────────────────────────────────────────────────────────────


@rule("ATTRIBUTE_ALLOWLIST")
def attribute_allowlist(policy: Any, context: RuleContext) -> list[Check]:
    """Every named attribute must be in its allowlist.

    Rule payload::

        {"allow": {"actor_location": ["SG", "HK"]},
         "on_violation": "fail" | "escalate" | "requires_human"}

    An attribute the request does not carry is a violation, not a pass: an
    allowlist that cannot be evaluated has not been satisfied.
    """
    allow: dict[str, Any] = policy.rule.get("allow", {})
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    checks: list[Check] = []
    for key, permitted in allow.items():
        actual = _lookup(context, key)
        permitted_set = _as_set(permitted)
        ok = actual is not None and str(actual) in permitted_set
        checks.append(
            Check(
                check=f"{policy.rule_type}: {key} {actual!s}",
                regime=policy.regime,
                result=CheckResult.PASS if ok else on_violation,
                citation=_citation(policy, context, key),
                policy_id=policy.id,
            )
        )
    return checks


@rule("ATTRIBUTE_DENYLIST")
def attribute_denylist(policy: Any, context: RuleContext) -> list[Check]:
    """No named attribute may match its denylist.

    Rule payload::

        {"deny": {"actor_location": ["RU"]}, "on_violation": "fail"}
    """
    deny: dict[str, Any] = policy.rule.get("deny", {})
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    checks: list[Check] = []
    for key, forbidden in deny.items():
        actual = _lookup(context, key)
        hit = actual is not None and str(actual) in _as_set(forbidden)
        checks.append(
            Check(
                check=f"{policy.rule_type}: {key} {actual!s}",
                regime=policy.regime,
                result=on_violation if hit else CheckResult.PASS,
                citation=_citation(policy, context, key),
                policy_id=policy.id,
            )
        )
    return checks


@rule("REQUIRES_FLAG")
def requires_flag(policy: Any, context: RuleContext) -> list[Check]:
    """Named attributes must all be truthy.

    Rule payload::

        {"flags": ["accredited_investor"], "on_violation": "fail"}
    """
    flags: list[str] = list(policy.rule.get("flags", []))
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    checks: list[Check] = []
    for flag in flags:
        ok = bool(_lookup(context, flag))
        checks.append(
            Check(
                check=f"{policy.rule_type}: {flag}",
                regime=policy.regime,
                result=CheckResult.PASS if ok else on_violation,
                citation=policy.citation,
                policy_id=policy.id,
            )
        )
    return checks


@rule("REQUIRES_HUMAN_APPROVAL")
def requires_human_approval(policy: Any, context: RuleContext) -> list[Check]:
    """Mandate human review, optionally only for certain tools or purposes.

    Rule payload::

        {"tools": ["draft_meeting_pack"], "purposes": ["ADVICE"]}

    Empty or absent lists mean "any". This is how an automated-decision regime is
    expressed: the work may be prepared, but a person signs it off.
    """
    tools = _as_set(policy.rule.get("tools"))
    purposes = _as_set(policy.rule.get("purposes"))
    if tools and context.envelope.tool not in tools:
        return []
    if purposes and context.envelope.purpose not in purposes:
        return []
    return [
        Check(
            check=policy.rule_type,
            regime=policy.regime,
            result=CheckResult.REQUIRES_HUMAN,
            citation=policy.citation,
            policy_id=policy.id,
        )
    ]


@rule("TWO_SIGNATURE_APPROVAL")
def two_signature_approval(policy: Any, context: RuleContext) -> list[Check]:
    """Mandate two-signature approval: one marks, another confirms (ADR 0006).

    Rule payload::

        {"tools": ["mark_criterion"], "purposes": ["MARK_CRITERION"]}

    Empty or absent lists mean "any". Produces REQUIRES_HUMAN outcome but also
    signals to the governance layer that confirmation_required=True.
    """
    tools = _as_set(policy.rule.get("tools"))
    purposes = _as_set(policy.rule.get("purposes"))
    if tools and context.envelope.tool not in tools:
        return []
    if purposes and context.envelope.purpose not in purposes:
        return []
    return [
        Check(
            check=policy.rule_type,
            regime=policy.regime,
            result=CheckResult.REQUIRES_HUMAN,
            citation=policy.citation,
            policy_id=policy.id,
        )
    ]


@rule("CLASSIFIER_CONFIDENCE")
def classifier_confidence(policy: Any, context: RuleContext) -> list[Check]:
    """Escalate when a classifier label is less sure than the policy allows.

    Rule payload::

        {"min_confidence": 0.8}

    The threshold is the policy. Below it the check is ESCALATE, so the
    decision cannot be ALLOW. A request with no classifier record was not
    classified; the rule does not invent uncertainty for a typed envelope.
    A missing threshold is itself uncertain and escalates.
    """
    record = context.attributes.get("classifier")
    if not isinstance(record, dict) or "confidence" not in record:
        return []
    label = record.get("label")
    confidence = float(record["confidence"])
    citation = f"{policy.citation} — classified as {label} at {confidence:.2f}"
    threshold = policy.rule.get("min_confidence")
    if threshold is None or confidence < float(threshold):
        result = CheckResult.ESCALATE
    else:
        result = CheckResult.PASS
    return [
        Check(
            check=policy.rule_type,
            regime=policy.regime,
            result=result,
            citation=citation,
            policy_id=policy.id,
        )
    ]


@rule("CONTACT_FREQUENCY")
def contact_frequency(policy: Any, context: RuleContext) -> list[Check]:
    """Refuse another contact once the client is at the policy's limit.

    Rule payload::

        {"max_contacts": 2, "window_days": 30, "tools": ["send_note"]}

    ``tools`` narrows the check to those tools. With no list, the check
    applies to every request the pack engages. The count is the attribute
    ``contact_count``. A missing count, limit, or window fails closed: the
    rule cannot be treated as satisfied. Origination reads the same policy
    rows and records do-not-contact when a recommendation would cross one.
    """
    tools = _as_set(policy.rule.get("tools"))
    if tools and context.envelope.tool not in tools:
        return []
    limit = policy.rule.get("max_contacts")
    window = policy.rule.get("window_days")
    count = context.attributes.get("contact_count")
    citation = policy.citation or "Origination: contact frequency"
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or isinstance(window, bool)
        or not isinstance(window, int)
        or window < 1
        or isinstance(count, bool)
        or not isinstance(count, int)
    ):
        return [
            Check(
                check=policy.rule_type,
                regime=policy.regime,
                result=CheckResult.FAIL,
                citation=f"{citation} — contact frequency could not be evaluated",
                policy_id=policy.id,
            )
        ]
    ok = count < limit
    return [
        Check(
            check=policy.rule_type,
            regime=policy.regime,
            result=CheckResult.PASS if ok else CheckResult.FAIL,
            citation=f"{citation} — {count} contacts in {window} days, limit {limit}",
            policy_id=policy.id,
        )
    ]


@rule("PRINCIPAL_EQUALITY")
def principal_equality(policy: Any, context: RuleContext) -> list[Check]:
    """The principal must equal one named attribute.

    Rule payload::

        {"equals": "subject_owner", "on_violation": "fail"}

    ``equals`` is an attribute, or an envelope field when the attribute is
    absent. The comparison is string equality. A missing name, a missing
    value, a value that is not a string, or an empty principal fails closed.
    ``tools`` and ``purposes`` narrow the rule; empty or absent means any.
    """
    tools = _as_set(policy.rule.get("tools"))
    purposes = _as_set(policy.rule.get("purposes"))
    if tools and context.envelope.tool not in tools:
        return []
    if purposes and context.envelope.purpose not in purposes:
        return []
    on_violation = CheckResult(policy.rule.get("on_violation", "fail"))
    key = policy.rule.get("equals")
    citation = policy.citation or "Principal equality"
    principal = context.envelope.principal
    if not isinstance(key, str) or not key.strip():
        ok = False
        detail = "principal equality could not be evaluated"
    else:
        actual = _lookup(context, key)
        ok = isinstance(actual, str) and bool(principal) and actual == principal
        if ok:
            detail = f"principal matches {key}"
        elif not isinstance(actual, str) or actual == "":
            detail = f"principal equality could not be evaluated for {key}"
            ok = False
        else:
            detail = f"principal does not match {key}"
    return [
        Check(
            check=policy.rule_type,
            regime=policy.regime,
            result=CheckResult.PASS if ok else on_violation,
            citation=f"{citation} — {detail}",
            policy_id=policy.id,
        )
    ]


def _texts(value: Any) -> list[str] | None:
    """A list of strings, or None when the value is not one."""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return value


def _content_check(
    policy: Any, name: str, *, ok: bool, detail: str
) -> Check:
    citation = policy.citation or "Governed client communication: content policy"
    if detail:
        citation = f"{citation} — {detail}"
    return Check(
        check=name,
        regime=policy.regime,
        result=CheckResult.PASS if ok else CheckResult.FAIL,
        citation=citation,
        policy_id=policy.id,
        policy_hash=policy.content_hash or None,
    )


def _validator_audience_check(policy: Any, rule: dict, context: RuleContext) -> Check:
    from kognita.verification import _normalize_map, _reaches
    status = context.attributes.get("validator_status")
    audience = context.attributes.get("audience")
    ok = _reaches(
        status,
        audience if isinstance(audience, str) else "",
        _normalize_map(rule.get("validator_audiences")),
    )
    return _content_check(
        policy,
        "CONTENT_POLICY: validator status",
        ok=ok,
        detail=(
            "validator status is allowed for this audience"
            if ok
            else "validator status is not allowed for this audience"
        ),
    )


@rule("CONTENT_POLICY")
def content_policy(policy: Any, context: RuleContext) -> list[Check]:
    """Check disclosures, claims, tone, risk language, attribution, and jurisdiction.

    Rule payload, every field optional, at least one substantive constraint::

        {"levels": ["house view"],
         "lanes": ["client"],
         "required_disclosures": ["Past performance is not a guide"],
         "prohibited_claims": ["guaranteed"],
         "approved_tones": ["neutral"],
         "permitted_jurisdictions": ["SG"],
         "permitted_channels": ["secure message"],
         "require_risk_language": true,
         "require_research_attribution": true,
         "validator_audiences": {"VERIFIED": ["client", "rm"], "INFERRED": ["rm"]}}

    A level or lane list that does not include this send means the rule is not
    in force for it. An empty allowlist fails closed. A missing content body
    fails closed. The attributes are ``communication_level``, ``lane``,
    ``content``, ``tone``, ``channel``, ``recipient_jurisdiction``,
    ``risk_language``, ``research_attribution``, ``validator_status``, and
    ``audience``.
    """
    rule = policy.rule if isinstance(policy.rule, dict) else {}
    substantive = (
        "sources",
        "required_disclosures",
        "sensitive",
        "prohibited_claims",
        "approved_tones",
        "permitted_jurisdictions",
        "permitted_channels",
        "require_risk_language",
        "require_research_attribution",
        "validator_audiences",
    )
    if not any(key in rule for key in substantive):
        return [
            _content_check(
                policy,
                "CONTENT_POLICY",
                ok=False,
                detail="content policy could not be evaluated",
            )
        ]
    levels = rule.get("levels")
    if levels:
        parsed = _texts(levels)
        current = str(context.attributes.get("communication_level") or "")
        if parsed is None or current not in parsed:
            return []
    lanes = rule.get("lanes")
    if lanes:
        parsed = _texts(lanes)
        lane = str(context.attributes.get("lane") or "client")
        if parsed is None or lane not in parsed:
            return []
    if "validator_audiences" in rule and not any(
        key in rule for key in substantive if key != "validator_audiences"
    ):
        return [_validator_audience_check(policy, rule, context)]
    content = context.attributes.get("content")
    if not isinstance(content, str):
        return [
            _content_check(
                policy,
                "CONTENT_POLICY",
                ok=False,
                detail="content policy could not be evaluated",
            )
        ]
    checks: list[Check] = []
    if "validator_audiences" in rule:
        checks.append(_validator_audience_check(policy, rule, context))
    if "sources" in rule:
        permitted = _texts(rule.get("sources"))
        cited = context.attributes.get("source_system")
        checks.append(_content_check(
            policy, "CONTENT_POLICY: sources",
            ok=bool(permitted) and isinstance(cited, str) and cited.strip().casefold() != "model" and cited in permitted,
            detail="claim source must be in the policy sources list",
        ))
    if "sensitive" in rule:
        from types import SimpleNamespace
        from kognita.sensitive import check_sensitive
        checks.append(check_sensitive(content, SimpleNamespace(permitted_sensitive=rule["sensitive"])))
    if "required_disclosures" in rule:
        required = _texts(rule.get("required_disclosures"))
        if required is None:
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: disclosures",
                    ok=False,
                    detail="content policy could not be evaluated",
                )
            )
        else:
            missing = [item for item in required if item not in content]
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: disclosures",
                    ok=not missing,
                    detail=(
                        "disclosures present"
                        if not missing
                        else "missing disclosure: " + ", ".join(missing)
                    ),
                )
            )
    if "prohibited_claims" in rule:
        banned = _texts(rule.get("prohibited_claims"))
        folded = content.casefold()
        if banned is None:
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: prohibited claims",
                    ok=False,
                    detail="content policy could not be evaluated",
                )
            )
        else:
            hits = [item for item in banned if item.casefold() in folded]
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: prohibited claims",
                    ok=not hits,
                    detail=(
                        "no prohibited claim"
                        if not hits
                        else "prohibited claim: " + ", ".join(hits)
                    ),
                )
            )
    if "approved_tones" in rule:
        tones = _texts(rule.get("approved_tones"))
        tone = str(context.attributes.get("tone") or "")
        ok = bool(tones) and tone in tones
        checks.append(
            _content_check(
                policy,
                "CONTENT_POLICY: tone",
                ok=ok,
                detail="approved tone" if ok else "tone is not an approved tone",
            )
        )
    if "permitted_channels" in rule:
        channels = _texts(rule.get("permitted_channels"))
        channel = str(context.attributes.get("channel") or "")
        ok = bool(channels) and channel in channels
        checks.append(
            _content_check(
                policy,
                "CONTENT_POLICY: channel",
                ok=ok,
                detail=(
                    "channel permitted"
                    if ok
                    else "channel is not permitted by this rule"
                ),
            )
        )
    if "permitted_jurisdictions" in rule:
        places = _texts(rule.get("permitted_jurisdictions"))
        jurisdiction = str(context.attributes.get("recipient_jurisdiction") or "")
        ok = bool(places) and jurisdiction in places
        checks.append(
            _content_check(
                policy,
                "CONTENT_POLICY: jurisdiction",
                ok=ok,
                detail=(
                    f"recipient jurisdiction {jurisdiction}"
                    if ok
                    else "cross-border rule was not satisfied"
                ),
            )
        )
    if "require_risk_language" in rule:
        required = rule.get("require_risk_language")
        if not isinstance(required, bool):
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: risk language",
                    ok=False,
                    detail="content policy could not be evaluated",
                )
            )
        else:
            present = context.attributes.get("risk_language") is True
            ok = present or not required
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: risk language",
                    ok=ok,
                    detail=(
                        "investment-risk language present"
                        if ok
                        else "investment-risk language is missing"
                    ),
                )
            )
    if "require_research_attribution" in rule:
        required = rule.get("require_research_attribution")
        attribution = str(context.attributes.get("research_attribution") or "").strip()
        if not isinstance(required, bool):
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: research attribution",
                    ok=False,
                    detail="content policy could not be evaluated",
                )
            )
        else:
            ok = bool(attribution) or not required
            checks.append(
                _content_check(
                    policy,
                    "CONTENT_POLICY: research attribution",
                    ok=ok,
                    detail=(
                        "research attribution present"
                        if ok
                        else "research attribution is missing"
                    ),
                )
            )
    return checks


@rule("AI_DISCLOSURE")
def ai_disclosure(policy: Any, context: RuleContext) -> list[Check]:
    """Decide when the client must be told that AI was involved.

    Rule payload::

        {"triggers": ["AI generated content"],
         "jurisdictions": ["SG"],
         "use_cases": ["portfolio-review"],
         "text": "This message used AI."}

    ``triggers`` and ``text`` are required. An empty jurisdiction or use-case
    list means every one. A trigger that is not active on this send means the
    rule is not in force. The active triggers are the attribute
    ``disclosure_triggers``.
    """
    rule = policy.rule if isinstance(policy.rule, dict) else {}
    citation = policy.citation or "Governed client communication: AI disclosure"
    triggers = _texts(rule.get("triggers"))
    text = rule.get("text")
    if not triggers or not isinstance(text, str) or not text.strip():
        return [
            Check(
                check="AI_DISCLOSURE",
                regime=policy.regime,
                result=CheckResult.FAIL,
                citation=f"{citation} — disclosure rule could not be evaluated",
                policy_id=policy.id,
                policy_hash=policy.content_hash or None,
            )
        ]
    active = context.attributes.get("disclosure_triggers") or []
    if not isinstance(active, (list, tuple, set, frozenset)):
        active = []
    if not set(triggers).intersection(str(item) for item in active):
        return []
    jurisdictions = rule.get("jurisdictions")
    if jurisdictions:
        places = _texts(jurisdictions)
        jurisdiction = str(context.attributes.get("recipient_jurisdiction") or "")
        if not places or not jurisdiction:
            return [
                Check(
                    check="AI_DISCLOSURE",
                    regime=policy.regime,
                    result=CheckResult.FAIL,
                    citation=f"{citation} — disclosure rule could not be evaluated",
                    policy_id=policy.id,
                    policy_hash=policy.content_hash or None,
                )
            ]
        if jurisdiction not in places:
            return [
                Check(
                    check="AI_DISCLOSURE",
                    regime=policy.regime,
                    result=CheckResult.PASS,
                    citation=(
                        f"{citation} — disclosure is not required in {jurisdiction}"
                    ),
                    policy_id=policy.id,
                    policy_hash=policy.content_hash or None,
                )
            ]
    use_cases = rule.get("use_cases")
    if use_cases:
        named = _texts(use_cases)
        use_case = str(context.attributes.get("use_case") or "")
        if named is None:
            return [
                Check(
                    check="AI_DISCLOSURE",
                    regime=policy.regime,
                    result=CheckResult.FAIL,
                    citation=f"{citation} — disclosure rule could not be evaluated",
                    policy_id=policy.id,
                    policy_hash=policy.content_hash or None,
                )
            ]
        if use_case not in named:
            return [
                Check(
                    check="AI_DISCLOSURE",
                    regime=policy.regime,
                    result=CheckResult.PASS,
                    citation=(
                        f"{citation} — disclosure is not required for this use case"
                    ),
                    policy_id=policy.id,
                    policy_hash=policy.content_hash or None,
                )
            ]
    found = text in str(context.attributes.get("content") or "")
    return [
        Check(
            check="AI_DISCLOSURE",
            regime=policy.regime,
            result=CheckResult.PASS if found else CheckResult.FAIL,
            citation=(
                f"{citation} — disclosure present"
                if found
                else f"{citation} — required disclosure is missing"
            ),
            policy_id=policy.id,
            policy_hash=policy.content_hash or None,
        )
    ]


def request_field_keys(envelope: Any, attributes: dict[str, Any], *fields: str) -> set[str]:
    """Strings a request carries under ``fields``, from arguments and attributes."""
    found: set[str] = set()
    sources = (getattr(envelope, "arguments", None) or {}, attributes or {})
    for source in sources:
        if not isinstance(source, dict):
            continue
        for field in fields:
            value = source.get(field)
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
    return found


def request_model_keys(envelope: Any, attributes: dict[str, Any]) -> set[str]:
    """Model version, model name, model card, and ``provider/model@version``.

    A revocation and a circuit breaker match the same set, so a model stopped
    under one spelling is the model the breaker saw on the flag.
    """
    found = request_field_keys(
        envelope, attributes, "model_version", "model", "model_card"
    )
    subject_type = getattr(envelope, "subject_type", None)
    subject_id = getattr(envelope, "subject_id", None)
    if subject_type == "model" and isinstance(subject_id, str) and subject_id.strip():
        found.add(subject_id.strip())
    providers = request_field_keys(envelope, attributes, "provider")
    names = request_field_keys(envelope, attributes, "model")
    versions = request_field_keys(envelope, attributes, "model_version")
    if subject_type == "model" and isinstance(subject_id, str) and subject_id.strip():
        names.add(subject_id.strip())
    for provider in providers:
        for name in names:
            for version in versions:
                found.add(f"{provider}/{name}@{version}")
    return found


def breaker_matches(rule: dict[str, Any], context: RuleContext) -> bool:
    """Whether a circuit-breaker policy applies to this request.

    A policy that does not name a known dimension and value matches every
    request. A breaker that cannot be scoped does not stay silent.
    """
    dimension = str(rule.get("dimension") or "")
    value = str(rule.get("value") or "").strip()
    if dimension not in {"usecase", "model", "prompt", "policy"} or not value:
        return True
    envelope = context.envelope
    attributes = context.attributes
    if dimension == "usecase":
        return envelope.purpose == value
    if dimension == "model":
        return value in request_model_keys(envelope, attributes)
    if dimension == "prompt":
        return value in request_field_keys(
            envelope, attributes, "prompt_template_version"
        )
    usecases = {
        str(item).strip()
        for item in (rule.get("usecases") or [])
        if str(item).strip()
    }
    if not usecases:
        return True
    return envelope.purpose in usecases


@rule("CIRCUIT_BREAKER")
def circuit_breaker(policy: Any, context: RuleContext) -> list[Check]:
    """Deny the dimension a tripped breaker named.

    A request outside that dimension is not checked. The policy is still in
    force for the dimension it names. ``on_violation`` defaults to fail.
    """
    rule = policy.rule if isinstance(policy.rule, dict) else {}
    if not breaker_matches(rule, context):
        return []
    try:
        result = CheckResult(rule.get("on_violation", "fail"))
    except ValueError:
        result = CheckResult.FAIL
    return [
        Check(
            check="CIRCUIT_BREAKER",
            regime=policy.regime,
            result=result,
            citation=policy.citation or "Circuit breaker",
            policy_id=policy.id,
        )
    ]


@rule("PROHIBITED")
def prohibited(policy: Any, context: RuleContext) -> list[Check]:
    """An unconditional bar, once the policy is engaged at all.

    Engagement is decided by ``applies_to`` and the pack's ``engages`` predicate,
    so reaching this evaluator already means the rule is in scope.
    """
    return [
        Check(
            check=policy.rule.get("description", policy.rule_type),
            regime=policy.regime,
            result=CheckResult(policy.rule.get("on_violation", "fail")),
            citation=policy.citation,
            policy_id=policy.id,
        )
    ]


__all__ = [
    "rule",
    "build_registry",
    "CORE_RULES",
    "Evaluator",
    "breaker_matches",
    "request_field_keys",
    "request_model_keys",
]
