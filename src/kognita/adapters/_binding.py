"""Private model transport for framework bindings."""
from __future__ import annotations

import json
from kognita.vocabulary import Outcome


def _complete(binding, model, messages):
    from kognita.adapters.binding import GovernedResult, citations_of, _refused

    try:
        body = json.dumps(
            {"model": model, "messages": [dict(message) for message in messages]}
        ).encode()
    except (TypeError, ValueError) as exc:
        return _refused(f"the model call was refused before it was sent: {exc}")
    try:
        response = binding.gateway.handle(
            "POST",
            "/v1/chat/completions",
            {key: value for key, value in {
                "agent_name": binding.agent_name,
                "agent_credential": binding.agent_credential,
                "delegation_id": binding.delegation_id,
                "principal": binding.principal,
                "purpose": binding.purpose,
            }.items() if value is not None},
            body,
            session=binding.session,
            run=binding.run,
        )
    except Exception as exc:
        return _refused(f"the model call was refused before it was sent: {exc}")
    evaluation = response.evaluation
    released = (
        response.status == 200
        and evaluation is not None
        and evaluation.outcome is Outcome.ALLOW
    )
    data = None
    if released:
        try:
            parsed = json.loads(response.body.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return _refused(f"the model call was refused before it was sent: {exc}")
        data = parsed if isinstance(parsed, dict) else None
        released = data is not None
    outcome = evaluation.outcome if evaluation is not None else Outcome.DENY
    return GovernedResult(
        released=released,
        outcome=outcome,
        citations=citations_of(evaluation),
        data=data,
        evaluation=evaluation,
    )
