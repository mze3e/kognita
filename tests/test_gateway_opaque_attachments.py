"""Opaque content must fail admission with real authority and credentials."""
import base64
import json

import pytest

from kognita.tools import Run
from kognita.vocabulary import Outcome
from test_gateway_run_controls import _setup


@pytest.fixture(autouse=True)
def _legacy_workload_credentials():
    """Disable the legacy handler wrapper; _setup issues real credentials."""
    yield


_TYPES = ("file", "image", "audio", "input_file", "input_image", "input_audio", "image_url")
_ENCODED = base64.b64encode(b"email ana@example.org").decode()


def _payload(path, parts):
    if path == "/v1/responses":
        return {"model": "gateway-model", "input": parts}
    return {"model": "gateway-model", "messages": [{"role": "user", "content": parts}]}


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("part_type", _TYPES)
@pytest.mark.parametrize("representation", ["base64", "data_url", "file_id", "remote_url"])
def test_opaque_attachment_is_denied_before_forwarding(session, evidence, path, part_type, representation):
    gateway, backend, headers = _setup(session, evidence)
    if representation == "file_id":
        data = {"file_id": "file-opaque"}
    elif representation == "remote_url":
        data = {"url": "https://example.org/opaque"}
    elif representation == "data_url":
        data = {"url": "data:application/octet-stream;base64," + _ENCODED}
    else:
        data = {"data": _ENCODED, "format": "wav"}
    part = {"type": part_type, **data}
    # Include provider-native nesting as well as the flat Responses forms.
    if part_type in ("file", "image_url", "input_audio"):
        part = {"type": part_type, part_type: data}
    response = gateway.handle(
        "POST", path, headers,
        json.dumps(_payload(path, [{"type": "text", "text": "hello"}, part])).encode(),
        session=session, run=Run(),
    )
    assert response.status == 403
    assert response.evaluation.outcome is Outcome.DENY
    assert any(c.citation == "S5-UNSUPPORTED-CONTENT" for c in response.evaluation.checks)
    assert backend.calls == []


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("part_type", ["text", "input_text"])
def test_plain_text_parts_still_forward(session, evidence, path, part_type):
    gateway, backend, headers = _setup(session, evidence)
    parts = [{"type": part_type, "text": "hello"}]
    response = gateway.handle("POST", path, headers, json.dumps(_payload(path, parts)).encode(),
                              session=session, run=Run())
    assert response.status == 200
    assert response.evaluation.outcome is Outcome.ALLOW
    assert len(backend.calls) == 1
    assert json.loads(backend.calls[0]["body"]) == _payload(path, parts)
