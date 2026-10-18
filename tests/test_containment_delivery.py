"""Real authority integrations, outside the legacy authority fixture allowlist."""
from threading import Event, Thread

from sqlmodel import Session, select

from conftest import NOW
from kognita.models import EvidenceEvent, _AnomalyDay
from kognita.recorder import alerts
from kognita.credentials import verify
from test_classification_floor import _authorize
from test_mcp import _register, _proxy, _call


def _exporter(monkeypatch, writer):
    import sys
    from types import ModuleType, SimpleNamespace
    spans = []
    class _Span:
        def __init__(self, name):
            self.name, self.attributes = name, {}
        def set_attribute(self, key, value):
            self.attributes[key] = value
        def end(self):
            provider.processor.exporter.export([self])
    class _Provider:
        def __init__(self, **kwargs):
            pass
        def add_span_processor(self, processor):
            self.processor = processor
        def get_tracer(self, name):
            return SimpleNamespace(start_span=_Span)
    class _Exporter:
        def export(self, batch):
            spans.extend(batch)
            return success
        def shutdown(self):
            pass
    success = object()
    provider = _Provider()
    definitions = {
        'opentelemetry.exporter.otlp.proto.http.trace_exporter': dict(OTLPSpanExporter=_Exporter),
        'opentelemetry.sdk.resources': dict(Resource=SimpleNamespace(create=lambda attrs: attrs)),
        'opentelemetry.sdk.trace': dict(TracerProvider=lambda **kwargs: provider),
        'opentelemetry.sdk.trace.export': dict(SimpleSpanProcessor=lambda exporter: SimpleNamespace(exporter=exporter),
            SpanExportResult=SimpleNamespace(SUCCESS=success, FAILURE=object())),
    }
    for name, values in definitions.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    from kognita.otel import build_hook
    writer.telemetry = build_hook()
    return SimpleNamespace(get_finished_spans=lambda: spans)


def _delivered(session, memory, action, owner):
    found = [a for a in alerts(session, as_of=NOW)['alerts'] if a['kind'] == action]
    assert len(found) == 1
    assert found[0]['fired'] and found[0]['citation']
    recipients = found[0]['_recipients']
    assert recipients == [owner, 'Jane Smith']
    spans = [s for s in memory.get_finished_spans() if s.name == 'kognita.alert'
             and s.attributes['kognita._action'] == action]
    assert sorted(s.attributes['kognita._recipient'] for s in spans) == sorted(recipients)


def test_containment_cancels_running_proxy_and_delivers_alert(session, engine, evidence, monkeypatch, capsys):
    from test_usecases import _open, _activate, _spec
    _open(session, evidence)
    _activate(session, evidence, _spec(usecase_id='COLLABORATION', knowledge_lead='Jane Smith'))
    _register(session)
    did, credential = _authorize(session, evidence)
    memory = _exporter(monkeypatch, evidence)
    session.add(_AnomalyDay(agent_name='dossier-agent', day=NOW.date().isoformat(), body={'actions': 998}))
    session.commit()
    started, stopped = Event(), Event()
    class _Slow:
        def __call__(self, method, url, headers, body):
            started.set()
            assert stopped.wait(5)
            raise RuntimeError('cancelled backend')
        def _cancel(self, rpc_id):
            stopped.set()
            return True
    proxy = _proxy(engine, evidence, _Slow())
    headers = dict(agent_name='dossier-agent', agent_credential=credential, delegation_id=did)
    from test_mcp import _Upstream
    completed = _proxy(engine, evidence, _Upstream()).handle('POST', '/mcp', headers, _call(rpc_id=3), session=session)
    session.commit()
    assert completed.status == 200
    responses = []
    def call():
        with Session(engine) as worker:
            responses.append(proxy.handle('POST', '/mcp', headers, _call(), session=worker))
            worker.commit()
    thread = Thread(target=call)
    thread.start()
    try:
        assert started.wait(3)
        crossing = proxy.handle('POST', '/mcp', headers, _call(rpc_id=2), session=session)
        session.commit()
        assert crossing.status == 403
        thread.join(3)
        assert not thread.is_alive()
        assert stopped.is_set()
        assert responses[0].status == 502
        assert b'cancelled' in responses[0].body
        assert not verify(session, 'dossier-agent', credential)
        outcomes = [e.payload.get('outcome') for e in session.exec(select(EvidenceEvent))
                    if e.payload.get('citation') == 'S6C-IN-FLIGHT']
        assert outcomes == ['completed', 'cancelled']
        _delivered(session, memory, 'ANOMALY_ALERT', 'Head of Research Ops')
        from contextlib import nullcontext
        from types import SimpleNamespace
        from kognita import cli
        import json
        monkeypatch.setattr(cli, '_recorder_session', lambda args: nullcontext(session))
        assert cli.cmd_recorder_alerts(SimpleNamespace()) == 0
        assert any(a['kind'] == 'ANOMALY_ALERT' for a in json.loads(capsys.readouterr().out)['alerts'])
    finally:
        stopped.set()
        thread.join(5)


def test_gateway_sustained_decline_steps_down_and_delivers_alert(session, engine, monkeypatch):
    from datetime import timedelta
    from test_autonomy_s1 import _agent, _samples
    from test_usecases import _open, _activate, _spec
    from kognita.evidence import EvidenceWriter
    from kognita.registry import _deployment_hash
    from kognita.authority import grant_to, delegate, Delegation
    from kognita.credentials import issue
    from kognita.gateway import ClientConfiguration
    from test_gateway import _gateway, _Upstream, _body, PATH, MODEL
    from datetime import timedelta
    writer = EvidenceWriter(engine, clock=lambda: NOW-timedelta(days=40))
    agent = _agent(session, writer)
    writer = EvidenceWriter(engine, clock=lambda: NOW)
    _open(session, writer)
    _activate(session, writer, _spec(knowledge_lead='Jane Smith', validators=('Jane Smith', 'validator'),
        _autonomy_step_down=dict(minimum_sample=4, window_days=7, quality=.7, reliability=.8)))
    permission = dict(agent.tool_allowlist['read'], required_level=2, verb='recommend', permissions=['recommend'])
    agent.tool_allowlist = {'model_call': permission, '_deployment': dict(system_prompt='', instructions='', tools=[], permissions=[], data_sources=[], orchestration='', memory_behaviour='')}
    agent.autonomy_level = 2
    agent.deployment_hash = _deployment_hash(agent, agent.tool_allowlist['_deployment'])
    # Keep the approved observation digest aligned with the synthetic historical deployment.
    from kognita.vocabulary import ActorType, EventType
    from kognita.evidence import EvidenceWriter
    old = EvidenceWriter(engine, clock=lambda: NOW-timedelta(days=40))
    old.emit(session, correlation_id='deployment:agent', event_type=EventType.POLICY_CHANGE,
             actor_type=ActorType.HUMAN, actor_id='owner', payload={'deployment_hash': agent.deployment_hash})
    for days, good in ((18, 4), (11, 3), (4, 2)):
        _samples(session, engine, agent, NOW-timedelta(days=days), good=good)
    scope = {'client': ['123'], 'model': [MODEL]}
    grant_to('alice', 'recommend', timedelta(hours=2), scope, session=session, evidence=writer, actor_id='admin', now=NOW)
    did = delegate(session, Delegation(human='alice', agent='agent', purpose='research-summary', client='123',
        allowed_actions=['recommend'], allowed_data=scope, valid_from=NOW, valid_until=NOW+timedelta(hours=1),
        jurisdiction='SG', task_ref='decline'), evidence=writer, actor_id='alice', now=NOW)
    cred = issue(session, 'agent', evidence=writer, actor_id='owner')
    memory = _exporter(monkeypatch, writer)
    backend = _Upstream(b'{}')
    gateway = _gateway(engine, writer, backend, client=ClientConfiguration(principal='alice', purpose='research-summary',
        agent_names=frozenset({'agent'}), actor_location='SG'))
    response = gateway.handle('POST', PATH, dict(agent_name='agent', agent_credential=cred, delegation_id=did), _body('hello'), session=session)
    assert agent.autonomy_level == 1
    assert response.status == 403 and not backend.calls
    assert any(c.citation == 'S1-LEVEL' for c in response.evaluation.failures())
    _delivered(session, memory, 'ALERT', 'owner')
