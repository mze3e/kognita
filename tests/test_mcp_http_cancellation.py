"""Real socket cancellation, with real deployment authority and credentials."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import pytest
from sqlmodel import Session, select

from kognita.evidence import verify_chain
from kognita.mcp import BackendServer
from kognita.models import EvidenceEvent
from kognita.registry import set_kill_switch
from test_actions_s6c import _manifest
from test_mcp import _call, _proxy
from test_mcp_client_argument_scope import _delegated


@pytest.mark.parametrize('control', ['kill', 'timeout', 'completed'])
@pytest.mark.parametrize('phase', ['headers', 'body'])
def test_default_http_transport_closes_inflight_connection(session, evidence, engine, control, phase):
    did, credential = _delegated(session, evidence)
    _manifest(session, evidence, max_duration=0.15 if control == 'timeout' else 3)
    entered, disconnected = Event(), Event()
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if phase == 'body' and control != 'completed':
                self.send_response(200)
                self.send_header('Content-Length', '100')
                self.end_headers()
                self.wfile.flush()
            entered.set()
            if control == 'completed':
                payload = b'{"result":{"marked":"finished"}}'
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                # The tool is sleeping pending input; EOF proves actual transport
                # shutdown, independently of any cancellation adapter claim.
                Event().wait(0.05)
                self.connection.settimeout(10)
                try:
                    if self.connection.recv(1) == b'':
                        disconnected.set()
                except Exception as exc:
                    errors.append(exc)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = Thread(target=server.serve_forever, daemon=True)
    serving.start()
    session.commit()

    def kill():
        try:
            assert entered.wait(10)
            with Session(engine) as other:
                set_kill_switch(other, 'dossier-agent', True, evidence=evidence,
                                actor_id='Head of Research Ops')
                other.commit()
        except Exception as exc:
            errors.append(exc)

    stopper = Thread(target=kill) if control == 'kill' else None
    if stopper:
        stopper.start()
    try:
        proxy = _proxy(engine, evidence, None, servers=(
            BackendServer('notes', f'http://127.0.0.1:{server.server_port}/'),))
        response = proxy.handle('POST', '/mcp', {
            'agent_name': 'dossier-agent', 'agent_credential': credential,
            'delegation_id': did}, _call(), session=session)
        assert entered.is_set(), response.body
        # Socket EOF confirms local disconnection, not remote action cancellation.
        expected = {'kill': 'connection_aborted', 'timeout': 'connection_aborted',
                    'completed': 'completed'}[control]
        if control == 'completed':
            assert json.loads(response.body)['result']['marked'] == 'finished'
        else:
            assert json.loads(response.body)['error']['message'] == expected
            assert disconnected.wait(1), 'upstream did not observe socket EOF'
            assert b'finished' not in response.body
        events = session.exec(select(EvidenceEvent).where(
            EvidenceEvent.correlation_id == response.evaluation.request_id)).all()
        outcomes = [e.payload['outcome'] for e in events
                    if e.payload.get('citation') == 'S6C-IN-FLIGHT']
        assert outcomes == [expected]
        assert verify_chain(session) > 0
        if stopper:
            stopper.join(2)
        assert not errors
    finally:
        server.shutdown()
        server.server_close()
        serving.join(2)
