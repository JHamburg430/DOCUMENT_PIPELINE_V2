import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
import run_inference_node_client as client


def test_service_attachment_uses_advertised_port_and_control_auth(tmp_path):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.headers.get('X-Node-Local-Token'), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"node_id":"fixture","inference_enabled":false}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path/'node.json'
    config.write_text(json.dumps({'node_id': 'fixture', 'port': 1, 'bind': '127.0.0.1',
        'agent_url': f'http://127.0.0.1:{server.server_port}', 'local_control_token': 'fixture-control'}))
    app = SimpleNamespace(config_path=str(config))
    try:
        result = client.InferenceNodeClient._attached_request(app, '/v1/node/settings', method='POST', payload={'action': 'disable_inference'})
        assert result['inference_enabled'] is False
        assert seen == [('/v1/node/settings', 'fixture-control', {'action': 'disable_inference'})]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_attachment_does_not_accept_another_node_identity(tmp_path):
    config = tmp_path/'node.json'
    config.write_text('{"node_id":"expected"}')
    messages = []
    app = SimpleNamespace(config_path=str(config), _attached_request=lambda path: {'node_id': 'wrong'},
        _update_controls=lambda: None, status_var=SimpleNamespace(set=messages.append), after=lambda *args: None,
        _attach_existing_node=lambda: None)
    client.InferenceNodeClient._attach_existing_node(app)
    assert app.attached is False
    assert messages == ['PROVIS service is not available']
