#!/usr/bin/env python3
"""Owner's assistant tool: list targets, send explicitly authorized input, read replies.

Requires an active owner sign-in from the node's private panel. This tool never
authorizes itself to message an agent: obtain the user's instruction first.
"""
import argparse
import json
import sys
import uuid
from pathlib import Path
from manuals_rag_inference.agent_link_client import call
from manuals_rag_inference.agent_link_session import load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('targets', 'send', 'messages'))
    parser.add_argument('--input', type=Path, help='UTF-8 JSON containing id, node_id, thread_id, message. Use - for stdin.')
    args = parser.parse_args()
    payload = {}
    try:
        if args.action == 'send':
            if args.input is None: raise ValueError('Explicit message input required')
            payload = json.load(sys.stdin) if str(args.input) == '-' else json.loads(args.input.read_text(encoding='utf-8'))
            if set(payload) - {'id', 'node_id', 'thread_id', 'message'}: raise ValueError('Supported message fields required')
            payload.setdefault('id', str(uuid.uuid4()))
        session = load_session()
        result = call(session['gateway_url'], session['session_token'], args.action, payload)
        print(json.dumps(result, ensure_ascii=False))
    except Exception:
        print(json.dumps({'status': 'unavailable_or_unconfirmed', 'id': payload.get('id'), 'detail': 'Verify owner sign-in. Inspect delivery status before resending; no automatic retry was made.'}))
        raise SystemExit(1)


if __name__ == '__main__': main()
