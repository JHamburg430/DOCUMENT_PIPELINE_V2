#!/usr/bin/env python3
"""Owner-authorized CLI for the versioned PROVIS Node Channel."""
import argparse
import json
import sys
import uuid
from pathlib import Path
from manuals_rag_inference.agent_link_client import call
from manuals_rag_inference.agent_link_session import load_session
from manuals_rag_inference.node_channel_protocol import PROTOCOL_ID


def _workspace(session):
    if session.get('node_channel_protocol') != PROTOCOL_ID:
        raise PermissionError('PROVIS_NODE_CHANNEL_VERSION_MISMATCH: verify the owner again with a current node')
    return call(session['gateway_url'], session['session_token'], 'workspace', {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'workspaces', 'history', 'send'))
    parser.add_argument('--input', type=Path,
                        help='UTF-8 JSON with pc_name, message, confirmation. Use - for stdin.')
    args = parser.parse_args()
    payload = {}
    try:
        session = load_session()
        if args.action == 'status':
            result = call(session['gateway_url'], session['session_token'], 'availability', {})
        else:
            workspace = _workspace(session)
            if args.action == 'workspaces':
                result = {'node_channel_protocol': PROTOCOL_ID,
                    'current_pc_name': workspace['node_id'],
                    'public_sessions': [row for row in workspace['participants'] if row.get('online')]}
            elif args.action == 'history':
                result = call(session['gateway_url'], session['session_token'], 'history',
                              {'conversation_id': workspace['channel_id']})
            else:
                if args.input is None: raise ValueError('Explicit message input required')
                payload = json.load(sys.stdin) if str(args.input) == '-' else json.loads(args.input.read_text(encoding='utf-8'))
                if set(payload) != {'pc_name', 'message', 'confirmation'} or payload['confirmation'] is not True:
                    raise PermissionError('Exact PC, message, and explicit confirmation required')
                current = {row['node_id'] for row in workspace['participants'] if row.get('online')}
                if payload['pc_name'] == workspace['node_id'] or payload['pc_name'] not in current:
                    raise PermissionError('Exact current PC name required')
                message = payload['message'].strip()
                if not message or message.startswith('@'):
                    raise ValueError('One unaddressed message body required')
                result = call(session['gateway_url'], session['session_token'], 'post', {
                    'id': str(uuid.uuid4()), 'kind': 'channel',
                    'target_node_id': payload['pc_name'], 'message': message,
                    'origin': 'owner_mcp', 'hop_count': 0})
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({'status': 'unavailable_or_unconfirmed', 'detail': str(error)[:500]}))
        raise SystemExit(1)


if __name__ == '__main__': main()
