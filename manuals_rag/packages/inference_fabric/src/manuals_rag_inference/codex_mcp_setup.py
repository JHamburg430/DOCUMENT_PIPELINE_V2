"""Explicit owner-approved Codex registration; inspection never changes config."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .agent_link_session import default_path

SERVER_NAME = 'provis-agent-link'
SESSION_ENV = 'PROVIS_AGENT_LINK_SESSION_PATH'
EXPECTED_TOOLS = {'provis_list_published_sessions', 'provis_message_history', 'provis_send_message'}


class SetupError(RuntimeError):
    """A fixed, nonsecret message suitable for displaying in the node UI."""


@dataclass(frozen=True)
class SetupPlan:
    codex: str
    server: tuple[str, ...]
    session_path: str

    @property
    def command(self):
        return [self.codex, 'mcp', 'add', SERVER_NAME, '--env',
                SESSION_ENV + '=' + self.session_path, '--', *self.server]

    @property
    def display_command(self):
        # Display PowerShell syntax on Windows; execution always uses argv/shell=False.
        if os.name == 'nt':
            return '& ' + ' '.join("'" + value.replace("'", "''") + "'" for value in self.command)
        return shlex.join(self.command)


def setup_plan(*, codex=None, executable=None, frozen=None, platform=None, script=None, session_path=None):
    frozen = bool(getattr(sys, 'frozen', False)) if frozen is None else frozen
    platform = sys.platform if platform is None else platform
    executable = Path(executable or sys.executable).absolute()
    if codex is None:
        candidates = [shutil.which('codex.exe' if platform == 'win32' else 'codex')]
        if platform == 'win32':
            candidates.insert(0, str(Path.home() / 'AppData/Local/Programs/OpenAI/Codex/bin/codex.exe'))
        codex = next((item for item in candidates if item and Path(item).is_file()), None)
    if not codex or not Path(codex).is_file():
        raise SetupError('Codex CLI was not found. Install Codex for this user, then check again.')
    if frozen:
        if executable.is_symlink() or not executable.is_file():
            raise SetupError('The installed node executable is unavailable. Nothing was registered.')
        server = (str(executable), '--agent-link-mcp')
    else:
        adapter = Path(script) if script else Path(__file__).resolve().parents[4] / 'scripts/run_agent_link_mcp.py'
        if adapter.is_symlink() or not adapter.is_file():
            raise SetupError('The local MCP adapter is missing. Nothing was registered.')
        server = (str(executable), str(adapter.absolute()))
    target = Path(session_path) if session_path else default_path()
    if not target.is_absolute() or target.is_symlink():
        raise SetupError('An absolute regular owner-session path is required.')
    return SetupPlan(str(Path(codex).absolute()), server, str(target))


def _run(command, *, runner, **kwargs):
    try:
        return runner(command, capture_output=True, text=True, timeout=15,
                      shell=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                      **kwargs)
    except subprocess.TimeoutExpired:
        raise SetupError('The local Codex command timed out. Check setup before retrying; no automatic retry was made.') from None
    except OSError:
        raise SetupError('The local Codex command could not start. Check the Codex installation.') from None


def configuration_status(plan, *, runner=subprocess.run):
    result = _run([plan.codex, 'mcp', 'get', SERVER_NAME, '--json'], runner=runner)
    if result.returncode:
        if 'No MCP server named' in (result.stderr or ''):
            return {'state': 'not_configured', 'revision': 'absent', 'message': 'Not added to Codex on this PC.'}
        raise SetupError('Codex could not read its MCP configuration. Check permissions locally; diagnostic output was withheld.')
    try:
        document = json.loads(result.stdout)
        transport = document['transport']
    except (ValueError, TypeError, KeyError):
        raise SetupError('Codex returned an unsupported configuration response.') from None
    revision = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    same_command = (transport.get('type') == 'stdio' and
                    transport.get('command') == plan.server[0] and
                    (transport.get('args') or []) == list(plan.server[1:]))
    if not same_command:
        return {'state': 'conflict', 'revision': revision, 'message': 'A different adapter uses this server name. Replacement requires approval of the exact command above.'}
    if (transport.get('env') or {}).get(SESSION_ENV) != plan.session_path:
        return {'state': 'needs_update', 'revision': revision, 'message': 'Adapter configured, but its owner-session path needs updating.'}
    if document.get('enabled') is False:
        return {'state': 'disabled', 'revision': revision, 'message': 'The Codex MCP entry is disabled. Review it in Codex before enabling it.'}
    return {'state': 'configured', 'revision': revision, 'message': 'Added to Codex with the correct local owner-session path.'}


def probe_adapter(plan, *, runner=subprocess.run):
    """Initialize and list tools only. Never load a capability or contact the gateway."""
    requests = [
        {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05', 'capabilities': {}, 'clientInfo': {'name': 'provis-setup-check', 'version': '1'}}},
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}},
    ]
    env = {**os.environ, SESSION_ENV: plan.session_path, 'PYINSTALLER_RESET_ENVIRONMENT': '1'}
    result = _run(list(plan.server), runner=runner, env=env,
                  input=''.join(json.dumps(request) + '\n' for request in requests))
    try:
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        initialized = next(row['result'] for row in rows if row.get('id') == 1)
        tools = next(row['result']['tools'] for row in rows if row.get('id') == 2)
        valid = (result.returncode == 0 and initialized['protocolVersion'] == '2024-11-05'
                 and {tool['name'] for tool in tools} == EXPECTED_TOOLS)
    except (ValueError, KeyError, TypeError, StopIteration):
        valid = False
    if not valid:
        raise SetupError('The local MCP adapter did not pass its initialize/tools check. No gateway call or message was made.')
    return {'state': 'adapter_ready', 'message': 'Local MCP tools respond. Owner sign-in and gateway delivery are checked separately.'}


def register(plan, *, approved=False, owner_active=lambda: False, expected_revision=None,
             replace_existing=False, runner=subprocess.run):
    if approved is not True or not owner_active():
        raise SetupError('Current owner verification and explicit approval are required to change Codex configuration.')
    current = configuration_status(plan, runner=runner)
    if current['state'] == 'disabled' or (current['state'] == 'conflict' and not replace_existing):
        raise SetupError(current['message'])
    if current['state'] != 'configured' and expected_revision != current['revision']:
        raise SetupError('Codex setup changed since it was checked. Check and approve the new setup before retrying.')
    # Test local transport before making the persistent change.
    probe_adapter(plan, runner=runner)
    if not owner_active():
        raise SetupError('Owner verification expired. Nothing was registered.')
    if current['state'] != 'configured':
        result = _run(plan.command, runner=runner)
        if result.returncode:
            raise SetupError('Codex did not confirm registration. Check setup before retrying; no automatic retry was made.')
    result = configuration_status(plan, runner=runner)
    if result['state'] != 'configured':
        raise SetupError('Registration was not confirmed by readback. Check Codex before retrying.')
    return result
