"""Node app updater. Signed artifacts only; no remote shell or general installer."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .node_release import DEFAULT_VERSION, canonical, verify_release, verify_artifact


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise PermissionError('Update redirects are not permitted')


def platform_id():
    machine = platform.machine().lower()
    machine = {'amd64':'x86_64', 'arm64':'aarch64'}.get(machine, machine)
    return ('windows' if os.name == 'nt' else sys.platform) + '-' + machine


def bootstrap(config):
    value = config.get('app_update', {})
    keys = value.get('public_keys', {}) if type(value) is dict else {}
    enabled = value.get('enabled') is True and type(keys) is dict and bool(keys)
    release_version = value.get('version', DEFAULT_VERSION)
    from .node_release import version
    version(release_version)
    return {'version': release_version, 'public_keys': keys, 'enabled': enabled}


def private_json(path, value):
    path = Path(path)
    if path.is_symlink(): raise PermissionError('Private regular file required')
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name+'.'+secrets.token_hex(8)+'.tmp')
    # Exclusive/no-follow creation: never follow a pre-created temporary symlink.
    with temp.open('x', encoding='utf-8') as stream:
        temp.chmod(0o600)
        json.dump(value, stream)
    os.replace(temp, path)
    path.chmod(0o600)


def root_for(config_path):
    root = Path(config_path).resolve().parent / 'app-updates'
    if root.is_symlink(): raise PermissionError('Private updater directory required')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise PermissionError('Private updater directory required')
    return root


def request(gateway, secret, action, payload, *, timeout=20):
    url = urlsplit(gateway)
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Managed HTTPS gateway required')
    body = canonical({**payload, 'nonce': secrets.token_hex(16), 'issued_at': time.time()})
    signature = hmac.new(secret, body, hashlib.sha256).hexdigest()
    req = Request(gateway.rstrip('/')+'/v1/nodes/updates/'+action, data=body,
                  headers={'Content-Type':'application/json', 'X-Node-Signature':signature})
    with build_opener(NoRedirect()).open(req, timeout=timeout) as response:
        raw = response.read(65537)
    if len(raw)>65536: raise ValueError('Bounded update response required')
    return json.loads(raw)


def command(app=None):
    if app is None:
        app = sys.executable if getattr(sys, 'frozen', False) else sys.argv[0]
    app = str(Path(app).resolve())
    if os.name != 'nt':
        with Path(app).open('rb') as source:
            if source.read(4) == b'\x7fELF':
                return [app]
    return [app] if app.lower().endswith('.exe') else [sys.executable, app]


def stage(update, *, config_path, settings, gateway, current_version):
    manifest = verify_release(update['release'], settings['public_keys'], platform=platform_id(), current_version=current_version)
    root = root_for(config_path)
    if len(list(root.iterdir())) >= 6: raise ValueError('Release retention limit; administrator cleanup required')
    directory = root / manifest['release_id']
    directory.mkdir(mode=0o700)  # no reuse of a partially applied release
    target = directory / manifest['filename']
    req = Request(gateway.rstrip('/')+'/v1/node-releases/'+manifest['release_id'])
    with build_opener(NoRedirect()).open(req, timeout=60) as response, target.open('xb') as output:
        target.chmod(0o600)
        size = 0
        deadline = time.monotonic()+300
        while chunk := response.read1(65536):
            if time.monotonic()>deadline: raise TimeoutError('App download exceeded time limit')
            size += len(chunk)
            if size > manifest['size']: raise PermissionError('Artifact exceeded signed size')
            output.write(chunk)
    verify_artifact(target, manifest)
    if os.name != 'nt':
        target.chmod(0o700)
    private_json(directory/'release.json', update['release'])
    return directory


class UpdateWorker:
    def __init__(self, agent, gateway, config_path, settings, busy, emit):
        self.agent, self.gateway, self.config_path = agent, gateway, config_path
        self.settings, self.busy, self.emit = settings, busy, emit
        self.stop = threading.Event()
        self.applying = False
        self.thread = threading.Thread(target=self.run, daemon=True, name='provis-app-update')
        self.thread.start()

    def report(self, update, status):
        request(self.gateway, self.agent.secret, 'report', {'node_id':self.agent.node_id, 'id':update['id'], 'claim':update['claim'], 'status':status})
        self.agent._record_activity('app_update', '', status)

    def run(self):
        while not self.stop.wait(15):
            update = None
            previous = self.agent.inference_enabled
            try:
                update = request(self.gateway, self.agent.secret, 'poll', {'node_id':self.agent.node_id}).get('update')
                if not update: continue
                self.applying = True
                self.report(update, 'downloading')
                directory = stage(update, config_path=self.config_path, settings=self.settings, gateway=self.gateway, current_version=self.settings['version'])
                self.report(update, 'verified')
                self.agent.begin_update()
                self.agent.publish_heartbeat(self.gateway)
                self.report(update, 'waiting_for_idle')
                deadline = time.monotonic()+120
                while True:
                    with self.agent._request_lock: active = sum(self.agent._active_by_model.values()) + self.agent._active_commands
                    if not active and not self.busy(): break
                    if time.monotonic()>deadline or self.stop.wait(1): raise TimeoutError('Node busy')
                if self.stop.is_set(): raise TimeoutError('App shutting down')
                config = Path(self.config_path).resolve()
                private_json(directory/'config-backup.json', json.loads(config.read_text(encoding='utf-8')))
                plan = {'release': update['release'], 'id':update['id'], 'claim':update['claim'],
                        'config_path':str(config), 'gateway':self.gateway, 'node_id':self.agent.node_id,
                        'previous_app':command()[-1], 'previous_enabled':previous, 'parent_pid':os.getpid(),
                        'probe_nonce':secrets.token_hex(32), 'previous_version':self.settings['version']}
                private_json(directory/'plan.json', plan)
                self.report(update, 'restarting')
                self.emit(directory/'plan.json')
                return
            except Exception:
                if update:
                    self.agent.cancel_update(previous)
                    try: self.report(update, 'failed')
                    except Exception: self.agent._record_activity('app_update', '', 'failed')
                self.applying = False


def validate_plan(plan_path, config_path, settings):
    path = Path(plan_path).resolve(strict=True)
    root = root_for(config_path)
    if path.name != 'plan.json' or path.parent.parent != root or path.is_symlink() or path.stat().st_size>32768:
        raise PermissionError('Private updater plan required')
    plan = json.loads(path.read_text(encoding='utf-8'))
    manifest = verify_release(plan['release'], settings['public_keys'], platform=platform_id(), current_version=plan['previous_version'])
    if path.parent.name != manifest['release_id'] or Path(plan['config_path']).resolve() != Path(config_path).resolve():
        raise PermissionError('Update plan path mismatch')
    verify_artifact(path.parent/manifest['filename'], manifest)
    return path, plan, manifest


def _pid_running(pid):
    if type(pid) is not int or pid <= 0: raise ValueError('Process ID required')
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x100000,False,pid)
        if not handle: return False
        try: return kernel.WaitForSingleObject(handle,0)==258
        finally: kernel.CloseHandle(handle)
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False


class ChildJob:
    """Only the process tree created by this helper can be stopped on failure."""
    def __init__(self, child):
        self.child, self.handle = child, None
        if os.name == 'nt':
            import ctypes
            self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p]
            self.kernel.CreateJobObjectW.restype=ctypes.c_void_p
            self.kernel.AssignProcessToJobObject.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
            self.kernel.TerminateJobObject.argtypes=[ctypes.c_void_p,ctypes.c_uint]
            self.kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            self.handle = self.kernel.CreateJobObjectW(None,None)
            if not self.handle or not self.kernel.AssignProcessToJobObject(self.handle,int(child._handle)):
                if self.handle: self.kernel.CloseHandle(self.handle)
                raise RuntimeError('Cannot contain new app process; inspect before retrying')
    def kill(self):
        if self.handle: self.kernel.TerminateJobObject(self.handle,1)
        elif self.child.poll() is None: self.child.terminate()
        self.child.wait(timeout=15)
    def close(self):
        if self.handle: self.kernel.CloseHandle(self.handle);self.handle=None
        if hasattr(self.child, "close_handle"): self.child.close_handle()


def _start_candidate(args, launch):
    if os.name != 'nt' or launch is not None:
        child=(launch or subprocess.Popen)(args)
        return child,ChildJob(child)
    # Assign the new Windows process to its job BEFORE allowing its bootloader
    # to start. Every GUI child is contained; rollback never kills another app.
    import _winapi
    import ctypes
    handle,thread,pid,_=_winapi.CreateProcess(args[0],subprocess.list2cmdline(args),None,None,False,
                                            4|subprocess.CREATE_NO_WINDOW,None,None,subprocess.STARTUPINFO())
    class Child:
        _handle=handle
        def poll(self):
            return _winapi.GetExitCodeProcess(handle) if _winapi.WaitForSingleObject(handle,0)==0 else None
        def wait(self,timeout):
            if _winapi.WaitForSingleObject(handle,int(timeout*1000))==258: raise subprocess.TimeoutExpired(args,timeout)
            return _winapi.GetExitCodeProcess(handle)
        def terminate(self): _winapi.TerminateProcess(handle,1)
        closed=False
        def close_handle(self):
            if not self.closed: _winapi.CloseHandle(handle);self.closed=True
    child=Child()
    try:
        job=ChildJob(child)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.ResumeThread.argtypes=[ctypes.c_void_p];kernel.ResumeThread.restype=ctypes.c_ulong
        if kernel.ResumeThread(int(thread))==0xffffffff: raise OSError('Cannot resume update process')
        return child,job
    except Exception:
        child.terminate();child.wait(timeout=15);_winapi.CloseHandle(handle)
        raise
    finally: _winapi.CloseHandle(thread)


def helper(plan_path, config_path, settings, *, launch=None, sleep=time.sleep):
    path, plan, manifest = validate_plan(plan_path,config_path,settings)
    deadline=time.monotonic()+60
    while _pid_running(plan['parent_pid']):
        if time.monotonic()>deadline:
            _report_plan(plan,'failed');return False
        sleep(.25)
    target=path.parent/manifest['filename']
    child=job=None
    committed=False
    try:
        child,job=_start_candidate(command(target)+['--update-probe',str(path)],launch)
        deadline=time.monotonic()+100
        while time.monotonic()<deadline and child.poll() is None:
            receipt=path.parent/'healthy.json'
            if receipt.exists():
                data=json.loads(receipt.read_text(encoding='utf-8'))
                if data == {'nonce':plan['probe_nonce'],'version':manifest['version'],'node_id':plan['node_id']}:
                    private_json(root_for(config_path)/'active.json', {'release':plan['release']})
                    committed=True
                    _report_plan(plan,'healthy')
                    return True
            sleep(.5)
    except Exception:
        pass
    finally:
        if job and committed: job.close()
    try:
        if job: job.kill()
        backup=json.loads((path.parent/'config-backup.json').read_text(encoding='utf-8'))
        private_json(config_path, backup)
        # Previous app reports rollback only after its own fresh heartbeat.
        flags={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        (launch or subprocess.Popen)(command(plan['previous_app'])+['--update-rollback',str(path)], **flags)
    except Exception: _report_plan(plan,'interrupted')
    finally:
        if job: job.close()
    return False


def _report_plan(plan, status):
    try:
        config=json.loads(Path(plan['config_path']).read_text(encoding='utf-8'))
    except Exception:
        return False
    payload={'node_id':plan['node_id'], 'id':plan['id'], 'claim':plan['claim'], 'status':status}
    for attempt in range(3):
        try:
            request(plan['gateway'], config['shared_secret'].encode(), 'report', payload, timeout=5)
            return True
        except HTTPError as exc:
            # Authentication, claim and transition errors are permanent. Only
            # ask the gateway again for a server-side failure.
            if exc.code < 500 and exc.code not in {408, 429}:
                break
        except (URLError, TimeoutError, ConnectionError, OSError):
            pass
        except Exception:
            break
        if attempt < 2:
            time.sleep(0.5 * (2 ** attempt))
    return False  # retain the local signed receipt; never expose credentials


def active_app(config_path, settings):
    pointer=root_for(config_path)/'active.json'
    if not pointer.exists(): return None
    if pointer.is_symlink() or pointer.stat().st_size>16384: raise PermissionError('Invalid active release')
    document=json.loads(pointer.read_text(encoding='utf-8'))['release']
    manifest=verify_release(document, settings['public_keys'],platform=platform_id())
    from .node_release import version
    if version(manifest['version']) <= version(settings['version']): return None
    artifact=root_for(config_path)/manifest['release_id']/manifest['filename']
    verify_artifact(artifact,manifest)
    return artifact
