"""Signed updater contract; fixture signing keys never leave tmp_path."""
import base64
import copy
import hashlib
import hmac
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from manuals_rag_inference import node_updater as updater
from manuals_rag_inference.node_release import canonical, verify_release, verify_artifact, ReleaseCatalog, FILENAMES
from manuals_rag_inference.node_update_queue import NodeUpdateQueue
from manuals_rag_inference.agent_link import AgentLink
from manuals_rag_inference.models import NodeSnapshot

OWNER='feb94da3-80b4-4f51-b6aa-3688b9b8e41e'

@pytest.fixture
def release(tmp_path):
    key=Ed25519PrivateKey.generate()
    keys={'fixture':base64.b64encode(key.public_key().public_bytes_raw()).decode()}
    content=b'signed-fixture-app'
    m={'format':1,'release_id':str(uuid.uuid4()),'version':'2026.10.16.2','platform':updater.platform_id(),
       'filename':FILENAMES[updater.platform_id()],'size':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    document={'manifest':m,'publisher':'fixture','signature':base64.b64encode(key.sign(canonical(m))).decode()}
    (tmp_path/(m['release_id']+'.json')).write_text(json.dumps(document),encoding='utf-8')
    (tmp_path/(m['release_id']+'.artifact')).write_bytes(content)
    return SimpleNamespace(document=document,manifest=m,keys=keys,content=content,catalog=ReleaseCatalog(tmp_path,keys))


def test_signed_artifact_and_catalog(release):
    doc,path=release.catalog.get(release.manifest['release_id'])
    verify_release(doc,release.keys,platform=updater.platform_id(),current_version='2026.10.16.1')
    verify_artifact(path,release.manifest)
    path.write_bytes(b'X'*len(release.content))
    with pytest.raises(PermissionError): release.catalog.get(release.manifest['release_id'])

@pytest.mark.parametrize('field,value',[('version','2026.10.16.3'),('filename','../../app.exe'),('size',300000000),('sha256','a'*64),('format',True)])
def test_tampered_manifest_rejected(release,field,value):
    doc=copy.deepcopy(release.document);doc['manifest'][field]=value
    with pytest.raises((ValueError,PermissionError)): verify_release(doc,release.keys)


def test_unknown_key_platform_and_downgrade_rejected(release):
    with pytest.raises(PermissionError): verify_release(release.document,{})
    with pytest.raises(PermissionError): verify_release(release.document,release.keys,platform='other')
    with pytest.raises(PermissionError): verify_release(release.document,release.keys,current_version='2026.10.16.2')
    with pytest.raises(ValueError): release.catalog.get('../../something')

@pytest.fixture
def queue(release):
    nodes=[SimpleNamespace(node_id=n,secret='secret-'+n,identity_subject='john',identity_tenant='tenant',enrollment_method='entra') for n in ('work','john')]
    seen=set()
    def claim(node,nonce,issued):
        if (node,nonce) in seen: raise PermissionError('replay')
        seen.add((node,nonce))
    enrollment=SimpleNamespace(nodes=lambda:nodes,claim_inference_request=claim)
    scheduler=SimpleNamespace(snapshots=lambda:tuple(NodeSnapshot(n.node_id,'http://node','tail',(),node_app_version='2026.10.16.1',node_app_platform=updater.platform_id(),remote_updates_enabled=True,observed_at_epoch=time.time()) for n in nodes))
    validator=SimpleNamespace(validate=lambda token:dict(object_id=OWNER if token=='owner' else 'other',subject='john',tenant_id='tenant',expires_at_epoch=time.time()+1000))
    link=AgentLink(validator,enrollment,owner_object_id=OWNER,path=':memory:')
    updates=NodeUpdateQueue(release.catalog,enrollment,scheduler,link.db,link.lock)
    link.updates=updates
    token=link.session('owner',{'node_id':'work'})['session_token']
    yield SimpleNamespace(updates=updates,link=link,token=token,nodes=nodes,scheduler=scheduler)
    link.db.close()


def signed_node(queue,action,node='john',**payload):
    body=canonical(dict(node_id=node,issued_at=time.time(),nonce=str(uuid.uuid4()),**payload))
    sig=hmac.new(('secret-'+node).encode(),body,hashlib.sha256).hexdigest()
    return queue.updates.node(action,body,sig)


def enqueue(queue,release,identifier=None):
    return queue.link.dispatch('request-update',queue.token,dict(id=identifier or str(uuid.uuid4()),node_id='john',release_id=release.manifest['release_id']))


def test_owner_only_request_and_idempotent(queue,release):
    with pytest.raises(PermissionError): queue.link.session('other',{'node_id':'work'})
    with pytest.raises(PermissionError): queue.link.dispatch('request-update','secret-work',{})
    result=enqueue(queue,release)
    assert enqueue(queue,release,result['id'])==result
    with pytest.raises(ValueError): enqueue(queue,release)


def test_claim_once_exact_node_and_terminal_status(queue,release):
    enqueue(queue,release)
    assert signed_node(queue,'poll',node='work')['update'] is None
    update=signed_node(queue,'poll')['update']
    assert signed_node(queue,'poll')['update'] is None
    with pytest.raises(PermissionError): signed_node(queue,'report',node='work',id=update['id'],claim=update['claim'],status='healthy')
    with pytest.raises(PermissionError): signed_node(queue,'report',id=update['id'],claim='wrong',status='healthy')
    signed_node(queue,'report',id=update['id'],claim=update['claim'],status='verified')
    with pytest.raises(PermissionError): signed_node(queue,'report',id=update['id'],claim=update['claim'],status='downloading')
    signed_node(queue,'report',id=update['id'],claim=update['claim'],status='healthy')
    with pytest.raises(PermissionError): signed_node(queue,'report',id=update['id'],claim=update['claim'],status='failed')


def test_replay_and_signature_required(queue):
    body=canonical(dict(node_id='john',issued_at=time.time(),nonce='one'))
    sig=hmac.new(b'secret-john',body,hashlib.sha256).hexdigest()
    with pytest.raises(PermissionError): queue.updates.node('poll',body,'wrong')
    assert queue.updates.node('poll',body,sig)=={'update':None}
    with pytest.raises(PermissionError): queue.updates.node('poll',body,sig)
    stale=canonical(dict(node_id='john',issued_at=0,nonce='stale'))
    with pytest.raises(PermissionError): queue.updates.node('poll',stale,hmac.new(b'secret-john',stale,hashlib.sha256).hexdigest())


def test_old_node_needs_bootstrap(queue,release):
    queue.scheduler.snapshots=lambda:(NodeSnapshot('john','http://node','tail',()),)
    with pytest.raises(PermissionError): enqueue(queue,release)


def test_root_for_bootstraps_missing_config_parent_and_preserves_existing_config(tmp_path):
    first_run_config=tmp_path/'first-run'/'Local'/'ProVIS'/'inference-node.json'
    assert not first_run_config.parent.exists()
    first_run_root=updater.root_for(first_run_config)
    assert first_run_root == first_run_config.parent/'app-updates'
    assert first_run_root.is_dir()

    existing_config=tmp_path/'existing'/'Local'/'ProVIS'/'inference-node.json'
    existing_config.parent.mkdir(parents=True)
    original=b'{"gateway_url":"https://fixture.invalid","node_id":"fixture"}'
    existing_config.write_bytes(original)
    existing_root=updater.root_for(existing_config)
    assert existing_root.is_dir()
    assert existing_config.read_bytes()==original


def plan_fixture(tmp_path,release):
    config=tmp_path/'employee.json';config.write_text('{"shared_secret":"fixture"}',encoding='utf-8')
    directory=updater.root_for(config)/release.manifest['release_id'];directory.mkdir()
    (directory/release.manifest['filename']).write_bytes(release.content)
    previous=tmp_path/'previous.exe';previous.write_bytes(b'old')
    plan={'release':release.document,'id':str(uuid.uuid4()),'claim':'fixture-claim','config_path':str(config),'gateway':'https://fixture.invalid','node_id':'john',
          'previous_app':str(previous),'previous_enabled':False,'parent_pid':1234,'probe_nonce':'fixture-nonce','previous_version':'2026.10.16.1'}
    updater.private_json(directory/'plan.json',plan)
    updater.private_json(directory/'config-backup.json',{'shared_secret':'fixture'})
    return config,directory,plan,{'public_keys':release.keys,'enabled':True,'version':'2026.10.16.1'}


def test_active_pointer_requires_signature_and_newer_version(tmp_path,release):
    config,directory,plan,settings=plan_fixture(tmp_path,release)
    assert updater.active_app(config,settings) is None
    updater.private_json(updater.root_for(config)/'active.json',{'release':release.document})
    assert updater.active_app(config,settings)==directory/release.manifest['filename']
    (directory/release.manifest['filename']).write_bytes(b'bad')
    with pytest.raises(PermissionError): updater.active_app(config,settings)


def test_helper_commits_only_after_reconnection_receipt(tmp_path,release,monkeypatch):
    config,directory,plan,settings=plan_fixture(tmp_path,release)
    monkeypatch.setattr(updater,'_pid_running',lambda pid:False)
    monkeypatch.setattr(updater,'ChildJob',lambda child:SimpleNamespace(kill=lambda:pytest.fail('healthy child killed'),close=lambda:None))
    reports=[];monkeypatch.setattr(updater,'_report_plan',lambda plan,status:reports.append(status))
    def launch(args,**kwargs):
        updater.private_json(directory/'healthy.json',{'nonce':plan['probe_nonce'],'version':release.manifest['version'],'node_id':'john'})
        return SimpleNamespace(poll=lambda:None)
    assert updater.helper(directory/'plan.json',config,settings,launch=launch)
    assert reports==['healthy']
    assert updater.active_app(config,settings)==directory/release.manifest['filename']


def test_final_update_report_retries_transient_gateway_failure(tmp_path,monkeypatch):
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'shared_secret':'fixture-secret'}),encoding='utf-8')
    plan={'config_path':str(config),'gateway':'https://gateway.fixture.invalid',
          'node_id':'john','id':'update-id','claim':'single-use-claim'}
    attempts=[]
    def send(gateway,secret,action,payload,*,timeout):
        attempts.append((gateway,secret,action,payload,timeout))
        if len(attempts)==1:
            raise updater.URLError('temporary network failure')
        return {'status':'healthy'}
    monkeypatch.setattr(updater,'request',send)
    monkeypatch.setattr(updater.time,'sleep',lambda _:None)

    assert updater._report_plan(plan,'healthy') is True
    assert len(attempts)==2
    assert attempts[-1][1:3]==(b'fixture-secret','report')
    assert attempts[-1][3]['status']=='healthy'
    assert all(item[4]==5 for item in attempts)


@pytest.mark.parametrize('status',[408,429,503])
def test_final_update_report_retries_transient_http_status(tmp_path,monkeypatch,status):
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'shared_secret':'fixture-secret'}),encoding='utf-8')
    plan={'config_path':str(config),'gateway':'https://gateway.fixture.invalid',
          'node_id':'john','id':'update-id','claim':'single-use-claim'}
    attempts=[]
    def send(*args,**kwargs):
        attempts.append(True)
        if len(attempts)==1:
            raise updater.HTTPError('https://gateway.fixture.invalid',status,'temporary',{},None)
        return {'status':'healthy'}
    monkeypatch.setattr(updater,'request',send)
    monkeypatch.setattr(updater.time,'sleep',lambda _:None)

    assert updater._report_plan(plan,'healthy') is True
    assert len(attempts)==2


def test_final_update_report_does_not_retry_rejected_claim(tmp_path,monkeypatch):
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'shared_secret':'fixture-secret'}),encoding='utf-8')
    plan={'config_path':str(config),'gateway':'https://gateway.fixture.invalid',
          'node_id':'john','id':'update-id','claim':'single-use-claim'}
    attempts=[]
    def send(*args,**kwargs):
        attempts.append(True)
        raise updater.HTTPError('https://gateway.fixture.invalid',409,'claim rejected',{},None)
    monkeypatch.setattr(updater,'request',send)
    monkeypatch.setattr(updater.time,'sleep',lambda _:None)

    assert updater._report_plan(plan,'healthy') is False
    assert attempts==[True]


def test_final_update_report_preserves_local_receipt_after_total_network_failure(tmp_path,monkeypatch):
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'shared_secret':'fixture-secret'}),encoding='utf-8')
    receipt=tmp_path/'healthy.json'
    expected={'nonce':'probe-nonce','version':'2026.10.17.8','node_id':'john'}
    receipt.write_text(json.dumps(expected),encoding='utf-8')
    plan={'config_path':str(config),'gateway':'https://gateway.fixture.invalid',
          'node_id':'john','id':'update-id','claim':'single-use-claim'}
    attempts=[]
    def send(*args,**kwargs):
        attempts.append(True)
        raise updater.URLError('temporary network failure')
    monkeypatch.setattr(updater,'request',send)
    monkeypatch.setattr(updater.time,'sleep',lambda _:None)

    assert updater._report_plan(plan,'healthy') is False
    assert len(attempts)==3
    assert json.loads(receipt.read_text(encoding='utf-8'))==expected


def test_failed_start_relaunches_previous_and_restores_config(tmp_path,release,monkeypatch):
    config,directory,plan,settings=plan_fixture(tmp_path,release)
    config.write_text('{"broken":true}',encoding='utf-8')
    monkeypatch.setattr(updater,'_pid_running',lambda pid:False)
    stopped=[];monkeypatch.setattr(updater,'ChildJob',lambda child:SimpleNamespace(kill=lambda:stopped.append(True),close=lambda:None))
    launched=[]
    def launch(args,**kwargs): launched.append(args);return SimpleNamespace(poll=lambda:1)
    assert not updater.helper(directory/'plan.json',config,settings,launch=launch)
    assert stopped==[True]
    assert json.loads(config.read_text(encoding='utf-8'))=={'shared_secret':'fixture'}
    assert launched[-1][-2:]==['--update-rollback',str(directory/'plan.json')]
    assert str(tmp_path/'previous.exe') in launched[-1]
    assert not (updater.root_for(config)/'active.json').exists()


def test_redirects_rejected():
    with pytest.raises(PermissionError): updater.NoRedirect().redirect_request(None,None,302,'',{},'https://elsewhere.invalid')


def test_disabled_without_publisher_key_and_bad_build_config():
    from build_inference_node_bundle import managed_config
    assert updater.bootstrap({})['enabled'] is False
    assert updater.bootstrap({'app_update':{'enabled':True}})['enabled'] is False
    with pytest.raises(ValueError): managed_config('https://fixture.invalid',{'enabled':True,'version':'2026.10.16.2','public_keys':{'bad':'bad'}})


def test_update_drain_rejects_new_work_and_resume():
    from manuals_rag_inference.node_agent import OllamaNode
    node=OllamaNode(node_id='fixture',endpoint='http://127.0.0.1:1',tailnet_name='fixture',secret=b'fixture')
    node.begin_update()
    assert node.inference_enabled is False
    with pytest.raises(PermissionError): node.set_inference_enabled(True)
    with pytest.raises(PermissionError): node.command({},'bad')
    assert node._active_commands==0
    node.cancel_update(False)
    assert node.inference_enabled is False
    node.set_inference_enabled(True)
    assert node.inference_enabled


@pytest.mark.parametrize('busy', [False,True])
def test_worker_waits_for_busy_work_and_preserves_previous_pause(tmp_path,release,monkeypatch,busy):
    from manuals_rag_inference.node_agent import OllamaNode
    config,directory,plan,settings=plan_fixture(tmp_path,release)
    node=OllamaNode(node_id='john',endpoint='http://127.0.0.1:1',tailnet_name='fixture',secret=b'fixture')
    node.set_inference_enabled(False)
    monkeypatch.setattr(node,'publish_heartbeat',lambda gateway:{'status':'accepted'})
    update={'id':plan['id'],'claim':plan['claim'],'release':release.document}
    monkeypatch.setattr(updater,'request',lambda *a,**kw:{'update':update})
    monkeypatch.setattr(updater,'stage',lambda *a,**kw:directory)
    ticks=iter((0,121))
    monkeypatch.setattr(updater.time,'monotonic',lambda:next(ticks))
    worker=updater.UpdateWorker.__new__(updater.UpdateWorker)
    worker.agent,worker.gateway,worker.config_path,worker.settings=node,'https://fixture.invalid',str(config),settings
    worker.busy=lambda:busy
    emitted=[];worker.emit=lambda path:emitted.append(path)
    class Stop:
        polls=0
        def wait(self,seconds): self.polls+=1;return self.polls>1
        def is_set(self): return False
    worker.stop=Stop();worker.applying=False
    reports=[];worker.report=lambda update,stage:reports.append(stage)
    worker.run()
    if busy:
        assert emitted==[] and reports[-1]=='failed'
        assert node.inference_enabled is False and node._updating is False
    else:
        assert reports==['downloading','verified','waiting_for_idle','restarting']
        assert emitted==[directory/'plan.json']
        assert json.loads((directory/'plan.json').read_text(encoding='utf-8'))['previous_enabled'] is False


def test_http_updates_require_signed_node_and_registered_artifact(queue,release):
    from http.server import ThreadingHTTPServer
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from manuals_rag_inference.gateway import create_gateway_handler
    server=ThreadingHTTPServer(('127.0.0.1',0),create_gateway_handler(SimpleNamespace(agent_link=queue.link)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url='http://127.0.0.1:'+str(server.server_port)
    try:
        enqueue(queue,release)
        body=canonical(dict(node_id='john',issued_at=time.time(),nonce=str(uuid.uuid4())))
        with pytest.raises(HTTPError) as rejected: urlopen(Request(url+'/v1/nodes/updates/poll',data=body))
        assert rejected.value.code==403
        sig=hmac.new(b'secret-john',body,hashlib.sha256).hexdigest()
        with urlopen(Request(url+'/v1/nodes/updates/poll',data=body,headers={'X-Node-Signature':sig})) as response:
            assert json.load(response)['update']['release']==release.document
        with urlopen(url+'/v1/node-releases/'+release.manifest['release_id']) as response: assert response.read()==release.content
        with pytest.raises(HTTPError) as rejected: urlopen(url+'/v1/node-releases/not-a-uuid')
        assert rejected.value.code==404
    finally: server.shutdown();server.server_close();thread.join(2)


def test_signed_config_embedded_in_windows_bundle_and_linux_app(tmp_path,release):
    import zipfile
    from build_inference_node_bundle import build as bundle
    from build_inference_node_app import build as linux
    config=dict(enabled=True,version='2026.10.16.2',public_keys=release.keys)
    windows=bundle(tmp_path/'node.zip',gateway_url='https://fixture.invalid',update_config=config)
    app=linux(tmp_path/'node.pyz',gateway_url='https://fixture.invalid',update_config=config)
    for artifact,path in ((windows,'provis-inference-node/provis-node-config.json'),(app,'provis-node-config.json')):
        with zipfile.ZipFile(artifact) as archive:
            assert json.loads(archive.read(path))['app_update']==config
            assert b'PRIVATE KEY' not in archive.read(path)


@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows process containment')
def test_windows_candidate_job_contains_and_stops_only_created_process():
    import sys
    child,job=updater._start_candidate([sys.executable,'-c','import time; time.sleep(30)'],None)
    try:
        assert child.poll() is None
        job.kill()
        assert child.poll() is not None
    finally: job.close()


def test_launch_failure_recovers_previous_app(tmp_path,release,monkeypatch):
    config,directory,plan,settings=plan_fixture(tmp_path,release)
    monkeypatch.setattr(updater,'_pid_running',lambda pid:False)
    launched=[]
    def launch(args,**kw):
        launched.append(args)
        if len(launched)==1: raise OSError('fixture launch failed')
        return SimpleNamespace(poll=lambda:None)
    assert updater.helper(directory/'plan.json',config,settings,launch=launch) is False
    assert launched[-1][-2:]==['--update-rollback',str(directory/'plan.json')]


@pytest.fixture(scope='module')
def updater_gui_root(inference_gui_root):
    yield inference_gui_root


@pytest.mark.parametrize('theme',['light','dark'])
def test_owner_updates_panel_renders_metadata_only(release,theme,updater_gui_root):
    import tkinter as tk
    from manuals_rag_inference.agent_link_ui import UpdatePanel
    root=updater_gui_root
    root.app=SimpleNamespace(theme=theme,ui_font='Segoe UI')
    root.runtime=None
    root._work=lambda *args:None
    panel=UpdatePanel(root);panel.withdraw()
    try:
        panel.render(({'nodes':[{'node_id':'fixture-pc','version':'2026.10.16.1','platform':updater.platform_id(),'updates_enabled':True}]},
                      {'releases':[release.document]}, {'updates':[{'node_id':'fixture-pc','status':'verified','detail':''}]}))
        assert len(panel.nodes.get_children())==1 and len(panel.releases.get_children())==1
        assert panel.nodes.item(panel.nodes.get_children()[0])['values'][0]=='fixture-pc'
        assert 'verified' in panel.history.get('1.0','end')
    finally: panel.destroy()


def test_update_panel_refresh_discovers_nodes_releases_and_status(updater_gui_root):
    from manuals_rag_inference.agent_link_ui import UpdatePanel
    root=updater_gui_root
    root.app=SimpleNamespace(theme='light',ui_font='Segoe UI')
    requests=[]
    root.runtime=SimpleNamespace(request=lambda *args:requests.append(args) or {})
    root._work=lambda action,kind:action()
    panel=UpdatePanel(root);panel.withdraw()
    try:
        assert [request[0] for request in requests]==['update-nodes','update-releases','update-status']
        requests.clear()
        panel.refresh()
        assert [request[0] for request in requests]==['update-nodes','update-releases','update-status']
    finally: panel.destroy()


@pytest.mark.parametrize('compatible,confirmed,submitted',[(True,True,True),(True,False,False),(False,True,False)])
def test_update_panel_submits_only_compatible_confirmed_selection(
    release,updater_gui_root,monkeypatch,compatible,confirmed,submitted
):
    import tkinter.messagebox as messagebox
    from manuals_rag_inference.agent_link_ui import UpdatePanel
    root=updater_gui_root
    root.app=SimpleNamespace(theme='light',ui_font='Segoe UI')
    requests=[]
    root.runtime=SimpleNamespace(request=lambda *args:requests.append(args) or {'id':'fixture'})
    work=[]
    root._work=lambda action,kind:work.append((kind,action()))
    prompts=[]
    monkeypatch.setattr(messagebox,'askyesno',lambda *args,**kwargs:prompts.append('confirm') or confirmed)
    monkeypatch.setattr(messagebox,'showinfo',lambda *args,**kwargs:prompts.append('incompatible'))
    panel=UpdatePanel(root);panel.withdraw()
    try:
        # Ignore the constructor's initial read-only refresh.
        requests.clear();work.clear()
        panel.render(({'nodes':[{'node_id':'fixture-pc','version':'2026.10.16.1',
                                 'platform':release.manifest['platform'] if compatible else 'other-platform',
                                 'updates_enabled':True}]},
                      {'releases':[{'manifest':release.manifest,'publisher':'fixture'}]},
                      {'updates':[]}))
        panel.nodes.selection_set(panel.nodes.get_children()[0])
        panel.releases.selection_set(panel.releases.get_children()[0])
        panel.send()
        update_calls=[args for args in requests if args[0]=='request-update']
        assert bool(update_calls)==submitted
        assert bool(prompts and prompts[0]=='confirm')==compatible
        assert bool(prompts and prompts[0]=='incompatible')==(not compatible)
        if submitted:
            assert update_calls[0][1]['node_id']=='fixture-pc'
            assert update_calls[0][1]['release_id']==release.manifest['release_id']
            assert work[-1][0]=='update_requested'
    finally: panel.destroy()


def test_stage_download_checks_size_and_never_follows_redirect(tmp_path,release,monkeypatch):
    import io
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self,*args): self.close()
    seen=[]
    class Opener:
        def open(self,req,timeout):
            seen.append(req.full_url);return Response(release.content)
    monkeypatch.setattr(updater,'build_opener',lambda *args:Opener())
    settings={'public_keys':release.keys}
    directory=updater.stage({'release':release.document},config_path=tmp_path/'node.json',settings=settings,gateway='https://fixture.invalid',current_version='2026.10.16.1')
    assert (directory/release.manifest['filename']).read_bytes()==release.content
    assert seen==['https://fixture.invalid/v1/node-releases/'+release.manifest['release_id']]
    with pytest.raises(FileExistsError): updater.stage({'release':release.document},config_path=tmp_path/'node.json',settings=settings,gateway='https://fixture.invalid',current_version='2026.10.16.1')
