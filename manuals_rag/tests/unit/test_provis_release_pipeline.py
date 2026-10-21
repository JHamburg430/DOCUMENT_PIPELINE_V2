"""Offline publication contract; fixture keys/bytes and disposable catalogs only."""
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import uuid
from urllib.error import HTTPError
from email.message import Message

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from PyInstaller.archive.writers import CArchiveWriter

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'manuals_rag/scripts'))
import provis_release_pipeline as pipeline

@pytest.fixture
def bundle(tmp_path):
    key=Ed25519PrivateKey.generate()
    keys={'fixture':base64.b64encode(key.public_key().public_bytes_raw()).decode()}
    directory=tmp_path/'bundle';directory.mkdir()
    def sign(manifest):return {'manifest':manifest,'publisher':'fixture','signature':base64.b64encode(key.sign(pipeline.canonical(manifest))).decode()}
    index={'schema':1,'repository':pipeline.REPOSITORY,'version':'2026.10.21.9','tag':'provis-node/2026.10.21.9','source_commit':'a'*40,'artifacts':[]}
    app=None;app_document=None
    for slot,(kind,platform) in pipeline.SLOTS.items():
        data_name='provis-installer-binding.json' if kind=='installer' else 'provis-node-config.json'
        data={'schema':1,'app_release':app_document,'gateway_url':'https://gateway.fixture.example:8445'} if kind=='installer' else {'gateway_url':'https://gateway.fixture.example:8445','app_update':{'enabled':True,'version':index['version'],'public_keys':keys}}
        resource=tmp_path/(slot+'-binding.json');resource.write_bytes(pipeline.canonical(data))
        frozen=tmp_path/(slot+'-archive')
        CArchiveWriter(str(frozen),[(data_name,str(resource),True,'x')],'python311.dll')
        raw=frozen.read_bytes()
        manifest={'format':1,'release_id':str(uuid.uuid4()),'version':index['version'],'platform':platform,'filename':'PROVISInferenceNode.exe' if platform=='windows-x86_64' else 'PROVISInferenceNode','size':len(raw),'sha256':pipeline.sha(raw)}
        if kind=='installer':manifest.update(artifact_type='provis-node-installer',filename='PROVISInferenceNodeInstaller.exe',app_release_id=app['release_id'],app_sha256=app['sha256'])
        if slot=='windows-app':app=manifest
        signed=sign(manifest);doc=pipeline.canonical(signed)
        if slot=='windows-app':app_document=signed
        metadata,artifact=slot+'.json',slot+'.bin'
        (directory/metadata).write_bytes(doc);(directory/artifact).write_bytes(raw)
        index['artifacts'].append({'slot':slot,'manifest':metadata,'artifact':artifact,'manifest_sha256':pipeline.sha(doc),'artifact_sha256':pipeline.sha(raw)})
    def approve():(directory/'approved-release.json').write_bytes(pipeline.canonical(sign(index)))
    approve()
    for name in ('apps','installers','ledger'):(tmp_path/name).mkdir()
    return directory,keys,index,approve,sign

class FakeGitHub:
    def __init__(self):self.published=False;self.stored={};self.fail=None;self.count=0
    def draft(self,index,receipt):return {'id':17,'draft':not self.published,'tag_name':index['tag'],'body':receipt}
    def assets(self,release,assets):
        for name,raw in assets.items():
            assert type(raw) is bytes
            if name in self.stored:assert self.stored[name]==raw
            self.stored[name]=raw
        if self.fail=='assets':raise RuntimeError('fixture interruption')
    def publish(self,release,index):
        if self.fail=='publish':raise RuntimeError('fixture finalization interrupted')
        self.published=True;self.count+=1

def run(bundle,tmp_path,github,**overrides):
    directory,keys,*_=bundle
    verified=pipeline.verify_bundle(directory,keys)
    args=dict(app_root=tmp_path/'apps',installer_root=tmp_path/'installers',ledger=tmp_path/'ledger',github=github,approval_sha256=verified['approval_sha256'])
    args.update(overrides)
    return pipeline.publish(verified,keys,**args)

def test_complete_signed_set_and_same_approved_bytes_publish_to_both_sinks(bundle,tmp_path):
    github=FakeGitHub();receipt=run(bundle,tmp_path,github)
    assert receipt['complete'] and github.published
    verified=pipeline.verify_bundle(bundle[0],bundle[1])
    assert github.stored==verified['assets']
    for obj in verified['objects'].values():
        root=tmp_path/('apps' if obj['kind']=='app' else 'installers')
        assert (root/(obj['manifest']['release_id']+'.artifact')).read_bytes()==obj['artifact']
        assert (root/(obj['manifest']['release_id']+'.json')).read_bytes()==obj['metadata']
    assert (tmp_path/'ledger'/('complete-'+receipt['version']+'.json')).exists()

def test_idempotent_reconciliation_preserves_exact_files(bundle,tmp_path):
    github=FakeGitHub();first=run(bundle,tmp_path,github)
    before={p:p.read_bytes() for d in ('apps','installers','ledger') for p in (tmp_path/d).iterdir()}
    assert run(bundle,tmp_path,github)==first
    assert all(p.read_bytes()==raw for p,raw in before.items())

@pytest.mark.parametrize('phase',['assets','publish'])
def test_partial_publication_resumes_without_overwrite(bundle,tmp_path,phase):
    github=FakeGitHub();github.fail=phase
    with pytest.raises(RuntimeError):run(bundle,tmp_path,github)
    if phase=='assets':assert not list((tmp_path/'apps').iterdir())
    else:assert len(list((tmp_path/'apps').glob('*.json')))==2
    assert not list((tmp_path/'ledger').glob('complete-*'))
    before={p:p.read_bytes() for d in ('apps','installers','ledger') for p in (tmp_path/d).iterdir()}
    github.fail=None;assert run(bundle,tmp_path,github)['complete']
    assert all(p.read_bytes()==raw for p,raw in before.items())

def test_different_signed_source_cannot_reuse_version_claim(bundle,tmp_path):
    github=FakeGitHub();run(bundle,tmp_path,github)
    bundle[2]['source_commit']='b'*40;bundle[3]()
    with pytest.raises(FileExistsError):run(bundle,tmp_path,github)

def test_exact_approval_digest_required_before_channel_writes(bundle,tmp_path):
    with pytest.raises(PermissionError):run(bundle,tmp_path,FakeGitHub(),approval_sha256='0'*64)
    assert not list((tmp_path/'apps').iterdir()) and not list((tmp_path/'ledger').iterdir())

@pytest.mark.parametrize('mutation',['signature','artifact','manifest','source','missing','duplicate','traversal','reserved','binding','version','publisher'])
def test_tamper_and_ambiguity_rejected(bundle,mutation):
    directory,keys,index,approve,sign=bundle
    item=index['artifacts'][0]
    if mutation=='signature':
        doc=json.loads((directory/'approved-release.json').read_bytes());doc['signature']='AAAA';(directory/'approved-release.json').write_text(json.dumps(doc))
    elif mutation=='artifact':(directory/item['artifact']).write_bytes(b'bad')
    elif mutation=='manifest':(directory/item['manifest']).write_bytes(b'{}')
    elif mutation=='source':index['source_commit']='not-a-sha';approve()
    elif mutation=='missing':index['artifacts'].pop();approve()
    elif mutation=='duplicate':index['artifacts'][1]=copy.deepcopy(item);approve()
    elif mutation=='traversal':item['artifact']='../outside';approve()
    elif mutation=='reserved':item['artifact']='SHA256SUMS';approve()
    elif mutation=='version':index['version']='2026.10.21';index['tag']='provis-node/2026.10.21';approve()
    elif mutation=='publisher':keys.clear()
    elif mutation=='binding':
        last=index['artifacts'][-1];doc=json.loads((directory/last['manifest']).read_bytes());doc['manifest']['app_sha256']='0'*64
        raw=pipeline.canonical(sign(doc['manifest']));(directory/last['manifest']).write_bytes(raw);last['manifest_sha256']=pipeline.sha(raw);approve()
    with pytest.raises((ValueError,PermissionError,KeyError)):pipeline.verify_bundle(directory,keys)

def test_rechecks_changed_bundle_and_never_overwrites_partial_destination(bundle,tmp_path):
    verified=pipeline.verify_bundle(bundle[0],bundle[1]);item=bundle[2]['artifacts'][0]
    (bundle[0]/item['artifact']).write_bytes(b'changed after verification')
    with pytest.raises(PermissionError):pipeline.publish(verified,bundle[1],app_root=tmp_path/'apps',installer_root=tmp_path/'installers',ledger=tmp_path/'ledger',github=FakeGitHub(),approval_sha256=verified['approval_sha256'])
    assert not list((tmp_path/'ledger').iterdir())
    target=tmp_path/'partial';target.write_bytes(b'original')
    with pytest.raises(FileExistsError):pipeline.append_exact(target,b'replacement')
    assert target.read_bytes()==b'original'

def test_existing_signed_version_conflict_detected_before_github(bundle,tmp_path):
    run(bundle,tmp_path,FakeGitHub())
    directory,keys,index,approve,sign=bundle
    item=index['artifacts'][1];doc=json.loads((directory/item['manifest']).read_bytes());doc['manifest']['release_id']=str(uuid.uuid4())
    raw=pipeline.canonical(sign(doc['manifest']));(directory/item['manifest']).write_bytes(raw);item['manifest_sha256']=pipeline.sha(raw);approve()
    with pytest.raises(FileExistsError):run(bundle,tmp_path,FakeGitHub())

def test_catalogs_and_ledger_must_be_isolated(bundle,tmp_path):
    with pytest.raises(ValueError):run(bundle,tmp_path,FakeGitHub(),installer_root=tmp_path/'apps')

def test_parallel_publisher_lock_refuses_second_writer(tmp_path):
    with pipeline.publication_lock(tmp_path):
        with pytest.raises(OSError):
            with pipeline.publication_lock(tmp_path):pass

class Response(io.BytesIO):
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def test_github_upload_returns_json_and_asset_redirect_does_not_forward_token():
    calls=[]
    class Opener:
        def open(self,request,**kwargs):
            calls.append(request)
            if len(calls)==1:return Response(b'{"id":8}')
            if len(calls)==2:
                headers=Message();headers['Location']='https://release-assets.githubusercontent.com/fixture?signature=temporary'
                raise HTTPError(request.full_url,302,'redirect',headers,None)
            return Response(b'fixture')
    github=pipeline.GitHub('private-fixture-token');github.opener=Opener()
    assert github.request('/releases/1/assets?name=test',method='POST',body=b'fixture',binary=True,upload=True)=={'id':8}
    assert github.request('/releases/assets/8',binary=True)==b'fixture'
    assert calls[1].get_header('Accept')=='application/octet-stream'
    assert calls[2].get_header('Authorization') is None

@pytest.mark.parametrize('location',['http://release-assets.githubusercontent.com/file','https://evil.invalid/file','https://user@release-assets.githubusercontent.com/file','https://githubusercontent.com.evil.invalid/file'])
def test_untrusted_download_redirect_refused(location):
    class Opener:
        def open(self,request,**kwargs):
            headers=Message();headers['Location']=location
            raise HTTPError(request.full_url,302,'redirect',headers,None)
    github=pipeline.GitHub('fixture');github.opener=Opener()
    with pytest.raises(PermissionError):github.request('/releases/assets/1',binary=True)

def test_exact_tag_namespace_and_annotated_tag_peeling():
    github=pipeline.GitHub('fixture');calls=[]
    def request(path,**kwargs):
        calls.append(path)
        return {'ref':'refs/tags/provis-node/2026.10.21.9','object':{'type':'tag','sha':'b'*40}} if len(calls)==1 else {'object':{'type':'commit','sha':'a'*40}}
    github.request=request;github.tag('provis-node/2026.10.21.9','a'*40)
    assert calls==['/git/ref/tags/provis-node%2F2026.10.21.9','/git/tags/'+'b'*40]

def test_published_release_missing_assets_is_never_modified():
    github=pipeline.GitHub('fixture');github.request=lambda *args,**kwargs:[]
    with pytest.raises(FileExistsError):github.assets({'id':1,'draft':False},{'app':b'fixture'})

def test_workflow_bootstrap_gates_permissions_and_triggers():
    ci=json.loads((ROOT/'.github/workflows/provis-inference-node-ci.yml').read_text())
    release=json.loads((ROOT/'.github/workflows/provis-release-publication.yml').read_text())
    assert set(ci['on'])=={'pull_request','push','workflow_dispatch'}
    assert set(release['on'])=={'workflow_dispatch'}
    for workflow in (ci,release):
        assert workflow['permissions']=={'contents':'read'}
        for job in workflow['jobs'].values():
            assert 'false' in job['if']
            for step in job['steps']:
                if 'uses' in step:assert len(step['uses'].split('@')[1])==40
                if step.get('uses','').startswith('actions/checkout@'):assert step['with']['persist-credentials'] is False
    assert ci['jobs']['test-and-package']['strategy']['matrix']['os']==['ubuntu-22.04','windows-2022']
    ci_text=json.dumps(ci)
    assert 'secrets.' not in ci_text and 'self-hosted' not in ci_text
    assert 'requirements-inference-onnx.txt' in ci_text
    publish=release['jobs']['publish']
    assert publish['environment']=='provis-release-publication'
    assert 'refs/heads/main' in publish['if']
    assert publish['steps'][0]['with']['ref']=='main'
    assert set(release['on']['workflow_dispatch']['inputs'])=={'version','approval_sha256'}
    assert release['concurrency']['cancel-in-progress'] is False
    script=publish['steps'][1]['run']
    assert '${{ inputs.' not in script  # Untrusted strings only enter through environment and quoted args.
    assert 'provis_release_pipeline.py verify' in script and 'publish --execute' in script

def test_root_instruction_links_and_immutable_architecture_policy():
    import subprocess
    existing=subprocess.check_output(['git','show','b70537f96c928d32dd72dd28df5882a1f710c1e3:.AGENT.md'],cwd=ROOT)
    assert (ROOT/'.AGENT.md').read_bytes().replace(b'\r\n',b'\n')==existing.replace(b'\r\n',b'\n')
    for name in ('AGENTS.md','CONTRIBUTING.md','README.md'):
        text=(ROOT/name).read_text()
        assert 'provis_inference_cicd.md' in text
    assert 'Historical pre-consolidation' in (ROOT/'manuals_rag/docs/runbooks/provis_node_release_pipeline.md').read_text()

def test_published_asset_content_conflict_does_not_upload_or_delete():
    github=pipeline.GitHub('fixture')
    calls=[]
    def request(path,**kwargs):
        calls.append((path,kwargs))
        return [{'id':1,'name':'app'}] if 'per_page' in path else b'different bytes'
    github.request=request
    with pytest.raises(PermissionError):github.assets({'id':17,'draft':True},{'app':b'approved bytes'})
    assert len(calls)==2 and all(kwargs.get('method','GET')=='GET' for _,kwargs in calls)

def test_bad_tag_or_ref_namespace_is_rejected():
    github=pipeline.GitHub('fixture')
    github.request=lambda *args,**kwargs:{'ref':'refs/heads/provis-node/2026.10.21.9','object':{'type':'commit','sha':'a'*40}}
    with pytest.raises(PermissionError):github.tag('provis-node/2026.10.21.9','a'*40)

def test_catalog_capacity_is_checked_without_mutating_files(bundle,tmp_path):
    # Foreign files deliberately do not consume schema capacity; invalid metadata does.
    for n in range(64):(tmp_path/'apps'/str(n)).write_bytes(b'preserve')
    verified=pipeline.verify_bundle(bundle[0],bundle[1])
    assert pipeline.catalog_preflight(tmp_path/'apps',[o for o in verified['objects'].values() if o['kind']=='app'],bundle[1])
    for n in range(64):(tmp_path/'apps'/(str(n)+'.json')).write_bytes(b'{}')
    with pytest.raises(ValueError,match='capacity'):run(bundle,tmp_path,FakeGitHub())
    assert all((tmp_path/'apps'/str(n)).read_bytes()==b'preserve' for n in range(64))

def test_cli_verification_is_read_only(bundle,tmp_path,capsys,monkeypatch):
    keys=tmp_path/'trusted-keys.json';keys.write_text(json.dumps(bundle[1]))
    before={p:p.read_bytes() for p in bundle[0].iterdir()}
    monkeypatch.setattr(sys,'argv',['provis_release_pipeline.py','verify','--bundle',str(bundle[0]),'--public-keys',str(keys)])
    pipeline.main()
    assert json.loads(capsys.readouterr().out)['verified'] is True
    assert before=={p:p.read_bytes() for p in bundle[0].iterdir()}
    assert not list((tmp_path/'apps').iterdir()) and not list((tmp_path/'ledger').iterdir())

def test_incomplete_staging_never_exposes_partial_metadata(tmp_path,monkeypatch):
    target=tmp_path/'release.json'
    real=pipeline.os.fsync
    def fail(descriptor):raise OSError('fixture disk failure')
    monkeypatch.setattr(pipeline.os,'fsync',fail)
    with pytest.raises(OSError):pipeline.append_exact(target,b'complete bytes')
    assert not target.exists()
    staged=list(tmp_path.glob('.pending-*'));assert len(staged)==1
    monkeypatch.setattr(pipeline.os,'fsync',real)
    pipeline.append_exact(target,b'complete bytes')
    assert target.read_bytes()==b'complete bytes' and staged[0].exists()
    assert target.stat().st_nlink==1

def test_catalog_publication_order_does_not_depend_on_index_order(bundle,tmp_path,monkeypatch):
    bundle[2]['artifacts'].reverse();bundle[3]()
    paths=[];append=pipeline.append_exact
    def record(path,raw):
        paths.append(Path(path));return append(path,raw)
    monkeypatch.setattr(pipeline,'append_exact',record)
    run(bundle,tmp_path,FakeGitHub())
    catalog_paths=[p.parent.name for p in paths if p.parent in (tmp_path/'apps',tmp_path/'installers')]
    assert catalog_paths==['apps','apps','apps','apps','installers','installers']

def test_installer_builder_stages_exact_public_binding_without_source_changes(bundle,tmp_path):
    import ast
    from build_provis_node_windows_installer import stage_release,INSTALLER_ROOT
    source=INSTALLER_ROOT/'provis_node_windows_installer.py';before=source.read_bytes()
    document=json.loads((bundle[0]/'windows-app.json').read_bytes())
    entry,binding=stage_release(tmp_path/'staged',document,bundle[1],'https://gateway.fixture.example:8445')
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse((entry.parent/source.name).read_text()).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PUBLISHER','PUBLIC_KEYS','RELEASE_DOCUMENT','GATEWAY_BASE')}
    assert assignments['RELEASE_DOCUMENT']==document and assignments['PUBLIC_KEYS']==bundle[1]
    assert json.loads(binding.read_bytes())['app_release']==document
    assert source.read_bytes()==before

@pytest.mark.parametrize('case',['app-version','smoke-gateway','gateway-mismatch','installer-binding'])
def test_frozen_resource_mismatch_rejected_even_with_valid_outer_signatures(bundle,tmp_path,case):
    directory,keys,index,approve,sign=bundle
    slot='windows-installer' if case=='installer-binding' else 'linux-app'
    item=next(i for i in index['artifacts'] if i['slot']==slot)
    member='provis-installer-binding.json' if slot=='windows-installer' else 'provis-node-config.json'
    data=pipeline.frozen_json((directory/item['artifact']).read_bytes(),member)
    if case=='app-version':data['app_update']['version']='2026.10.21.8'
    elif case=='smoke-gateway':data['gateway_url']='https://gateway.invalid:8445'
    elif case=='gateway-mismatch':data['gateway_url']='https://other.fixture.example:8445'
    else:data['app_release']['manifest']['release_id']=str(uuid.uuid4())
    resource=tmp_path/'changed-resource';resource.write_bytes(pipeline.canonical(data))
    artifact=directory/item['artifact'];CArchiveWriter(str(artifact),[(member,str(resource),True,'x')],'python311.dll')
    raw=artifact.read_bytes();document=json.loads((directory/item['manifest']).read_bytes())
    document['manifest'].update(size=len(raw),sha256=pipeline.sha(raw))
    metadata=pipeline.canonical(sign(document['manifest']));(directory/item['manifest']).write_bytes(metadata)
    item.update(manifest_sha256=pipeline.sha(metadata),artifact_sha256=pipeline.sha(raw));approve()
    with pytest.raises(PermissionError):pipeline.verify_bundle(directory,keys)

def test_frozen_metadata_is_bounded_and_not_executed(tmp_path):
    resource=tmp_path/'large.json';resource.write_bytes(b'X'*20000)
    artifact=tmp_path/'archive';CArchiveWriter(str(artifact),[('binding.json',str(resource),True,'x')],'python311.dll')
    with pytest.raises(ValueError,match='Bounded'):pipeline.frozen_json(artifact.read_bytes(),'binding.json')

@pytest.mark.parametrize('pins',[{}, {'fixture':'-----BEGIN PRIVATE KEY-----'}, {'fixture':base64.b64encode(b'x'*64).decode()}, {'fixture':123}])
def test_only_bounded_public_verification_pins_are_accepted(pins):
    with pytest.raises(ValueError):pipeline.public_pins(pins)
