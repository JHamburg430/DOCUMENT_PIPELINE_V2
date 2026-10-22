#!/usr/bin/env python3
"""Verify a pre-approved signed bundle; reconcile append-only dual publication.

No signing keys, build-on-publish, remote shell or implicit publication. The CLI
defaults to offline verification. Deployment of a gateway is a separate action.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import uuid
import zlib
from urllib.error import HTTPError
from urllib.parse import quote,urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'manuals_rag/packages/inference_fabric/src'))
from manuals_rag_inference.node_release import canonical,verify_release,verify_artifact,version
from manuals_rag_inference.installer_release import verify_installer_release,verify_installer_artifact

REPOSITORY='JHamburg430/DOCUMENT_PIPELINE_V2'
SLOTS={'windows-app':('app','windows-x86_64'),'linux-app':('app','linux-x86_64'),'windows-installer':('installer','windows-x86_64')}
TESTS=('test_inference_node_updates.py','test_inference_node_listener_refresh.py','test_inference_native_linux_updates.py',
 'test_inference_node_broker_gateway.py','test_installer_release.py','test_inference_node_registration_reuse.py',
 'test_inference_registration_proof.py','test_inference_node_connect_diagnostics.py','test_inference_fabric_entra.py',
 'test_inference_entra_contract.py','test_inference_model_preparation.py','test_inference_gpu_placement.py','test_provis_release_pipeline.py')

def sha(raw):return hashlib.sha256(raw).hexdigest()

def public_pins(keys):
    if type(keys) is not dict or not 1<=len(keys)<=16:raise ValueError('Bounded trusted public pin map required')
    for identifier,value in keys.items():
        if type(identifier) is not str or not re.fullmatch('[A-Za-z0-9_.-]{1,80}',identifier) or type(value) is not str:raise ValueError('Public verification pins required')
        try:decoded=base64.b64decode(value,validate=True)
        except Exception:raise ValueError('Public verification pins required') from None
        if len(decoded)!=32:raise ValueError('Ed25519 public verification pin required')
    return keys

def regular(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink!=1 or getattr(path.lstat(),'st_file_attributes',0)&0x400:raise ValueError('Regular unlinked file required')
    return path

def read(path,limit=65536):
    path=regular(path)
    if path.stat().st_size>limit:raise ValueError('Bounded document required')
    return path.read_bytes()

def leaf(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}',value):raise ValueError('Plain asset basename required')
    return value

def frozen_json(artifact,name):
    import tempfile
    with tempfile.TemporaryDirectory(prefix='provis-release-verify-') as temporary:
        path=Path(temporary)/'snapshot';path.write_bytes(artifact)
        return _frozen_json_path(path,name)

def _frozen_json_path(path,name):
    """Read one bounded public data member; never import/execute frozen code."""
    from PyInstaller.archive.readers import CArchiveReader
    archive=CArchiveReader(str(path));entry=archive.toc.get(name)
    if not entry or len(entry)!=5:raise PermissionError('Required frozen release binding missing')
    offset,length,size,compressed,kind=entry
    if not 0<=length<=16384 or not 0<=size<=16384 or compressed not in (0,1):raise ValueError('Bounded frozen metadata required')
    start=archive._start_offset+offset
    if start<0 or start+length>Path(path).stat().st_size:raise ValueError('Frozen metadata bounds invalid')
    with Path(path).open('rb') as stream:stream.seek(start);raw=stream.read(length)
    if compressed:
        decoder=zlib.decompressobj();raw=decoder.decompress(raw,16385)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:raise ValueError('Bounded metadata stream required')
    if len(raw)!=size or len(raw)>16384:raise ValueError('Frozen metadata size mismatch')
    return json.loads(raw)

def verify_bundle(bundle,keys):
    public_pins(keys)
    bundle=Path(bundle).resolve(strict=True)
    raw=read(bundle/'approved-release.json')
    approved=json.loads(raw)
    if type(approved) is not dict or set(approved)!={'manifest','publisher','signature'}:raise ValueError('Signed approval required')
    index=approved['manifest']
    if type(index) is not dict or set(index)!={'schema','repository','version','tag','source_commit','artifacts'}:raise ValueError('Exact provenance required')
    if type(index['schema']) is not int or index['schema']!=1 or index['repository']!=REPOSITORY:raise ValueError('Repository provenance mismatch')
    if '.'.join(map(str,version(index['version'])))!=index['version']:raise ValueError('Canonical four-part version required')
    if index['tag']!='provis-node/'+index['version'] or not re.fullmatch('[0-9a-f]{40}',index['source_commit']):raise ValueError('Immutable source/tag required')
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(base64.b64decode(keys[approved['publisher']],validate=True)).verify(base64.b64decode(approved['signature'],validate=True),canonical(index))
    except Exception:raise PermissionError('Approval signature invalid') from None
    if type(index['artifacts']) is not list or len(index['artifacts'])!=3:raise ValueError('Complete Windows/Linux/installer set required')
    objects={};assets={'approved-release.json':raw};names={'approved-release.json','SHA256SUMS'}
    for item in index['artifacts']:
        if type(item) is not dict or set(item)!={'slot','manifest','artifact','manifest_sha256','artifact_sha256'}:raise ValueError('Exact artifact descriptor required')
        slot=item['slot']
        if type(slot) is not str or slot not in SLOTS or slot in objects:raise ValueError('Unique supported slot required')
        metadata,artifact=leaf(item['manifest']),leaf(item['artifact'])
        if metadata in names or artifact in names or metadata==artifact:raise ValueError('Unique asset names required')
        names.update((metadata,artifact))
        docraw=read(bundle/metadata);document=json.loads(docraw)
        kind,platform=SLOTS[slot]
        manifest=verify_release(document,keys,platform=platform) if kind=='app' else verify_installer_release(document,keys)
        if manifest['version']!=index['version']:raise ValueError('App/installer version mismatch')
        if document['publisher']!=approved['publisher']:raise PermissionError('One approved publisher required')
        if sha(docraw)!=item['manifest_sha256'] or manifest['sha256']!=item['artifact_sha256']:raise PermissionError('Approved bytes mismatch')
        path=regular(bundle/artifact)
        (verify_artifact if kind=='app' else verify_installer_artifact)(path,manifest)
        artifactraw=read(path,256*1024*1024)
        if sha(artifactraw)!=manifest['sha256']:raise PermissionError('Artifact changed while reading')
        assets[metadata]=docraw;assets[artifact]=artifactraw
        objects[slot]={'manifest':manifest,'document':document,'metadata':docraw,'artifact':artifactraw,'kind':kind}
    app,installer=objects['windows-app']['manifest'],objects['windows-installer']['manifest']
    if (installer['app_release_id'],installer['app_sha256'])!=(app['release_id'],app['sha256']):raise PermissionError('Installer/app binding mismatch')
    gateways=[]
    for slot in ('windows-app','linux-app'):
        config=frozen_json(objects[slot]['artifact'],'provis-node-config.json')
        if type(config) is not dict or type(config.get('app_update')) is not dict or type(config.get('gateway_url')) is not str:raise ValueError('Exact frozen managed configuration required')
        settings=config.get('app_update',{});url=urlsplit(config.get('gateway_url',''))
        embedded=settings.get('public_keys',{})
        if type(embedded) is not dict:raise ValueError('Trusted embedded public pin map required')
        public_pins(embedded)
        if settings.get('enabled') is not True or settings.get('version')!=index['version'] or not embedded or embedded.get(approved['publisher'])!=keys[approved['publisher']] or any(keys.get(k)!=v for k,v in embedded.items()):raise PermissionError('Frozen app version/publisher mismatch')
        if url.scheme!='https' or not url.hostname or url.hostname.endswith('.invalid') or url.username or url.password or url.query or url.fragment:raise PermissionError('Approved managed HTTPS gateway required')
        if url.port is not None and not 1<=url.port<=65535:raise ValueError('Valid managed HTTPS port required')
        gateways.append(config['gateway_url'])
    if len(set(gateways))!=1:raise PermissionError('Managed gateway mismatch')
    binding=frozen_json(objects['windows-installer']['artifact'],'provis-installer-binding.json')
    if type(binding) is not dict or set(binding)!={'schema','app_release','gateway_url'} or type(binding['schema']) is not int or binding['schema']!=1 or canonical(binding['app_release'])!=canonical(objects['windows-app']['document']) or binding['gateway_url']!=gateways[0]:raise PermissionError('Frozen installer/app binding mismatch')
    ids=[o['manifest']['release_id'] for o in objects.values()]
    if len(set(ids))!=3:raise ValueError('Cross-channel release IDs must differ')
    checksums=''.join(f'{sha(value.read_bytes() if isinstance(value,Path) else value)}  {name}\n' for name,value in sorted(assets.items())).encode()
    assets['SHA256SUMS']=checksums
    return {'bundle':bundle,'index':index,'approval_sha256':sha(raw),'objects':objects,'assets':assets}

def append_exact(path,raw):
    """Existing exact bytes reconcile; different bytes never overwrite."""
    path=Path(path)
    if path.exists() or path.is_symlink():
        if read(path,max(len(raw),65536))!=raw:raise FileExistsError('Immutable destination conflict')
        return False
    temporary=path.parent/('.pending-'+uuid.uuid4().hex)
    with os.fdopen(os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o600),'wb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        if os.name=='nt':os.rename(temporary,path)  # Windows rename refuses an existing destination.
        else:os.link(temporary,path)  # POSIX exclusive visibility; never use replacing rename.
    except FileExistsError:
        if read(path,max(len(raw),65536))!=raw:raise
    if temporary.exists():temporary.unlink()  # Own staging alias only; failed/crashed stages remain preserved.
    if os.name!='nt':
        descriptor=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    return True

@contextmanager
def publication_lock(directory):
    directory=Path(directory).resolve(strict=True)
    if not directory.is_dir():raise ValueError('Approved ledger directory required')
    path=directory/'.publication.lock'
    if path.is_symlink():raise ValueError('Regular lock required')
    stream=os.fdopen(os.open(path,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600),'r+b')
    try:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Private regular lock required')
        if os.name=='nt':
            import msvcrt
            if stream.tell()==0:stream.write(b'0');stream.flush()
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield directory
    finally:stream.close()  # OS releases lock on success, failure or process exit.

def catalog_preflight(root,objects,keys):
    root=Path(root).resolve(strict=True)
    if not root.is_dir():raise ValueError('Approved catalog root required')
    metadata_files=list(root.glob('*.json'))
    new=sum(not (root/(obj['manifest']['release_id']+'.json')).exists() for obj in objects)
    if len(metadata_files)+new>64:raise ValueError('Catalog capacity exceeded')
    for metadata in metadata_files:
        document=json.loads(read(metadata,16384));manifest=document.get('manifest',{})
        verifier=verify_installer_release if objects[0]['kind']=='installer' else verify_release
        existing=verifier(document,keys)
        if existing['release_id']!=metadata.stem:raise PermissionError('Existing catalog identity mismatch')
        (verify_installer_artifact if objects[0]['kind']=='installer' else verify_artifact)(root/(metadata.stem+'.artifact'),existing)
        for obj in objects:
            proposed=obj['manifest']
            if version(existing['version'])==version(proposed['version']) and existing['platform']==proposed['platform']:
                if canonical(document)!=canonical(obj['document']):raise FileExistsError('Version already reserved by different signed bytes')
    for obj in objects:
        identifier=obj['manifest']['release_id']
        for suffix,expected in (('.json',obj['metadata']),('.artifact',obj['artifact'])):
            dest=root/(identifier+suffix)
            if dest.exists() or dest.is_symlink():
                if read(dest,max(len(expected),65536))!=expected:raise FileExistsError('Catalog asset conflict')
    return root

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class GitHub:
    def __init__(self,token):
        if not token:raise PermissionError('Approved GitHub publisher credential required')
        self.token=token;self.opener=build_opener(NoRedirect())
    def request(self,path,*,method='GET',body=None,binary=False,upload=False):
        prefix=('https://uploads.github.com' if upload else 'https://api.github.com')+'/repos/'+REPOSITORY
        if not path.startswith('/'):raise ValueError('Fixed repository endpoint required')
        download=binary and method=='GET'
        headers={'Authorization':'Bearer '+self.token,'Accept':'application/octet-stream' if download else 'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28','Content-Type':'application/octet-stream' if binary else 'application/json'}
        data=body if binary else canonical(body) if body is not None else None
        try:
            with self.opener.open(Request(prefix+path,data=data,headers=headers,method=method),timeout=60) as response:
                raw=response.read(256*1024*1024+1)
                if len(raw)>256*1024*1024:raise ValueError('Bounded GitHub response required')
                return raw if download else json.loads(raw)
        except HTTPError as error:
            if error.code==404 and method=='GET':return None
            if error.code==302 and download and re.fullmatch(r'/releases/assets/[0-9]+',path):
                location=error.headers.get('Location','');url=urlsplit(location)
                if url.scheme!='https' or not url.hostname or not url.hostname.endswith('.githubusercontent.com') or url.username or url.password or url.fragment or url.port not in (None,443):raise PermissionError('Untrusted asset redirect') from None
                # One GitHub signed-asset hop, with no publisher credential forwarded.
                try:
                    with self.opener.open(Request(location,headers={'Accept':'application/octet-stream'}),timeout=60) as response:
                        raw=response.read(256*1024*1024+1)
                        if len(raw)>256*1024*1024:raise ValueError('Bounded asset readback required')
                        return raw
                except HTTPError:raise RuntimeError('Asset redirect readback failed') from None
            raise RuntimeError('GitHub publication HTTP_'+str(error.code)) from None
    def tag(self,tag,source):
        ref=self.request('/git/ref/tags/'+quote(tag,safe=''))
        if not ref or ref.get('ref')!='refs/tags/'+tag:raise PermissionError('Approved tag does not exist')
        obj=ref.get('object',{})
        for _ in range(5):
            if obj.get('type')=='commit' and obj.get('sha')==source:return
            if obj.get('type')!='tag' or not re.fullmatch('[0-9a-f]{40}',obj.get('sha','')):break
            tagged=self.request('/git/tags/'+obj['sha'])
            obj=(tagged or {}).get('object',{})
        raise PermissionError('Existing tag does not resolve to approved source')
    def draft(self,index,receipt):
        tag=index['tag'];self.tag(tag,index['source_commit'])
        release=self.request('/releases/tags/'+quote(tag,safe=''))
        marker='PROVIS approved-release SHA256: '+receipt
        if release is None:
            release=self.request('/releases',method='POST',body={'tag_name':tag,'target_commitish':index['source_commit'],'name':tag,'body':marker,'draft':True,'prerelease':False})
        if release.get('tag_name')!=tag or release.get('body')!=marker:raise FileExistsError('Existing GitHub release provenance conflict')
        return release
    def assets(self,release,assets):
        existing=self.request('/releases/'+str(release['id'])+'/assets?per_page=100')
        if len(existing)>=100:raise ValueError('Unexpected release asset capacity')
        byname={item['name']:item for item in existing}
        if len(byname)!=len(existing) or set(byname)-set(assets):raise FileExistsError('Unexpected or duplicate release assets')
        for name,value in assets.items():
            raw=value.read_bytes() if isinstance(value,Path) else value
            item=byname.get(name)
            if item is None:
                if not release['draft']:raise FileExistsError('Published release is incomplete; never mutate it')
                item=self.request('/releases/'+str(release['id'])+'/assets?name='+quote(name,safe=''),method='POST',body=raw,binary=True,upload=True)
            fetched=self.request('/releases/assets/'+str(item['id']),binary=True)
            if fetched is None or sha(fetched)!=sha(raw):raise PermissionError('GitHub asset readback mismatch')
    def publish(self,release,index):
        self.tag(index['tag'],index['source_commit'])
        if release['draft']:self.request('/releases/'+str(release['id']),method='PATCH',body={'draft':False})
        current=self.request('/releases/'+str(release['id']))
        if not current or current.get('draft') is not False or current.get('tag_name')!=index['tag'] or current.get('body')!=release.get('body'):raise PermissionError('Published release readback mismatch')
        self.tag(index['tag'],index['source_commit'])

def publish(verified,keys,*,app_root,installer_root,ledger,github,approval_sha256):
    verified=verify_bundle(verified['bundle'],keys)  # Recheck exact approved immutable snapshot before any destination write.
    if approval_sha256!=verified['approval_sha256']:raise PermissionError('Exact approved bundle digest required')
    with publication_lock(ledger) as directory:
        app_root=Path(app_root).resolve(strict=True);installer_root=Path(installer_root).resolve(strict=True)
        roots=(app_root,installer_root,directory)
        if any(a==b or a in b.parents or b in a.parents for n,a in enumerate(roots) for b in roots[n+1:]):raise ValueError('Distinct isolated catalogs and ledger required')
        objects=[verified['objects'][slot] for slot in SLOTS]  # Apps before their installer binding is visible.
        for root,kind in ((app_root,'app'),(installer_root,'installer')):catalog_preflight(root,[obj for obj in objects if obj['kind']==kind],keys)
        claim=directory/('version-'+verified['index']['version']+'.json')
        provenance={'version':verified['index']['version'],'source_commit':verified['index']['source_commit'],'approval_sha256':verified['approval_sha256']}
        append_exact(claim,canonical(provenance))
        def evidence(stage,**details):append_exact(directory/('event-'+verified['index']['version']+'-'+stage+'.json'),canonical({**provenance,'stage':stage,**details}))
        release=github.draft(verified['index'],verified['approval_sha256'])
        evidence('01-draft-identified',github_release_id=release['id'])
        github.assets(release,verified['assets'])  # All GitHub draft bytes verified before catalog visibility.
        evidence('02-draft-assets-verified',github_release_id=release['id'])
        for position,obj in enumerate(objects):
            root=app_root if obj['kind']=='app' else installer_root
            identifier=obj['manifest']['release_id']
            append_exact(root/(identifier+'.artifact'),obj['artifact'])
            append_exact(root/(identifier+'.json'),obj['metadata'])  # Metadata is per-asset visibility point.
            evidence('03-catalog-'+str(position),release_id=identifier,artifact_sha256=obj['manifest']['sha256'])
        for root,kind in ((app_root,'app'),(installer_root,'installer')):catalog_preflight(root,[obj for obj in objects if obj['kind']==kind],keys)
        github.publish(release,verified['index'])
        receipt={'schema':1,'repository':REPOSITORY,'version':verified['index']['version'],'source_commit':verified['index']['source_commit'],'tag':verified['index']['tag'],'approval_sha256':verified['approval_sha256'],'github_release_id':release['id'],'complete':True}
        append_exact(directory/('complete-'+verified['index']['version']+'.json'),canonical(receipt))
        return receipt

def ci(packaging=False,evidence_dir=None):
    env={**os.environ,'PYTHONPATH':os.pathsep.join(str(ROOT/path) for path in ('manuals_rag/packages/inference_fabric/src','manuals_rag/scripts','provis_inference_node/src'))}
    evidence=Path(evidence_dir) if evidence_dir else None
    if evidence:evidence.mkdir(parents=True,exist_ok=True)
    report={'schema':1,'repository':REPOSITORY,'tested_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty_worktree':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),'platform':sys.platform,'packaging_requested':packaging,'production_publication':False,'steps':[],'complete':False}
    def run(name,args,**options):
        completed=subprocess.run(args,cwd=ROOT,env=env,**options)
        report['steps'].append({'name':name,'exit_code':completed.returncode})
        completed.check_returncode()
        return completed
    try:
        junit=['--junitxml='+str(evidence/'pytest-results.xml')] if evidence else []
        run('focused-tests',[sys.executable,'-m','pytest','-q',*junit,*[str(ROOT/'manuals_rag/tests/unit'/name) for name in TESTS],str(ROOT/'provis_inference_node/tests/test_registration_reuse.py')])
        if packaging:
            if os.name=='nt':
                run('installer-unit-gui-fixtures',[sys.executable,'-m','pytest','-q',str(ROOT/'manuals_rag/tests/unit/test_provis_node_windows_installer.py')])
                run('native-app-isolated-profile',[sys.executable,str(ROOT/'manuals_rag/scripts/verify_windows_node_postinstall_launch.py')])
            import tempfile
            with tempfile.TemporaryDirectory(prefix='provis-ci-') as temporary:
                binary=Path(temporary)/('PROVISInferenceNodeInstaller.exe' if os.name=='nt' else 'PROVISInferenceNode')
                builder='build_provis_node_windows_installer.py' if os.name=='nt' else 'build_inference_node_linux.py'
                run('native-packaging',[sys.executable,str(ROOT/'manuals_rag/scripts'/builder),'--output',str(binary),*([] if os.name=='nt' else ['--gateway-url','https://gateway.invalid:8445'])])
                if os.name=='nt':
                    run('installer-archive-contract',[sys.executable,str(ROOT/'manuals_rag/scripts/verify_provis_node_windows_installer_archive.py'),str(binary)])
                run('frozen-self-test',[str(binary),'--self-test'])
                report['smoke_artifact']={'sha256':sha(binary.read_bytes()),'size':binary.stat().st_size,'production_artifact':False}
        report['complete']=True
    finally:
        if evidence:append_exact(evidence/('ci-'+uuid.uuid4().hex+'.json'),canonical(report))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','publish','ci'))
    parser.add_argument('--bundle',type=Path);parser.add_argument('--public-keys',type=Path)
    parser.add_argument('--approval-sha256');parser.add_argument('--app-catalog',type=Path)
    parser.add_argument('--installer-catalog',type=Path);parser.add_argument('--ledger',type=Path)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--packaging',action='store_true')
    parser.add_argument('--evidence-dir',type=Path)
    args=parser.parse_args()
    if args.action=='ci':ci(args.packaging,args.evidence_dir);return
    if not args.bundle or not args.public_keys:parser.error('Bundle and externally trusted public keys required')
    keys=json.loads(read(args.public_keys));verified=verify_bundle(args.bundle,keys)
    if args.action=='verify':print(json.dumps({'verified':True,'version':verified['index']['version'],'source_commit':verified['index']['source_commit'],'approval_sha256':verified['approval_sha256']}));return
    if not args.execute or not all((args.app_catalog,args.installer_catalog,args.ledger,args.approval_sha256)):parser.error('Explicit execute, approval digest and all approved destinations required')
    result=publish(verified,keys,app_root=args.app_catalog,installer_root=args.installer_catalog,ledger=args.ledger,github=GitHub(os.getenv('PROVIS_GITHUB_PUBLISH_TOKEN')),approval_sha256=args.approval_sha256)
    print(json.dumps(result))

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,PermissionError,RuntimeError):
        raise SystemExit('Release pipeline stopped; preserve partial state and run verification/reconciliation. No automatic overwrite, deletion or retry.') from None
