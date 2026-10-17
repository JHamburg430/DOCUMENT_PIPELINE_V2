#!/usr/bin/env python3
"""Publish an app-only signed release into an operator-managed release directory.

Use an existing Ed25519 PEM private key; never ship the private key with a node.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from manuals_rag_inference.node_release import canonical, FILENAMES, verify_release, verify_artifact


def publish(artifact, output, *, private_key, publisher, version, platform):
    key=load_pem_private_key(Path(private_key).read_bytes(),password=None)
    if not isinstance(key,Ed25519PrivateKey): raise ValueError('Ed25519 signing key required')
    artifact=Path(artifact)
    manifest={'format':1,'release_id':str(uuid.uuid4()),'version':version,'platform':platform,
              'filename':FILENAMES[platform],'size':artifact.stat().st_size,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    document={'manifest':manifest,'publisher':publisher,'signature':base64.b64encode(key.sign(canonical(manifest))).decode()}
    verify_release(document,{publisher:base64.b64encode(key.public_key().public_bytes_raw()).decode()})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/(manifest['release_id']+'.artifact')
    with artifact.open('rb') as source,target.open('xb') as dest: shutil.copyfileobj(source,dest)
    verify_artifact(target,manifest)
    with (output/(manifest['release_id']+'.json')).open('x',encoding='utf-8') as stream: json.dump(document,stream,indent=2)
    return manifest['release_id']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private-key',type=Path,required=True)
    parser.add_argument('--publisher',required=True)
    parser.add_argument('--version',required=True)
    parser.add_argument('--platform',choices=FILENAMES,required=True)
    args=parser.parse_args()
    print(publish(args.artifact,args.output,private_key=args.private_key,publisher=args.publisher,version=args.version,platform=args.platform))

if __name__=='__main__': main()
