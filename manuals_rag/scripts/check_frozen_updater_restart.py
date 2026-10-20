"""Offline PyInstaller regression probe; no credentials or node listeners.

Freeze this script with inference_fabric/src on --paths and --collect-all
cryptography. Run the source script with --check <frozen executable> to compare
the legacy launch against the production independent-runtime environment.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    mode = sys.argv[1]
    if mode == '--check':
        results = {}
        with tempfile.TemporaryDirectory(prefix='provis-frozen-restart-') as folder:
            for launch_mode in ('inherited', 'independent'):
                output = Path(folder) / (launch_mode + '.json')
                subprocess.run([sys.argv[2], 'parent', launch_mode, str(output)],
                               check=True, timeout=45)
                deadline = time.monotonic() + 30
                while not output.exists() and time.monotonic() < deadline:
                    time.sleep(.1)
                results[launch_mode] = json.loads(output.read_text())
        assert results['independent']['signature_roundtrip'], results
        # Legacy behavior may vary with bootloader/platform. The fixed path
        # must always pass after the parent exits and cleans its extraction.
        print(json.dumps(results, sort_keys=True))
    elif mode == 'parent':
        from manuals_rag_inference.node_updater import independent_app_environment
        environment = independent_app_environment() if sys.argv[2] == 'independent' else os.environ.copy()
        subprocess.Popen([sys.executable, 'child', sys.argv[3]], env=environment,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    elif mode == 'child':
        time.sleep(5)
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            key = Ed25519PrivateKey.generate()
            message = b'PROVIS_FROZEN_RESTART_TEST'
            key.public_key().verify(key.sign(message), message)
            result = {'signature_roundtrip': True}
        except Exception as error:
            result = {'signature_roundtrip': False, 'category': type(error).__name__,
                      'missing_module': getattr(error, 'name', None)}
        Path(sys.argv[2]).write_text(json.dumps(result))
    else:
        raise ValueError('Expected --check, parent or child')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if len(sys.argv) == 4 and sys.argv[1] == 'parent':
            Path(sys.argv[3]).write_text(json.dumps({'parent_failed': type(error).__name__,
                                                    'detail': str(error)}))
        else:
            raise
