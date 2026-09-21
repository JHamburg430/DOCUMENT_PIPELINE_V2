import importlib.util
from pathlib import Path
import sys


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "rebuild_visual_page_index.py"
_SPEC = importlib.util.spec_from_file_location("rebuild_visual_page_index", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_page_uri_is_version_scoped_and_deterministic():
    document = {
        "tenant_id": "tenant",
        "source_document_id": "document",
        "document_version_id": "version",
    }

    assert _MODULE._page_uri(document, 12) == (
        "s3://manuals-artifacts/tenant/document-assets/document/version/page-images/page-0012.png"
    )
