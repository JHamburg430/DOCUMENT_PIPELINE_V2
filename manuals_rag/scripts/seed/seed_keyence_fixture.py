from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx


API_BASE = "http://127.0.0.1:8600"
AUTH = {"Authorization": "Bearer admin-token"}


def upload_and_ingest(pdf_path: Path) -> dict[str, str]:
    with httpx.Client(base_url=API_BASE, timeout=120.0) as client:
        with pdf_path.open("rb") as handle:
            response = client.post(
                "/documents/upload",
                headers=AUTH,
                files={"files": (pdf_path.name, handle, "application/pdf")},
            )
        response.raise_for_status()
        uploaded = response.json()["uploaded"][0]
        ingest = client.post(f"/documents/{uploaded['document_id']}/ingest", headers=AUTH)
        ingest.raise_for_status()
        run_id = ingest.json()["run_id"]
        for _ in range(90):
            run = client.get(f"/ingestion-runs/{run_id}", headers=AUTH)
            run.raise_for_status()
            payload = run.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(1)
    raise TimeoutError(f"Ingestion timed out for {pdf_path}")


def main() -> int:
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../Technical_Documents/Keyence/CA-EN100U_Datasheet.pdf")
    result = upload_and_ingest(pdf_path.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
