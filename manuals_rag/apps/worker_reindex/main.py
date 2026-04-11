from __future__ import annotations

from manuals_rag_common.logging import configure_logging
from manuals_rag_common.queue import dequeue, enqueue


def main() -> None:
    configure_logging()
    while True:
        job = dequeue("reindex_jobs", timeout=5)
        if not job:
            continue
        enqueue("ingest_jobs", job)


if __name__ == "__main__":
    main()
