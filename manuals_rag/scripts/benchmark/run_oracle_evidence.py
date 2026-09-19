#!/usr/bin/env python3
"""Isolate synthesis from retrieval using explicitly labelled source candidates.

This is a diagnostic, not an independently adjudicated accuracy benchmark.
Offline mode exercises graceful model-unavailability behavior only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
from contextlib import nullcontext
from manuals_rag_answering.generator import generate_answer_with_trace
from manuals_rag_schemas.documents import SearchResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    cases = json.loads(args.evidence.read_text())['items']
    if args.case_id:
        cases = [case for case in cases if case['case_id'] in args.case_id]
    if not cases:
        raise ValueError('No cases selected')
    report = {'mode': 'offline_model_unavailable' if args.offline else 'live_synthesis',
              'notice': 'Candidate-source synthesis diagnostic; not human-adjudicated accuracy.',
              'complete': False, 'expected_cases': len(cases), 'items': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = (patch('manuals_rag_answering.generator.chat_json', side_effect=RuntimeError('Offline diagnostic: model intentionally unavailable'))
               if args.offline else nullcontext())
    with context:
        for case in cases:
            print(json.dumps({'event': 'started', 'case_id': case['case_id']}), flush=True)
            started = perf_counter()
            item = {'case_id': case['case_id'], 'query': case['query']}
            try:
                if case.get('missing_chunk_ids'):
                    raise ValueError('Expected source chunks are missing')
                results = [SearchResult(chunk_id=e['id'], score=1, title=e['title'],
                    document_version_id=e['document_version_id'], source_document_id=e['source_document_id'],
                    pages=list(range(e['page_from'], e['page_to'] + 1)),
                    section_path=e['section_path_text'].split(' / '), content=e['content'],
                    metadata=e['metadata_json']) for e in case['evidence']]
                answer, trace = generate_answer_with_trace(case['query'], results)
                item.update(answer=answer.model_dump(), trace=trace,
                            evidence_ids=[result.chunk_id for result in results], status='returned')
            except Exception as exc:
                item.update(status='error', error=f'{type(exc).__name__}: {exc}')
            item['elapsed_seconds'] = round(perf_counter() - started, 3)
            report['items'].append(item)
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(report, indent=2, default=str))
            temporary.replace(args.output)
            print(json.dumps({'event': 'completed', 'case_id': case['case_id'], 'status': item['status']}), flush=True)
    report['complete'] = True
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, default=str))
    temporary.replace(args.output)
    if any(item['status'] == 'error' for item in report['items']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
