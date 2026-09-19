#!/usr/bin/env python3
"""Export benchmark-labelled evidence for source audit; never label it gold automatically."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from manuals_rag_common.db import fetch_all
from manuals_rag_evals.agent_eval_schema import build_expected_evidence_graph


def prepare(dataset: Path) -> dict:
    items = []
    for line in dataset.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case = raw.get('case', raw)
        graph = build_expected_evidence_graph(case)
        ids = sorted({chunk_id for node in graph.nodes for chunk_id in node.expected_chunk_ids})
        rows = fetch_all('''select id, source_document_id, document_version_id, title,
            page_from, page_to, section_path_text, content, metadata_json, is_active
            from retrieval_chunks where id = any(%s::text[])''', (ids,)) if ids else []
        found = {str(row['id']) for row in rows}
        items.append({'case_id': case['case_id'], 'query': case['query'],
                      'category': graph.category, 'expected_graph': graph.model_dump(),
                      'source_audit_status': 'pending',
                      'missing_chunk_ids': sorted(set(ids) - found), 'evidence': rows})
    return {'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'notice': 'Benchmark-labelled candidates, not independently validated gold evidence.',
            'items': items}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = prepare(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, default=str, indent=2) + '\n')
    print(json.dumps({'cases': len(report['items']), 'missing_evidence_cases': sum(bool(i['missing_chunk_ids']) for i in report['items']), 'output': str(args.output)}))
