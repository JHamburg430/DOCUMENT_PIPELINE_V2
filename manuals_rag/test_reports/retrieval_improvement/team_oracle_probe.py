"""Task-specific, read-only oracle. Run only under coordinator's GPU lease."""
import json
from pathlib import Path
from time import monotonic
from manuals_rag_answering.agentic_retrieval import RetrievalHop, _assess_hop_evidence, verify_retrieval_claim
from manuals_rag_answering.generator import generate_answer_with_trace
from manuals_rag_schemas.documents import SearchResult

root = Path('test_reports/retrieval_improvement')
report = json.loads((root / 'oracle_candidates.json').read_text())
case = next(c for c in report['items'] if c['case_id'] == 'curated_cross_document_v2::6')
results = [SearchResult(chunk_id=e['id'], score=1, title=e['title'],
    document_version_id=e['document_version_id'], source_document_id=e['source_document_id'],
    pages=list(range(e['page_from'], e['page_to']+1)),
    section_path=e['section_path_text'].split(' / '), content=e['content'],
    metadata=e['metadata_json']) for e in case['evidence']]
output = {'case_id': case['case_id'], 'query': case['query'], 'evidence_ids': [r.chunk_id for r in results],
          'audit_scope': 'assistant audit of extracted source text, not PDF/human/held-out', 'stages': []}

def save():
    (root / 'team_oracle_probe.json').write_text(json.dumps(output, indent=2))

save()
try:
    for index, query in enumerate([
        'For CV-X482, what does the Condition list setting control?',
        'For LJ-X8000, what does the Standard Angle setting control?',
    ]):
        started = monotonic()
        print(json.dumps({'stage': 'verify', 'query': query}), flush=True)
        hop = RetrievalHop(hop_id=f'claim_{index+1}', objective=query, query=query)
        _, assessment = _assess_hop_evidence(query, results)
        verdict = verify_retrieval_claim(hop, query, results, assessment)
        output['stages'].append({'stage': 'verify', 'query': query, 'verdict': verdict,
                                 'elapsed_seconds': monotonic()-started})
        save()
    print(json.dumps({'stage': 'synthesis'}), flush=True)
    started = monotonic()
    answer, trace = generate_answer_with_trace(case['query'], results)
    output['stages'].append({'stage': 'synthesis', 'answer': answer.model_dump(), 'trace': trace,
                            'elapsed_seconds': monotonic()-started})
    output['completed'] = True
    save()
except Exception as exc:
    output['completed'] = False
    output['error'] = f'{type(exc).__name__}: {exc}'
    save()
    raise
