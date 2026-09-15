import pytest
from manuals_rag_retrieval.applicability import version_relation, assess_metadata_applicability


@pytest.mark.parametrize('requested,expression,state', [
    ('2.0', '2', 'applicable'), ('3', '2', 'unknown'),
    ('2', '>=2', 'applicable'), ('1.9', '>=2', 'conflicting'),
    ('5', 'before 5.0', 'conflicting'), ('4.9', 'before 5', 'applicable'),
    ('3', '2 to 4', 'applicable'), ('5', '2 to 4', 'conflicting'),
    ('2', '4 to 2', 'unknown'), ('3', '2 or later', 'applicable'),
    ('3', 'vendor release A', 'unknown'),
])
def test_version_bounds(requested, expression, state):
    assert version_relation(requested, expression) == state


def metadata(subject='AX-20', **overrides):
    return {'firmware_applicability': [dict(subject=subject, version='>=2',
        relation='applies_to', grounded=True, confidence=.9,
        source_quote='AX-20 requires firmware 2 or later.', page_from=4, **overrides)]}


def test_grounded_matching_subject_and_alias():
    assert assess_metadata_applicability('AX20 firmware 1.0', metadata())['state'] == 'conflicting'


def test_external_subject_and_prefix_collision_stay_unknown():
    for query in ['AX-200 firmware 1.0', 'BX-20 firmware 1.0']:
        assert assess_metadata_applicability(query, metadata())['state'] == 'unknown'


def test_rejected_metadata_cannot_constrain():
    assert assess_metadata_applicability('AX-20 firmware 1', metadata(verification_status='rejected'))['state'] == 'unknown'


def test_comparison_does_not_cross_bind_versions():
    assert assess_metadata_applicability('AX-20 firmware 1 vs BX-20 firmware 3', metadata())['state'] == 'unknown'


def test_ordinary_query_remains_unconstrained():
    assert assess_metadata_applicability('How to configure AX-20?', metadata())['state'] == 'not_requested'


def test_claim_evidence_projection_requires_grounding_and_confirmation():
    from manuals_rag_retrieval.document_metadata import select_grounded_metadata_evidence
    confirmed = dict(kind='product_model', value='AB-200', relation='primary_product',
        grounded=True, confidence=.9, verification_status='confirmed', source_quote='AB-200 Manual', page_from=1)
    rejected = dict(confirmed, value='CD-300', verification_status='rejected')
    ungrounded = dict(confirmed, value='EF-400', grounded=False)
    evidence = select_grounded_metadata_evidence({'metadata_claims':[confirmed,rejected,ungrounded]},[8],'AB200')
    assert [claim['value'] for claim in evidence] == ['AB-200']
    assert evidence[0]['page_from'] == 1


def test_claim_projection_never_borrows_other_document_revision():
    from manuals_rag_retrieval.retriever import _attach_document_selection
    from manuals_rag_schemas.documents import SearchResult
    result = SearchResult(chunk_id='chunk',source_document_id='doc',document_version_id='new',
        title='manual',content='content',pages=[1],section_path=[],score=1,metadata={})
    claim = dict(kind='product_model', value='AB-200', relation='primary_product',grounded=True,
        confidence=.9,verification_status='confirmed',source_quote='AB-200 Manual',page_from=1)
    hit = {'source_document_id':'doc','payload':{'document_version_id':'old','metadata_claims':[claim]}}
    assert _attach_document_selection([result],[hit])[0].metadata['document_metadata_evidence'] == []
    hit['payload']['document_version_id'] = 'new'
    assert len(_attach_document_selection([result],[hit])[0].metadata['document_metadata_evidence']) == 1
