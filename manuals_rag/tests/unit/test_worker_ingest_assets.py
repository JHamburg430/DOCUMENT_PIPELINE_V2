from apps.worker_ingest.main import _docling_table_specs, _metadata_extraction_payload
from manuals_rag_parsers.metadata import DocumentMetadata, METADATA_PIPELINE_VERSION
from manuals_rag_schemas.enums import DocumentKind


def test_docling_table_specs_resolve_batched_page_numbers_and_bboxes():
    artifact = {
        "batches": [
            {
                "page_range": [7, 9],
                "document": {
                    "tables": [
                        {"prov": [{"page_no": 1, "bbox": {"l": 10, "t": 20, "r": 110, "b": 80}}]},
                        {"prov": [{"page_no": 3, "bbox": {"l": 5, "t": 10, "r": 20, "b": 30}}]},
                    ]
                },
            }
        ]
    }

    specs = _docling_table_specs(artifact)

    assert specs == [
        {
            "table_index": 1,
            "batch_table_index": 1,
            "page": 7,
            "bbox": {"l": 10, "t": 20, "r": 110, "b": 80},
        },
        {
            "table_index": 2,
            "batch_table_index": 2,
            "page": 9,
            "bbox": {"l": 5, "t": 10, "r": 20, "b": 30},
        },
    ]


def test_metadata_extraction_payload_persists_claim_ledger_and_pipeline_version():
    metadata = DocumentMetadata(
        manufacturer="Unknown",
        companies=[],
        product_family=None,
        product_model=None,
        product_families=[],
        product_models=[],
        devices=[],
        part_numbers=[],
        protocol_terms=[],
        settings=[],
        parameters=[],
        menu_labels=[],
        document_topics=[],
        title="Manual",
        document_kind=DocumentKind.manual,
        revision_date=None,
        effective_date=None,
        metadata_claims=[{"value": "AX-420", "verification_status": "confirmed"}],
        metadata_pipeline_version=METADATA_PIPELINE_VERSION,
    )

    payload = _metadata_extraction_payload(metadata)

    assert payload["metadata_claims"] == metadata.metadata_claims
    assert payload["metadata_pipeline_version"] == METADATA_PIPELINE_VERSION
