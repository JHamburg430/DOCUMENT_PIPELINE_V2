from apps.worker_ingest.main import _docling_table_specs


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
