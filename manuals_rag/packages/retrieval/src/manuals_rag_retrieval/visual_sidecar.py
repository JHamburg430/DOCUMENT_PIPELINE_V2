from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from qdrant_client import QdrantClient, models

from manuals_rag_common.config import settings
from manuals_rag_common.ids import deterministic_uuid


VISUAL_VECTOR_SIZE = 128


def visual_collection_name(corpus_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "_-" else "_" for char in corpus_id)
    return f"manuals_{safe}_visual_pages"


def visual_page_id(document_version_id: str, page: int) -> str:
    return deterministic_uuid(document_version_id, "visual-page", page)


class VisualEncoder(Protocol):
    def encode_query(self, queries: Sequence[str]) -> Sequence[Any]: ...

    def encode_document(self, images: Sequence[Any]) -> Sequence[Any]: ...


@dataclass(frozen=True)
class VisualPage:
    source_document_id: str
    document_version_id: str
    page: int
    image_uri: str
    title: str
    image: Any


def load_visual_encoder(model_name: str | None = None) -> VisualEncoder:
    try:
        from sentence_transformers import MultiVectorEncoder
    except ImportError as error:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Visual retrieval requires requirements-visual.txt and sentence-transformers MultiVectorEncoder."
        ) from error
    return MultiVectorEncoder(model_name or settings.visual_retrieval_model)


def _as_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    matrix = [[float(component) for component in row] for row in value]
    if not matrix or not matrix[0]:
        raise ValueError("visual encoder returned an empty multivector")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("visual encoder returned ragged multivectors")
    return matrix


def _mean_pool(matrix: list[list[float]]) -> list[float]:
    return [sum(row[index] for row in matrix) / len(matrix) for index in range(len(matrix[0]))]


def _filter(source_document_ids: Sequence[str] | None) -> models.Filter | None:
    ids = [str(value) for value in source_document_ids or [] if str(value)]
    if not ids:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="source_document_id", match=models.MatchAny(any=ids))]
    )


class VisualRetrievalSidecar:
    """Opt-in page-image retrieval, isolated from the production text index.

    A cheap pooled vector retrieves candidates; ColBERT-style page and query
    multivectors then rerank only that bounded candidate set with MaxSim.
    """

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        encoder: VisualEncoder | None = None,
    ) -> None:
        self.client = client or QdrantClient(url=settings.qdrant_url)
        self.encoder = encoder or load_visual_encoder()

    def ensure_collection(self, corpus_id: str, vector_size: int = VISUAL_VECTOR_SIZE) -> None:
        name = visual_collection_name(corpus_id)
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={
                "pooled": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                "late": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=0),
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )

    def upsert_pages(self, corpus_id: str, pages: Sequence[VisualPage]) -> int:
        if not pages:
            return 0
        vectors = self.encoder.encode_document([page.image for page in pages])
        if len(vectors) != len(pages):
            raise ValueError("visual encoder returned a mismatched page count")
        matrices = [_as_matrix(vector) for vector in vectors]
        vector_size = len(matrices[0][0])
        if any(len(row) != vector_size for matrix in matrices for row in matrix):
            raise ValueError("visual page embeddings use inconsistent dimensions")
        self.ensure_collection(corpus_id, vector_size)
        points = [
            models.PointStruct(
                id=visual_page_id(page.document_version_id, page.page),
                vector={"pooled": _mean_pool(matrix), "late": matrix},
                payload={
                    "source_document_id": page.source_document_id,
                    "document_version_id": page.document_version_id,
                    "page": page.page,
                    "image_uri": page.image_uri,
                    "title": page.title,
                    "is_active": True,
                },
            )
            for page, matrix in zip(pages, matrices, strict=True)
        ]
        self.client.upsert(collection_name=visual_collection_name(corpus_id), points=points, wait=True)
        return len(points)

    def search(
        self,
        corpus_id: str,
        query: str,
        *,
        source_document_ids: Sequence[str] | None = None,
        prefetch_limit: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("visual search query must not be empty")
        matrix = _as_matrix(self.encoder.encode_query([query])[0])
        qfilter = _filter(source_document_ids)
        result = self.client.query_points(
            collection_name=visual_collection_name(corpus_id),
            prefetch=models.Prefetch(
                query=_mean_pool(matrix),
                using="pooled",
                filter=qfilter,
                limit=prefetch_limit or settings.visual_retrieval_prefetch_limit,
            ),
            query=matrix,
            using="late",
            query_filter=qfilter,
            limit=limit or settings.visual_retrieval_result_limit,
            with_payload=True,
        )
        return [
            {
                "score": float(point.score),
                **dict(point.payload or {}),
                "visual_point_id": str(point.id),
            }
            for point in result.points
        ]
