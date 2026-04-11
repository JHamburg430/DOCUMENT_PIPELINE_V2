from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - dependency fallback for existing worker images
    class BM25Okapi:  # type: ignore[no-redef]
        def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
            import math

            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.avgdl = sum(len(document) for document in corpus) / max(len(corpus), 1)
            self.doc_freqs: list[dict[str, int]] = []
            document_counts: dict[str, int] = {}
            for document in corpus:
                frequencies: dict[str, int] = {}
                for token in document:
                    frequencies[token] = frequencies.get(token, 0) + 1
                self.doc_freqs.append(frequencies)
                for token in frequencies:
                    document_counts[token] = document_counts.get(token, 0) + 1
            corpus_size = len(corpus)
            self.idf = {
                token: math.log(1 + (corpus_size - count + 0.5) / (count + 0.5))
                for token, count in document_counts.items()
            }

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            scores: list[float] = []
            for document, frequencies in zip(self.corpus, self.doc_freqs, strict=True):
                doc_len = len(document) or 1
                score = 0.0
                for token in query_tokens:
                    frequency = frequencies.get(token, 0)
                    if not frequency:
                        continue
                    denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                    score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
                scores.append(score)
            return scores

from manuals_rag_common.config import settings
from manuals_rag_schemas.documents import RetrievalChunk, SearchResult
from manuals_rag_retrieval.embeddings import build_sparse_vector, embed_dense, tokenize


COLLECTION_PREFIX = "manuals_"
logger = logging.getLogger(__name__)


def collection_name(corpus_id: str) -> str:
    return f"{COLLECTION_PREFIX}{corpus_id}"


class QdrantStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)

    def ensure_collection(self, corpus_id: str, vector_size: int) -> None:
        name = collection_name(corpus_id)
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    def upsert_chunks(self, corpus_id: str, chunks: list[RetrievalChunk]) -> None:
        if not chunks:
            return
        dense_vectors = embed_dense([chunk.content_for_dense for chunk in chunks])
        self.ensure_collection(corpus_id, len(dense_vectors[0]))
        points: list[PointStruct] = []
        for chunk, dense in zip(chunks, dense_vectors, strict=True):
            indices, values = build_sparse_vector(chunk.content_for_sparse)
            points.append(
                PointStruct(
                    id=chunk.id,
                    vector={"dense": dense, "sparse": SparseVector(indices=indices, values=values)},
                    payload={
                        **chunk.metadata_json,
                        "chunk_id": chunk.id,
                        "document_version_id": chunk.document_version_id,
                        "source_document_id": chunk.source_document_id,
                        "chunk_type": chunk.chunk_type.value,
                        "chunk_level": chunk.chunk_level,
                        "title": chunk.title,
                        "page_from": chunk.page_from,
                        "page_to": chunk.page_to,
                        "section_path": chunk.metadata_json.get("section_path", []),
                        "content": chunk.content,
                        "content_for_rerank": chunk.content_for_rerank,
                        "priority_score": chunk.priority_score,
                        "is_active": chunk.is_active,
                    },
                )
            )
        for index in range(0, len(points), 64):
            self.client.upsert(collection_name(corpus_id), points[index : index + 64])

    def delete_document_chunks(
        self,
        corpus_id: str,
        *,
        source_document_id: str | None = None,
        document_version_id: str | None = None,
        chunk_ids: list[str] | None = None,
    ) -> None:
        if not self.client.collection_exists(collection_name(corpus_id)):
            return
        must: list[Any] = []
        if chunk_ids:
            must.append(FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids)))
        if source_document_id:
            must.append(FieldCondition(key="source_document_id", match=MatchValue(value=str(source_document_id))))
        if document_version_id:
            must.append(FieldCondition(key="document_version_id", match=MatchValue(value=str(document_version_id))))
        if not must:
            return
        self.client.delete(
            collection_name=collection_name(corpus_id),
            points_selector=FilterSelector(filter=Filter(must=must)),
            wait=True,
        )

    def _build_filter(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        must = []
        for key, value in filters.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                must.append({"key": key, "match": {"any": value}})
            else:
                must.append({"key": key, "match": {"value": value}})
        return {"must": must} if must else None

    def search_dense(self, corpus_id: str, query: str, filters: dict[str, Any], limit: int = 40) -> list[SearchResult]:
        dense = embed_dense([query])[0]
        name = collection_name(corpus_id)
        query_filter = self._build_filter(filters)
        try:
            dense_hits = self.client.search(
                collection_name=name,
                query_vector=NamedVector(name="dense", vector=dense),
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            if "Vector dimension error" in str(exc):
                try:
                    collection = self.client.get_collection(name)
                    configured = collection.config.params.vectors
                    expected_dim = configured.get("dense").size if isinstance(configured, dict) and configured.get("dense") else None
                except Exception:
                    expected_dim = None
                logger.warning(
                    "Dense search disabled for corpus_id=%s due to embedding dimension mismatch: query_dim=%s expected_dim=%s error=%s",
                    corpus_id,
                    len(dense),
                    expected_dim,
                    exc,
                )
                return []
            raise
        return self._hits_to_results(dense_hits)

    def search_sparse(self, corpus_id: str, query: str, filters: dict[str, Any], limit: int = 40) -> list[SearchResult]:
        points = self._scroll_points(corpus_id=corpus_id, filters=filters)
        if not points:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        corpus_tokens: list[list[str]] = []
        point_payloads: list[tuple[str, dict[str, Any]]] = []
        for point in points:
            payload = point.payload or {}
            text = self._bm25_document_text(payload)
            tokens = tokenize(text)
            if not tokens:
                continue
            point_payloads.append((str(point.id), payload))
            corpus_tokens.append(tokens)
        if not corpus_tokens:
            return []
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        ranked: list[tuple[str, dict[str, Any], float]] = []
        for (chunk_id, payload), score in zip(point_payloads, scores, strict=True):
            lexical_bonus = self._lexical_bonus(payload, set(query_tokens))
            combined_score = float(score) + lexical_bonus
            if combined_score <= 0:
                continue
            ranked.append((chunk_id, payload, combined_score))
        ranked.sort(key=lambda item: item[2], reverse=True)
        return [
            self._payload_to_result(chunk_id=chunk_id, payload=payload, score=score)
            for chunk_id, payload, score in ranked[:limit]
        ]

    def search(self, corpus_id: str, query: str, filters: dict[str, Any], limit: int = 12) -> list[SearchResult]:
        dense_results = self.search_dense(corpus_id=corpus_id, query=query, filters=filters, limit=limit * 4)
        sparse_results = self.search_sparse(corpus_id=corpus_id, query=query, filters=filters, limit=limit * 4)
        fused = self.fuse_rrf([dense_results, sparse_results], limit=limit)
        if not fused:
            points, _ = self.client.scroll(
                collection_name=collection_name(corpus_id),
                limit=256,
                with_payload=True,
            )
            query_terms = set(tokenize(query))
            records: dict[str, Any] = {}
            fallback_ranked: list[tuple[str, float]] = []
            for point in points:
                payload = point.payload or {}
                if not self._payload_matches(payload, filters):
                    continue
                lexical_score = self._lexical_bonus(payload, query_terms)
                if lexical_score <= 0:
                    continue
                fallback_ranked.append((str(point.id), lexical_score))
                records[str(point.id)] = payload
            return [
                SearchResult(
                    chunk_id=chunk_id,
                    score=score + float(records[chunk_id].get("priority_score", 0.0)) / 100.0,
                    title=records[chunk_id]["title"],
                    document_version_id=records[chunk_id]["document_version_id"],
                    source_document_id=records[chunk_id]["source_document_id"],
                    pages=list(range(records[chunk_id]["page_from"], records[chunk_id]["page_to"] + 1)),
                    section_path=records[chunk_id].get("section_path", []),
                    content=records[chunk_id]["content"],
                    metadata=records[chunk_id],
                )
                for chunk_id, score in sorted(fallback_ranked, key=lambda item: item[1], reverse=True)[:limit]
            ]
        return fused

    def _scroll_points(self, corpus_id: str, filters: dict[str, Any], batch_size: int = 256) -> list[Any]:
        name = collection_name(corpus_id)
        query_filter = self._build_filter(filters)
        offset = None
        points: list[Any] = []
        while True:
            batch, offset = self.client.scroll(
                collection_name=name,
                scroll_filter=query_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(batch)
            if offset is None or not batch:
                break
        return points

    def _hits_to_results(self, hits: list[Any]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(self._payload_to_result(chunk_id=str(hit.id), payload=payload, score=float(hit.score)))
        return results

    def _payload_to_result(self, *, chunk_id: str, payload: dict[str, Any], score: float) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            score=score + float(payload.get("priority_score", 0.0)) / 100.0,
            title=payload["title"],
            document_version_id=payload["document_version_id"],
            source_document_id=payload["source_document_id"],
            pages=list(range(payload["page_from"], payload["page_to"] + 1)),
            section_path=payload.get("section_path", []),
            content=payload["content"],
            metadata=payload,
        )

    @staticmethod
    def fuse_rrf(result_sets: list[list[SearchResult]], *, limit: int, k: int = 60) -> list[SearchResult]:
        fused_scores: dict[str, float] = {}
        best_result: dict[str, SearchResult] = {}
        for result_set in result_sets:
            for rank, result in enumerate(result_set, start=1):
                fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
                if result.chunk_id not in best_result or result.score > best_result[result.chunk_id].score:
                    best_result[result.chunk_id] = result
        ranked_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[:limit]
        return [
            best_result[chunk_id].model_copy(update={"score": fused_scores[chunk_id]})
            for chunk_id in ranked_ids
        ]

    def _lexical_bonus(self, payload: dict[str, Any], query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        content_terms = set(tokenize(str(payload.get("content", ""))))
        title_terms = set(tokenize(str(payload.get("title", ""))))
        section_terms = set(tokenize(" ".join(payload.get("section_path", []))))
        metadata_terms = set(
            tokenize(
                " ".join(
                    self._metadata_value_text(payload.get(key, ""))
                    for key in (
                        "product_model",
                        "product_models",
                        "manufacturer",
                        "document_kind",
                        "product_family",
                        "product_families",
                        "devices",
                        "part_numbers",
                        "settings",
                        "parameters",
                        "document_topics",
                        "chunk_type",
                    )
                )
            )
        )
        content_overlap = len(query_terms.intersection(content_terms))
        title_overlap = len(query_terms.intersection(title_terms))
        section_overlap = len(query_terms.intersection(section_terms))
        metadata_overlap = len(query_terms.intersection(metadata_terms))
        chunk_type = str(payload.get("chunk_type", ""))
        structured_bonus = 0.03 if chunk_type in {"warning_record", "procedure_record", "spec_record", "datasheet_record", "table_record"} else 0.0
        return (
            content_overlap * 0.035
            + title_overlap * 0.03
            + section_overlap * 0.025
            + metadata_overlap * 0.04
            + structured_bonus
        )

    def _bm25_document_text(self, payload: dict[str, Any]) -> str:
        title = str(payload.get("title", "")).strip()
        section_path = " ".join(str(part) for part in payload.get("section_path", []))
        content = str(payload.get("content", "")).strip()
        metadata_bits = [
            self._metadata_value_text(payload.get("product_model", "")).strip(),
            self._metadata_value_text(payload.get("product_models", "")).strip(),
            self._metadata_value_text(payload.get("product_family", "")).strip(),
            self._metadata_value_text(payload.get("product_families", "")).strip(),
            self._metadata_value_text(payload.get("devices", "")).strip(),
            self._metadata_value_text(payload.get("part_numbers", "")).strip(),
            self._metadata_value_text(payload.get("settings", "")).strip(),
            self._metadata_value_text(payload.get("parameters", "")).strip(),
            str(payload.get("manufacturer", "")).strip(),
            str(payload.get("document_kind", "")).strip(),
            self._metadata_value_text(payload.get("document_topics", "")).strip(),
            self._metadata_value_text(payload.get("document_protocol_terms", "")).strip(),
            str(payload.get("chunk_type", "")).strip(),
            " ".join(str(item) for item in payload.get("identifier_tokens", [])),
            " ".join(str(item) for item in payload.get("menu_labels", [])),
            " ".join(str(item) for item in payload.get("protocol_terms", [])),
        ]
        weighted_parts = [
            title,
            title,
            section_path,
            content,
            content,
            " ".join(bit for bit in metadata_bits if bit),
        ]
        return "\n".join(part for part in weighted_parts if part)

    def _payload_matches(self, payload: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if value in (None, "", [], {}):
                continue
            current = payload.get(key)
            if key == "product_model" and value in (payload.get("product_models") or []):
                continue
            if key == "product_family" and value in (payload.get("product_families") or []):
                continue
            if isinstance(value, list):
                if isinstance(current, list):
                    if not any(item in value for item in current):
                        return False
                elif current not in value:
                    return False
            elif isinstance(current, list):
                if value not in current:
                    return False
            elif current != value:
                return False
        return True

    @staticmethod
    def _metadata_value_text(value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value)
