from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any

import httpx

from manuals_rag_common.config import settings


MAX_EMBED_CHARS = 6000
EMBED_RETRY_STATUS_CODES = {500, 502, 503, 504}
EMBED_RETRY_LIMIT = 2
EMBED_BATCH_SIZE = 32


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-./]+", text.lower())


def expand_search_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    tokens = tokenize(compact)
    expansions: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        variants = {
            token,
            token.replace("-", ""),
            token.replace("/", " "),
            token.replace(".", " "),
            token.replace("-", " "),
        }
        for variant in variants:
            cleaned = re.sub(r"\s+", " ", variant).strip()
            if len(cleaned) < 2 or cleaned in seen:
                continue
            seen.add(cleaned)
            expansions.append(cleaned)
    if expansions:
        compact = f"{compact}\n\nSearch terms: {' | '.join(expansions)}"
    return compact


def build_sparse_vector(text: str, dim: int = 50000) -> tuple[list[int], list[float]]:
    counts: dict[int, float] = {}
    for token in tokenize(expand_search_text(text)):
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
        counts[token_hash] = counts.get(token_hash, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    indices = sorted(counts)
    values = [counts[index] / norm for index in indices]
    return indices, values


def normalize_for_embedding(text: str, max_chars: int = MAX_EMBED_CHARS) -> str:
    compact = expand_search_text(text)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) <= max_chars:
        return compact
    clipped = compact[:max_chars]
    last_space = clipped.rfind(" ")
    if last_space > int(max_chars * 0.7):
        clipped = clipped[:last_space]
    return clipped.strip()


def _embed_batch(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    response = client.post(
        "/api/embed",
        json={"model": settings.ollama_embed_model, "input": texts},
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings") or []
    if embeddings and isinstance(embeddings[0], list):
        return embeddings
    embedding = payload.get("embedding")
    if isinstance(embedding, list) and embedding and isinstance(embedding[0], (int, float)):
        return [embedding]
    raise ValueError("Unexpected embedding response shape.")


def _embed_one_with_fallbacks(client: httpx.Client, text: str) -> list[float]:
    payload: dict[str, Any] | None = None
    normalized = normalize_for_embedding(text)
    for limit in (MAX_EMBED_CHARS, 4000, 2500, 1200, 600):
        candidate = normalize_for_embedding(normalized, max_chars=limit)
        for attempt in range(EMBED_RETRY_LIMIT + 1):
            response = client.post(
                "/api/embed",
                json={"model": settings.ollama_embed_model, "input": candidate or "content unavailable"},
            )
            if response.status_code == 400:
                break
            if response.status_code in EMBED_RETRY_STATUS_CODES and attempt < EMBED_RETRY_LIMIT:
                time.sleep(0.5 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            break
        if payload is not None:
            break
    if payload is None:
        for attempt in range(EMBED_RETRY_LIMIT + 1):
            response = client.post(
                "/api/embed",
                json={"model": settings.ollama_embed_model, "input": "content unavailable"},
            )
            if response.status_code in EMBED_RETRY_STATUS_CODES and attempt < EMBED_RETRY_LIMIT:
                time.sleep(0.5 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            break
    embeddings = payload.get("embeddings") or []
    if embeddings and isinstance(embeddings[0], list):
        return embeddings[0]
    return payload["embedding"]


def embed_dense(texts: list[str]) -> list[list[float]]:
    with httpx.Client(base_url=settings.ollama_url, timeout=60.0) as client:
        vectors: list[list[float]] = []
        normalized_texts = [
            normalize_for_embedding(text) or "content unavailable"
            for text in texts
        ]
        for index in range(0, len(normalized_texts), EMBED_BATCH_SIZE):
            batch = normalized_texts[index : index + EMBED_BATCH_SIZE]
            try:
                vectors.extend(_embed_batch(client, batch))
            except Exception:
                for text in batch:
                    vectors.append(_embed_one_with_fallbacks(client, text))
        return vectors
