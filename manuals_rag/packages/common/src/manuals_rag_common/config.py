from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8600"))
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN",
        "postgresql://manuals:manuals@postgres:5432/manuals_rag",
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    minio_public_endpoint: str = os.getenv("MINIO_PUBLIC_ENDPOINT", os.getenv("MINIO_ENDPOINT", "minio:9000"))
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    minio_region: str = os.getenv("MINIO_REGION", "us-east-1")
    minio_secure: bool = _as_bool(os.getenv("MINIO_SECURE"), False)
    minio_public_secure: bool = _as_bool(os.getenv("MINIO_PUBLIC_SECURE"), minio_secure)
    minio_bucket_originals: str = os.getenv("MINIO_BUCKET_ORIGINALS", "manuals-originals")
    minio_bucket_artifacts: str = os.getenv("MINIO_BUCKET_ARTIFACTS", "manuals-artifacts")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    ollama_metadata_model: str = os.getenv("OLLAMA_METADATA_MODEL", "tinyllama:1.1b")
    ollama_fast_model: str = os.getenv("OLLAMA_FAST_MODEL", "qwen3.5:4b")
    ollama_answer_model: str = os.getenv("OLLAMA_ANSWER_MODEL", "qwen3.5:9b")
    ollama_eval_model: str = os.getenv("OLLAMA_EVAL_MODEL", os.getenv("OLLAMA_ANSWER_MODEL", "qwen3.5:9b"))
    ollama_eval_question_model: str = os.getenv("OLLAMA_EVAL_QUESTION_MODEL", "qwen3.5:27b")
    ollama_eval_question_num_ctx: int = int(os.getenv("OLLAMA_EVAL_QUESTION_NUM_CTX", "32768"))
    ollama_eval_question_timeout_seconds: float = float(os.getenv("OLLAMA_EVAL_QUESTION_TIMEOUT_SECONDS", "180"))
    auth_mode: str = os.getenv("AUTH_MODE", "local")
    local_admin_token: str = os.getenv("LOCAL_ADMIN_TOKEN", "admin-token")
    local_operator_token: str = os.getenv("LOCAL_OPERATOR_TOKEN", "operator-token")
    local_end_user_token: str = os.getenv("LOCAL_END_USER_TOKEN", "user-token")
    local_auditor_token: str = os.getenv("LOCAL_AUDITOR_TOKEN", "auditor-token")
    default_tenant_id: str = os.getenv("DEFAULT_TENANT_ID", "local-tenant")
    default_corpus_id: str = os.getenv("DEFAULT_CORPUS_ID", "manuals_vendor_keyence")
    haystack_rerank_model: str = os.getenv(
        "HAYSTACK_RERANK_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
    )
    haystack_rerank_device: str = os.getenv("HAYSTACK_RERANK_DEVICE", "auto")
    haystack_rerank_top_k: int = int(os.getenv("HAYSTACK_RERANK_TOP_K", "12"))


settings = Settings()
