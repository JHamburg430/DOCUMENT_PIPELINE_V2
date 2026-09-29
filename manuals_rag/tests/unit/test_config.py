from pathlib import Path

from manuals_rag_common.config import Settings, _as_bool


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_agentic_retrieval_is_disabled_by_default() -> None:
    field = Settings.__dataclass_fields__["agentic_retrieval_enabled"]

    assert field.default is False


def test_agentic_retrieval_can_be_explicitly_enabled() -> None:
    assert _as_bool("true", False) is True


def test_example_environment_uses_the_production_embedding_model() -> None:
    configured_default = Settings.__dataclass_fields__["ollama_embed_model"].default
    example_environment = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    assert f"OLLAMA_EMBED_MODEL={configured_default}" in example_environment
    assert "nomic-embed-text" not in example_environment
