from manuals_rag_common.config import Settings, _as_bool


def test_agentic_retrieval_is_disabled_by_default() -> None:
    field = Settings.__dataclass_fields__["agentic_retrieval_enabled"]

    assert field.default is False


def test_agentic_retrieval_can_be_explicitly_enabled() -> None:
    assert _as_bool("true", False) is True
