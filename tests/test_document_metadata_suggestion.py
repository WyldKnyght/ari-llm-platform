from ari_llm_platform.specialists.document_metadata_suggestion import (
    MAX_INPUT_CHARACTERS,
    suggest_document_metadata,
)


def test_default_off_returns_core_path_fallback():
    result = suggest_document_metadata("Neutral document metadata example")

    assert result.outcome == "not_enabled"
    assert result.use_core_path
    assert result.title_candidate is None
    assert result.keyword_candidates == ()


def test_enabled_neutral_input_returns_deterministic_non_authoritative_output():
    text = "Local documentation policy covers metadata review and retention."

    first = suggest_document_metadata(text, enabled=True)
    second = suggest_document_metadata(text, enabled=True)

    assert first == second
    assert first.outcome == "completed"
    assert first.use_core_path
    assert first.title_candidate == "Local Documentation Policy Covers Metadata Review And Retention"
    assert first.token_count == 8
    assert first.keyword_candidates == (
        "local",
        "documentation",
        "policy",
        "covers",
        "metadata",
    )


def test_invalid_and_oversized_inputs_return_core_path_fallback():
    empty = suggest_document_metadata("", enabled=True)
    non_string = suggest_document_metadata(None, enabled=True)
    oversized = suggest_document_metadata("x" * (MAX_INPUT_CHARACTERS + 1), enabled=True)

    assert empty.outcome == "invalid_input"
    assert non_string.outcome == "invalid_input"
    assert oversized.outcome == "input_too_large"
    assert all(result.use_core_path for result in (empty, non_string, oversized))


def test_private_or_runtime_like_input_is_rejected():
    values = (
        "https://example.invalid/private",
        "api_key=not-for-registry",
        "C:\\Private\\vault.db",
    )

    for value in values:
        result = suggest_document_metadata(value, enabled=True)

        assert result.outcome == "invalid_input"
        assert result.use_core_path
        assert result.title_candidate is None


def test_output_is_bounded_and_contains_no_input_echo_field():
    result = suggest_document_metadata(
        "One two three four five six seven eight nine ten eleven.",
        enabled=True,
    )

    assert result.outcome == "completed"
    assert len(result.keyword_candidates) <= 5
    assert result.token_count == 11
    assert not hasattr(result, "text")
    assert not hasattr(result, "raw_input")
