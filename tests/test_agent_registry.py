import json
from datetime import datetime, timedelta, timezone

from ari_llm_platform.agent_registry import (
    assess_registry_freshness,
    load_agent_registry,
    validate_agent_registry,
)


def valid_record():
    return {
        "specialist_id": "document-metadata-review",
        "display_name": "Document Metadata Review",
        "status": "proposed",
        "task_classes": ["document-management"],
        "capabilities": ["return generic metadata suggestions"],
        "allowed_tool_categories": [],
        "risk_tier": "low",
        "limitations": ["does not modify source material"],
        "validation_state": "unassessed",
    }


def valid_registry():
    return {"registry_version": "4.0", "specialists": [valid_record()]}


def test_valid_registry_has_no_executable_specialists(tmp_path):
    path = tmp_path / "agent-registry.json"
    path.write_text(json.dumps(valid_registry()), encoding="utf-8")

    result = load_agent_registry(path)

    assert result.is_valid
    assert result.registry is not None
    assert result.registry.specialists[0]["status"] == "proposed"
    assert result.executable_count == 0


def test_missing_required_field_is_invalid():
    payload = valid_registry()
    del payload["specialists"][0]["limitations"]

    result = validate_agent_registry(payload)

    assert not result.is_valid
    assert any(error.path.endswith(".limitations") for error in result.errors)


def test_unknown_record_field_is_invalid():
    payload = valid_registry()
    payload["specialists"][0]["routing_weight"] = 1

    result = validate_agent_registry(payload)

    assert not result.is_valid
    assert any(error.message == "unknown field" for error in result.errors)


def test_invalid_specialist_id_is_invalid():
    payload = valid_registry()
    payload["specialists"][0]["specialist_id"] = "Invalid Identifier"

    result = validate_agent_registry(payload)

    assert not result.is_valid
    assert any(
        error.message == "invalid specialist identifier"
        for error in result.errors
    )


def test_duplicate_specialist_id_is_invalid():
    payload = valid_registry()
    payload["specialists"].append(valid_record())

    result = validate_agent_registry(payload)

    assert not result.is_valid
    assert any(
        error.message == "duplicate specialist identifier"
        for error in result.errors
    )

def test_unsupported_contract_values_are_invalid():
    payload = valid_registry()
    payload["registry_version"] = "9.0"
    payload["specialists"][0]["status"] = "running"
    payload["specialists"][0]["risk_tier"] = "none"
    payload["specialists"][0]["validation_state"] = "passed"

    result = validate_agent_registry(payload)

    assert not result.is_valid
    assert sum(error.message == "contains an unsupported value" for error in result.errors) == 3
    assert any(error.path == "$.registry_version" for error in result.errors)


def test_invalid_timestamp_and_required_lists_are_invalid():
    payload = valid_registry()
    payload["specialists"][0]["validated_at"] = "not-a-timestamp"
    payload["specialists"][0]["task_classes"] = []
    payload["specialists"][0]["capabilities"] = [""]

    result = validate_agent_registry(payload)

    assert not result.is_valid
    assert any(error.path.endswith(".validated_at") for error in result.errors)
    assert any(error.path.endswith(".task_classes") for error in result.errors)
    assert any(error.path.endswith(".capabilities") for error in result.errors)


def test_prohibited_content_is_rejected():
    forbidden_values = [
        "https://example.invalid/endpoint",
        "api_key=private-value",
        "C:\\private\\vault.db",
        "powershell -Command Invoke-Thing",
        "authorized to execute external work",
    ]

    for forbidden in forbidden_values:
        payload = valid_registry()
        payload["specialists"][0]["notes"] = forbidden

        result = validate_agent_registry(payload)

        assert not result.is_valid
        assert any(error.message == "contains prohibited content" for error in result.errors)


def test_missing_or_malformed_file_is_invalid(tmp_path):
    missing_result = load_agent_registry(tmp_path / "missing.json")
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    malformed_result = load_agent_registry(malformed_path)

    assert not missing_result.is_valid
    assert missing_result.errors[0].message == "registry file was not found"
    assert not malformed_result.is_valid
    assert malformed_result.errors[0].message == "registry file is not valid JSON"


EVALUATED_AT = datetime(2026, 8, 25, 16, 0, tzinfo=timezone.utc)
WINDOW = timedelta(days=7)


def assessed_enabled_record(validated_at: str = "2026-08-20T16:00:00Z"):
    return {
        "specialist_id": "document-metadata-review",
        "display_name": "Document Metadata Review",
        "status": "enabled",
        "task_classes": ["document-management"],
        "capabilities": ["return generic metadata suggestions"],
        "allowed_tool_categories": [],
        "risk_tier": "low",
        "limitations": ["does not modify source material"],
        "validation_state": "assessed",
        "validated_at": validated_at,
    }


def assessed_enabled_registry(validated_at: str = "2026-08-20T16:00:00Z"):
    return {"registry_version": "4.0", "specialists": [assessed_enabled_record(validated_at)]}


def test_fresh_assessment_is_pure_and_does_not_authorize_execution():
    result = assess_registry_freshness(
        validate_agent_registry(assessed_enabled_registry()),
        "document-metadata-review",
        EVALUATED_AT,
        WINDOW,
    )

    assert result.is_fresh
    assert result.fallback.use_core_path
    assert result.fallback.reason == "no_execution_authority"


def test_expired_timestamp_returns_safe_fallback():
    result = assess_registry_freshness(
        validate_agent_registry(assessed_enabled_registry("2026-08-10T16:00:00Z")),
        "document-metadata-review",
        EVALUATED_AT,
        WINDOW,
    )

    assert not result.is_fresh
    assert result.classification == "fallback"
    assert result.fallback.reason == "stale"


def test_missing_or_future_timestamp_returns_safe_fallback():
    missing = assessed_enabled_registry()
    del missing["specialists"][0]["validated_at"]
    future = assessed_enabled_registry("2026-08-26T16:00:00Z")

    missing_result = assess_registry_freshness(
        validate_agent_registry(missing), "document-metadata-review", EVALUATED_AT, WINDOW
    )
    future_result = assess_registry_freshness(
        validate_agent_registry(future), "document-metadata-review", EVALUATED_AT, WINDOW
    )

    assert missing_result.fallback.reason == "missing_or_invalid_timestamp"
    assert future_result.fallback.reason == "future_timestamp"


def test_invalid_artifact_and_invalid_policy_input_return_safe_fallback():
    invalid_registry = validate_agent_registry({"registry_version": "9.0", "specialists": []})
    valid_registry = validate_agent_registry(assessed_enabled_registry())

    invalid_result = assess_registry_freshness(
        invalid_registry, "document-metadata-review", EVALUATED_AT, WINDOW
    )
    invalid_window = assess_registry_freshness(
        valid_registry, "document-metadata-review", EVALUATED_AT, timedelta(0)
    )

    assert invalid_result.fallback.reason == "contract_invalid"
    assert invalid_window.fallback.reason == "invalid_freshness_window"


def test_unavailable_or_unassessed_record_returns_safe_fallback():
    unavailable = assessed_enabled_registry()
    unavailable["specialists"][0]["status"] = "disabled"
    unassessed = assessed_enabled_registry()
    unassessed["specialists"][0]["validation_state"] = "unassessed"

    unavailable_result = assess_registry_freshness(
        validate_agent_registry(unavailable), "document-metadata-review", EVALUATED_AT, WINDOW
    )
    unassessed_result = assess_registry_freshness(
        validate_agent_registry(unassessed), "document-metadata-review", EVALUATED_AT, WINDOW
    )
    missing_result = assess_registry_freshness(
        validate_agent_registry(assessed_enabled_registry()), "unknown-specialist", EVALUATED_AT, WINDOW
    )

    assert unavailable_result.fallback.reason == "record_not_enabled"
    assert unassessed_result.fallback.reason == "record_unassessed"
    assert missing_result.fallback.reason == "record_not_found"

from ari_llm_platform.agent_registry import (
    RegistryFallback,
    RegistryFreshnessResult,
    decide_core_route,
)


def fresh_result(specialist_id: str = "document-metadata-review"):
    return RegistryFreshnessResult(
        specialist_id=specialist_id,
        classification="fresh",
        fallback=RegistryFallback(reason="no_execution_authority"),
    )


def fallback_result(reason: str = "record_not_enabled"):
    return RegistryFreshnessResult(
        specialist_id="document-metadata-review",
        classification="fallback",
        fallback=RegistryFallback(reason=reason),
    )


def generic_record(task_classes=None):
    return {"task_classes": task_classes or ["document-management"]}


def test_registry_fallback_always_chooses_core_path():
    decision = decide_core_route(
        "document-management",
        fallback_result("stale"),
        generic_record(),
    )

    assert decision.route == "core"
    assert decision.reason == "registry_fallback:stale"
    assert not decision.delegation_authorized


def test_unmatched_task_class_chooses_core_path():
    decision = decide_core_route(
        "code-diagnostics",
        fresh_result(),
        generic_record(),
    )

    assert decision.route == "core"
    assert decision.reason == "task_class_not_matched"
    assert not decision.delegation_authorized


def test_matching_fresh_record_still_does_not_delegate():
    decision = decide_core_route(
        "document-management",
        fresh_result(),
        generic_record(),
    )

    assert decision.route == "core"
    assert decision.reason == "delegation_not_implemented"
    assert decision.specialist_id == "document-metadata-review"
    assert not decision.delegation_authorized


def test_invalid_or_unknown_inputs_choose_core_path():
    invalid_task = decide_core_route("", fresh_result(), generic_record())
    invalid_freshness = decide_core_route("document-management", object(), generic_record())
    malformed_record = decide_core_route("document-management", fresh_result(), {"task_classes": [""]})
    unknown_classification = decide_core_route(
        "document-management",
        RegistryFreshnessResult(
            specialist_id="document-metadata-review",
            classification="unexpected",
            fallback=RegistryFallback(reason="unknown"),
        ),
        generic_record(),
    )

    assert invalid_task.reason == "invalid_request_metadata"
    assert invalid_freshness.reason == "indeterminate_input"
    assert malformed_record.reason == "indeterminate_input"
    assert unknown_classification.reason == "indeterminate_input"
    assert all(
        decision.route == "core" and not decision.delegation_authorized
        for decision in (invalid_task, invalid_freshness, malformed_record, unknown_classification)
    )