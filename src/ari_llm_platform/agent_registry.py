"""Static generic registry validation, freshness assessment, and core-only routing for ARI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SUPPORTED_REGISTRY_VERSION = "4.0"
REQUIRED_FIELDS = {
    "specialist_id",
    "display_name",
    "status",
    "task_classes",
    "capabilities",
    "allowed_tool_categories",
    "risk_tier",
    "limitations",
    "validation_state",
}
OPTIONAL_FIELDS = {"validated_at", "notes"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
ALLOWED_STATUSES = {"proposed", "disabled", "enabled", "stale", "invalid", "retired"}
ALLOWED_RISK_TIERS = {"low", "moderate", "high"}
ALLOWED_VALIDATION_STATES = {"unassessed", "assessed", "stale", "invalid"}
SPECIALIST_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")

PROHIBITED_PATTERNS = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|secret|password|credential)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:[A-Za-z]:\\|/home/|/Users/|~[/\\])"),
    re.compile(r"\b(?:curl|wget|powershell|cmd\.exe|bash|sh)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:authorized to execute|safe for tool use|write active durable memory)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class RegistryError:
    path: str
    message: str


@dataclass(frozen=True)
class AgentRegistry:
    registry_version: str
    specialists: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RegistryValidationResult:
    registry: AgentRegistry | None
    errors: tuple[RegistryError, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def executable_count(self) -> int:
        return 0


@dataclass(frozen=True)
class RegistryFallback:
    reason: str
    use_core_path: bool = True


@dataclass(frozen=True)
class RegistryFreshnessResult:
    specialist_id: str | None
    classification: str
    fallback: RegistryFallback

    @property
    def is_fresh(self) -> bool:
        return self.classification == "fresh"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    specialist_id: str | None
    reason: str
    delegation_authorized: bool = False


def load_agent_registry(path: str | Path) -> RegistryValidationResult:
    """Load and validate one explicit static JSON registry file without side effects."""
    registry_path = Path(path)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _invalid("$", "registry file was not found")
    except OSError:
        return _invalid("$", "registry file could not be read")
    except json.JSONDecodeError:
        return _invalid("$", "registry file is not valid JSON")
    return validate_agent_registry(payload)


def validate_agent_registry(payload: Any) -> RegistryValidationResult:
    """Validate an in-memory registry payload without authorizing any action."""
    errors: list[RegistryError] = []
    if not isinstance(payload, dict):
        return _invalid("$", "registry must be an object")

    _reject_unknown_fields(payload, {"registry_version", "specialists"}, "$", errors)
    version = payload.get("registry_version")
    if version != SUPPORTED_REGISTRY_VERSION:
        errors.append(RegistryError("$.registry_version", "unsupported registry version"))

    specialists = payload.get("specialists")
    if not isinstance(specialists, list):
        errors.append(RegistryError("$.specialists", "specialists must be a list"))
        return RegistryValidationResult(None, tuple(errors))

    seen_ids: set[str] = set()
    validated_specialists: list[dict[str, Any]] = []
    for index, record in enumerate(specialists):
        record_path = f"$.specialists[{index}]"
        _validate_record(record, record_path, seen_ids, errors)
        if isinstance(record, dict):
            validated_specialists.append(record)

    if errors:
        return RegistryValidationResult(None, tuple(errors))
    return RegistryValidationResult(
        AgentRegistry(str(version), tuple(validated_specialists)),
        (),
    )


def assess_registry_freshness(
    validation_result: RegistryValidationResult,
    specialist_id: str,
    evaluated_at: datetime,
    freshness_window: timedelta,
) -> RegistryFreshnessResult:
    """Assess one validated registry record using only explicit deterministic inputs."""
    if not isinstance(evaluated_at, datetime) or evaluated_at.tzinfo is None:
        return _fallback_result(None, "invalid_evaluation_time")
    if not isinstance(freshness_window, timedelta) or freshness_window <= timedelta(0):
        return _fallback_result(None, "invalid_freshness_window")
    if not validation_result.is_valid or validation_result.registry is None:
        return _fallback_result(None, "contract_invalid")

    record = next(
        (
            item
            for item in validation_result.registry.specialists
            if item.get("specialist_id") == specialist_id
        ),
        None,
    )
    if record is None:
        return _fallback_result(None, "record_not_found")
    if record.get("status") != "enabled":
        return _fallback_result(specialist_id, "record_not_enabled")
    if record.get("validation_state") != "assessed":
        return _fallback_result(specialist_id, "record_unassessed")

    validated_at = _parse_iso8601_timestamp(record.get("validated_at"))
    if validated_at is None:
        return _fallback_result(specialist_id, "missing_or_invalid_timestamp")
    if validated_at > evaluated_at:
        return _fallback_result(specialist_id, "future_timestamp")
    if evaluated_at - validated_at > freshness_window:
        return _fallback_result(specialist_id, "stale")

    return RegistryFreshnessResult(
        specialist_id=specialist_id,
        classification="fresh",
        fallback=RegistryFallback(reason="no_execution_authority"),
    )


def decide_core_route(
    requested_task_class: str,
    freshness_result: RegistryFreshnessResult,
    record: dict[str, Any] | None = None,
) -> RouteDecision:
    """Return a pure core-path decision without authorizing delegation."""
    if not isinstance(requested_task_class, str) or not requested_task_class.strip():
        return _core_route(None, "invalid_request_metadata")
    if not isinstance(freshness_result, RegistryFreshnessResult):
        return _core_route(None, "indeterminate_input")
    if freshness_result.classification not in {"fresh", "fallback"}:
        return _core_route(freshness_result.specialist_id, "indeterminate_input")
    if freshness_result.classification == "fallback":
        return _core_route(
            freshness_result.specialist_id,
            f"registry_fallback:{freshness_result.fallback.reason}",
        )
    if not isinstance(record, dict):
        return _core_route(freshness_result.specialist_id, "indeterminate_input")

    task_classes = record.get("task_classes")
    if not isinstance(task_classes, list) or not all(
        isinstance(item, str) and item.strip() for item in task_classes
    ):
        return _core_route(freshness_result.specialist_id, "indeterminate_input")
    if requested_task_class.strip() not in task_classes:
        return _core_route(freshness_result.specialist_id, "task_class_not_matched")
    return _core_route(freshness_result.specialist_id, "delegation_not_implemented")


def _validate_record(
    record: Any,
    path: str,
    seen_ids: set[str],
    errors: list[RegistryError],
) -> None:
    if not isinstance(record, dict):
        errors.append(RegistryError(path, "specialist record must be an object"))
        return

    _reject_unknown_fields(record, ALLOWED_FIELDS, path, errors)
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(RegistryError(f"{path}.{field}", "required field is missing"))

    specialist_id = record.get("specialist_id")
    if not isinstance(specialist_id, str) or not SPECIALIST_ID_RE.fullmatch(specialist_id):
        errors.append(RegistryError(f"{path}.specialist_id", "invalid specialist identifier"))
    elif specialist_id in seen_ids:
        errors.append(RegistryError(f"{path}.specialist_id", "duplicate specialist identifier"))
    else:
        seen_ids.add(specialist_id)

    _validate_string(record.get("display_name"), f"{path}.display_name", errors)
    _validate_choice(record.get("status"), ALLOWED_STATUSES, f"{path}.status", errors)
    _validate_string_list(record.get("task_classes"), f"{path}.task_classes", errors, non_empty=True)
    _validate_string_list(record.get("capabilities"), f"{path}.capabilities", errors, non_empty=True)
    _validate_string_list(record.get("allowed_tool_categories"), f"{path}.allowed_tool_categories", errors)
    _validate_choice(record.get("risk_tier"), ALLOWED_RISK_TIERS, f"{path}.risk_tier", errors)
    _validate_string_list(record.get("limitations"), f"{path}.limitations", errors, non_empty=True)
    _validate_choice(record.get("validation_state"), ALLOWED_VALIDATION_STATES, f"{path}.validation_state", errors)

    if "validated_at" in record:
        _validate_timestamp(record["validated_at"], f"{path}.validated_at", errors)
    if "notes" in record:
        _validate_string(record["notes"], f"{path}.notes", errors)

    for field, value in record.items():
        _reject_prohibited(value, f"{path}.{field}", errors)


def _validate_string(value: Any, path: str, errors: list[RegistryError]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(RegistryError(path, "must be a non-empty string"))


def _validate_string_list(
    value: Any,
    path: str,
    errors: list[RegistryError],
    *,
    non_empty: bool = False,
) -> None:
    if not isinstance(value, list):
        errors.append(RegistryError(path, "must be a list"))
        return
    if non_empty and not value:
        errors.append(RegistryError(path, "must be a non-empty list"))
        return
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(RegistryError(path, "must contain only non-empty strings"))


def _validate_choice(
    value: Any,
    choices: set[str],
    path: str,
    errors: list[RegistryError],
) -> None:
    if value not in choices:
        errors.append(RegistryError(path, "contains an unsupported value"))


def _validate_timestamp(value: Any, path: str, errors: list[RegistryError]) -> None:
    if _parse_iso8601_timestamp(value) is None:
        errors.append(RegistryError(path, "must be an ISO 8601 timestamp"))


def _parse_iso8601_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None

    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return timestamp if timestamp.tzinfo is not None else None


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    path: str,
    errors: list[RegistryError],
) -> None:
    for field in value.keys() - allowed:
        errors.append(RegistryError(f"{path}.{field}", "unknown field"))


def _reject_prohibited(value: Any, path: str, errors: list[RegistryError]) -> None:
    if isinstance(value, str) and any(pattern.search(value) for pattern in PROHIBITED_PATTERNS):
        errors.append(RegistryError(path, "contains prohibited content"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_prohibited(item, f"{path}[{index}]", errors)


def _invalid(path: str, message: str) -> RegistryValidationResult:
    return RegistryValidationResult(None, (RegistryError(path, message),))


def _fallback_result(specialist_id: str | None, reason: str) -> RegistryFreshnessResult:
    return RegistryFreshnessResult(
        specialist_id=specialist_id,
        classification="fallback",
        fallback=RegistryFallback(reason=reason),
    )


def _core_route(specialist_id: str | None, reason: str) -> RouteDecision:
    return RouteDecision(
        route="core",
        specialist_id=specialist_id,
        reason=reason,
        delegation_authorized=False,
    )
