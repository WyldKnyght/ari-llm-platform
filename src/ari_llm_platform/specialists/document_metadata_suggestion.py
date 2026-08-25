"""Bounded deterministic P4.5 Document Metadata Suggestion pilot."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INPUT_CHARACTERS = 4_000
MAX_KEYWORDS = 5
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_PROHIBITED_RE = re.compile(
    r"https?://|\b(?:api[_ -]?key|access[_ -]?token|secret|password|credential)\b|"
    r"\b(?:[A-Za-z]:\\|/home/|/Users/|~[/\\])",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "with",
})


@dataclass(frozen=True)
class MetadataSuggestion:
    outcome: str
    use_core_path: bool
    title_candidate: str | None = None
    token_count: int = 0
    keyword_candidates: tuple[str, ...] = ()
    reason: str | None = None


def suggest_document_metadata(text: object, *, enabled: bool = False) -> MetadataSuggestion:
    """Return a pure, local, non-authoritative metadata suggestion result."""
    if not enabled:
        return _fallback("not_enabled")
    if not isinstance(text, str) or not text.strip():
        return _fallback("invalid_input")
    if len(text) > MAX_INPUT_CHARACTERS:
        return _fallback("input_too_large")
    if _PROHIBITED_RE.search(text):
        return _fallback("invalid_input")

    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return _fallback("invalid_input")

    keywords = _keyword_candidates(tokens)
    return MetadataSuggestion(
        outcome="completed",
        use_core_path=True,
        title_candidate=_title_candidate(tokens),
        token_count=len(tokens),
        keyword_candidates=keywords,
    )


def _title_candidate(tokens: list[str]) -> str:
    return " ".join(tokens[:8]).title()


def _keyword_candidates(tokens: list[str]) -> tuple[str, ...]:
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = token.lower()
        if normalized in _STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(normalized)
        if len(keywords) == MAX_KEYWORDS:
            break
    return tuple(keywords)


def _fallback(reason: str) -> MetadataSuggestion:
    return MetadataSuggestion(outcome=reason, use_core_path=True, reason=reason)
