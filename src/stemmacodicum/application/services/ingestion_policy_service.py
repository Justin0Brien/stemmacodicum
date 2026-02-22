from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.infrastructure.parsers.docling_adapter import DoclingAdapter


@dataclass(slots=True)
class IngestionPolicyDecision:
    resource_kind: str
    should_extract_auto: bool
    should_vector_auto: bool
    reason: str


class IngestionPolicyService:
    STRUCTURED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".ods"}
    STRUCTURED_MEDIA_TYPES = {
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.oasis.opendocument.spreadsheet",
    }

    def __init__(
        self,
        *,
        structured_auto_extract: bool | None = None,
        structured_auto_vector: bool | None = None,
        narrative_auto_vector: bool | None = None,
    ) -> None:
        self.structured_auto_extract = (
            self._read_bool_env("STEMMA_STRUCTURED_AUTO_EXTRACT", False)
            if structured_auto_extract is None
            else bool(structured_auto_extract)
        )
        self.structured_auto_vector = (
            self._read_bool_env("STEMMA_STRUCTURED_AUTO_VECTOR", False)
            if structured_auto_vector is None
            else bool(structured_auto_vector)
        )
        self.narrative_auto_vector = (
            self._read_bool_env("STEMMA_NARRATIVE_AUTO_VECTOR", True)
            if narrative_auto_vector is None
            else bool(narrative_auto_vector)
        )

    def decide(self, *, media_type: str | None, original_filename: str | None) -> IngestionPolicyDecision:
        suffix = Path(original_filename or "").suffix.lower()
        normalized_media = (media_type or "").strip().lower()

        if suffix in self.STRUCTURED_EXTENSIONS or normalized_media in self.STRUCTURED_MEDIA_TYPES:
            return IngestionPolicyDecision(
                resource_kind="structured_data",
                should_extract_auto=self.structured_auto_extract,
                should_vector_auto=self.structured_auto_vector,
                reason="structured_data_default_policy",
            )

        if DoclingAdapter.supports(media_type, original_filename):
            return IngestionPolicyDecision(
                resource_kind="narrative_document",
                should_extract_auto=True,
                should_vector_auto=self.narrative_auto_vector,
                reason="docling_extractable_narrative",
            )

        return IngestionPolicyDecision(
            resource_kind="binary_blob",
            should_extract_auto=False,
            should_vector_auto=False,
            reason="non_extractable_media_type",
        )

    @staticmethod
    def _read_bool_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return default
