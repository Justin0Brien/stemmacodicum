from __future__ import annotations

import json
from dataclasses import dataclass

from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.structured_data_service import StructuredDataService
from stemmacodicum.core.errors import VerificationError
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.claim import Claim
from stemmacodicum.domain.models.verification import VerificationResult, VerificationRun
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo


@dataclass(slots=True)
class VerifyClaimOutcome:
    run_id: str
    claim_id: str
    status: str
    diagnostics: dict[str, object]


@dataclass(slots=True)
class VerifySetOutcome:
    run_id: str
    total: int
    passed: int
    failed: int


class VerificationService:
    def __init__(
        self,
        claim_repo: ClaimRepo,
        evidence_repo: EvidenceRepo,
        extraction_repo: ExtractionRepo,
        verification_repo: VerificationRepo,
        binding_service: EvidenceBindingService,
        structured_data_service: StructuredDataService | None = None,
    ) -> None:
        self.claim_repo = claim_repo
        self.evidence_repo = evidence_repo
        self.extraction_repo = extraction_repo
        self.verification_repo = verification_repo
        self.binding_service = binding_service
        self.structured_data_service = structured_data_service

    def verify_claim(self, claim_id: str, policy_profile: str = "strict") -> VerifyClaimOutcome:
        claim = self.claim_repo.get_claim_by_id(claim_id)
        if claim is None:
            raise VerificationError(f"Claim not found: {claim_id}")

        run = VerificationRun(
            id=new_uuid(),
            claim_set_id=claim.claim_set_id,
            policy_profile=policy_profile,
            created_at=now_utc_iso(),
        )
        self.verification_repo.insert_run(run)

        status, diagnostics = self._verify_claim_logic(claim)
        result = VerificationResult(
            id=new_uuid(),
            run_id=run.id,
            claim_id=claim.id,
            status=status,
            diagnostics_json=json.dumps(diagnostics, ensure_ascii=True, sort_keys=True),
            created_at=now_utc_iso(),
        )
        self.verification_repo.insert_result(result)

        return VerifyClaimOutcome(run_id=run.id, claim_id=claim.id, status=status, diagnostics=diagnostics)

    def verify_claim_set(self, claim_set_name: str, policy_profile: str = "strict") -> VerifySetOutcome:
        claim_set = self.claim_repo.get_claim_set_by_name(claim_set_name)
        if claim_set is None:
            raise VerificationError(f"Claim set not found: {claim_set_name}")

        claims = self.claim_repo.list_claims(claim_set_id=claim_set.id, limit=1_000_000)

        run = VerificationRun(
            id=new_uuid(),
            claim_set_id=claim_set.id,
            policy_profile=policy_profile,
            created_at=now_utc_iso(),
        )
        self.verification_repo.insert_run(run)

        passed = 0
        failed = 0

        for claim in claims:
            status, diagnostics = self._verify_claim_logic(claim)
            if status == "pass":
                passed += 1
            else:
                failed += 1

            result = VerificationResult(
                id=new_uuid(),
                run_id=run.id,
                claim_id=claim.id,
                status=status,
                diagnostics_json=json.dumps(diagnostics, ensure_ascii=True, sort_keys=True),
                created_at=now_utc_iso(),
            )
            self.verification_repo.insert_result(result)

        return VerifySetOutcome(run_id=run.id, total=len(claims), passed=passed, failed=failed)

    def _verify_claim_logic(self, claim: Claim) -> tuple[str, dict[str, object]]:
        binding = self.binding_service.validate_binding(claim.id)
        if not binding.ok:
            return (
                "fail",
                {
                    "reason": "binding_validation_failed",
                    "missing_roles": binding.missing_roles,
                    "weak_evidence_ids": binding.evidence_with_too_few_selectors,
                },
            )

        bundle = self.evidence_repo.list_evidence_for_claim(claim.id)
        if claim.claim_type == "narrative":
            return self._verify_narrative_claim(claim, bundle)

        if claim.claim_type == "quantitative":
            return self._verify_quantitative_claim(claim, bundle)

        return "fail", {"reason": f"unsupported_claim_type:{claim.claim_type}"}

    def _verify_narrative_claim(self, claim: Claim, bundle) -> tuple[str, dict[str, object]]:
        for item, selectors in bundle:
            if item.role != "quote":
                continue
            for selector in selectors:
                if selector.selector_type == "TextQuoteSelector":
                    payload = json.loads(selector.selector_json)
                    exact = str(payload.get("exact") or "").strip()
                    if not exact:
                        continue

                    source_text = self.extraction_repo.get_latest_document_text_for_resource(item.resource_id)
                    if source_text and exact.lower() in source_text.text_content.lower():
                        return (
                            "pass",
                            {
                                "reason": "narrative_quote_match_source",
                                "quote": exact,
                                "resource_id": item.resource_id,
                            },
                        )

                    if claim.narrative_text and exact.lower() in claim.narrative_text.lower():
                        return "pass", {"reason": "narrative_quote_match_claim_text", "quote": exact}

        return "fail", {"reason": "no_matching_text_quote_selector"}

    def _verify_quantitative_claim(self, claim: Claim, bundle) -> tuple[str, dict[str, object]]:
        value_items = [(item, selectors) for item, selectors in bundle if item.role == "value-cell"]
        if not value_items:
            return "fail", {"reason": "missing_value_cell"}

        value_item, value_selectors = value_items[0]

        table_selector_payload = None
        for selector in value_selectors:
            if selector.selector_type == "TableAddressSelector":
                table_selector_payload = json.loads(selector.selector_json)
                break

        if table_selector_payload is None:
            return "fail", {"reason": "missing_table_address_selector"}

        table_id = str(table_selector_payload.get("table_id") or "").strip()
        if not table_id:
            return "fail", {"reason": "table_address_missing_table_id"}

        extracted = self.extraction_repo.get_table_by_table_id(value_item.resource_id, table_id)
        if extracted is None:
            return "fail", {"reason": "table_not_found", "table_id": table_id}

        cell_ref = table_selector_payload.get("cell_ref") or {}
        row_index = cell_ref.get("row_index")
        col_index = cell_ref.get("col_index")

        rows = json.loads(extracted.row_headers_json)
        cols = json.loads(extracted.col_headers_json)
        cells = json.loads(extracted.cells_json)

        if row_index is None:
            row_index = self._resolve_row_index(rows, table_selector_payload.get("row_path"))
        if col_index is None:
            col_index = self._resolve_col_index(cols, table_selector_payload.get("col_path"))

        if row_index is None or col_index is None:
            return (
                "fail",
                {
                    "reason": "unable_to_resolve_cell_coordinates",
                    "row_index": row_index,
                    "col_index": col_index,
                },
            )

        actual_value_raw = self._cell_value(cells, row_index, col_index)
        if actual_value_raw is None:
            return "fail", {"reason": "cell_not_found", "row_index": row_index, "col_index": col_index}

        expected = _parse_numeric(claim.value_parsed if claim.value_parsed is not None else claim.value_raw)
        actual = _parse_numeric(actual_value_raw)

        if expected is None or actual is None:
            return "fail", {"reason": "numeric_parse_failed", "expected": claim.value_raw, "actual": actual_value_raw}

        if abs(expected - actual) > 1e-9:
            return (
                "fail",
                {
                    "reason": "value_mismatch",
                    "expected": expected,
                    "actual": actual,
                    "row_index": row_index,
                    "col_index": col_index,
                },
            )

        dataset_matches: list[dict[str, object]] = []
        dataset_failures: list[str] = []
        for item, selectors in bundle:
            for selector in selectors:
                if selector.selector_type != "DataCellSelector":
                    continue
                if self.structured_data_service is None:
                    dataset_failures.append("structured_data_service_unavailable")
                    continue
                try:
                    payload = json.loads(selector.selector_json)
                    match = self.structured_data_service.resolve_data_cell(item.resource_id, payload)
                    resolved_numeric = _parse_numeric(match.value_raw)
                    if resolved_numeric is None:
                        dataset_failures.append(
                            f"dataset_numeric_parse_failed:{item.resource_id}:{match.table_name}:{match.column_name}"
                        )
                        continue
                    dataset_matches.append(
                        {
                            "resource_id": item.resource_id,
                            "table_name": match.table_name,
                            "row_number": match.row_number,
                            "column_name": match.column_name,
                            "column_index": match.column_index,
                            "value_raw": match.value_raw,
                            "value_parsed": resolved_numeric,
                        }
                    )
                except Exception as exc:
                    dataset_failures.append(str(exc))

        if dataset_matches:
            if all(abs(float(m["value_parsed"]) - actual) > 1e-9 for m in dataset_matches):
                return (
                    "fail",
                    {
                        "reason": "cross_source_mismatch",
                        "pdf_table_value": actual,
                        "dataset_values": [m["value_parsed"] for m in dataset_matches],
                        "dataset_matches": dataset_matches,
                    },
                )
            if any(abs(float(m["value_parsed"]) - expected) > 1e-9 for m in dataset_matches):
                return (
                    "fail",
                    {
                        "reason": "dataset_claim_mismatch",
                        "expected": expected,
                        "dataset_values": [m["value_parsed"] for m in dataset_matches],
                        "dataset_matches": dataset_matches,
                    },
                )
        elif dataset_failures:
            return (
                "fail",
                {
                    "reason": "dataset_selector_failed",
                    "errors": dataset_failures[:25],
                },
            )

        # Optional semantic checks where provided by selector and claim.
        units = table_selector_payload.get("units") or {}
        period = table_selector_payload.get("period") or {}

        if claim.currency and units.get("currency") and str(units.get("currency")) != claim.currency:
            return "fail", {"reason": "currency_mismatch", "expected": claim.currency, "actual": units.get("currency")}

        if claim.scale_factor is not None and units.get("scale_factor") is not None:
            if int(units.get("scale_factor")) != int(claim.scale_factor):
                return "fail", {
                    "reason": "scale_factor_mismatch",
                    "expected": claim.scale_factor,
                    "actual": units.get("scale_factor"),
                }

        if claim.period_label and period.get("label") and str(period.get("label")) != claim.period_label:
            return "fail", {"reason": "period_mismatch", "expected": claim.period_label, "actual": period.get("label")}

        return (
            "pass",
            {
                "reason": "quantitative_match_cross_source" if dataset_matches else "quantitative_match",
                "table_id": table_id,
                "row_index": row_index,
                "col_index": col_index,
                "value": actual,
                "dataset_matches": dataset_matches,
            },
        )

    @staticmethod
    def _resolve_row_index(rows: list[str], row_path: object) -> int | None:
        if not isinstance(row_path, list) or not row_path:
            return None
        needle = str(row_path[-1]).strip().lower()
        for idx, value in enumerate(rows):
            if str(value).strip().lower() == needle:
                return idx
        return None

    @staticmethod
    def _resolve_col_index(cols: list[str], col_path: object) -> int | None:
        if not isinstance(col_path, list) or not col_path:
            return None
        needle = str(col_path[-1]).strip().lower()
        for idx, value in enumerate(cols):
            if str(value).strip().lower() == needle:
                return idx
        return None

    @staticmethod
    def _cell_value(cells: list[dict[str, object]], row_index: int, col_index: int) -> str | None:
        for c in cells:
            if int(c.get("row_index", -1)) == int(row_index) and int(c.get("col_index", -1)) == int(
                col_index
            ):
                return str(c.get("value") or "")
        return None


def _parse_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    neg = text.startswith("(") and text.endswith(")")
    cleaned = (
        text.replace(",", "")
        .replace("£", "")
        .replace("$", "")
        .replace("m", "")
        .replace("M", "")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return -parsed if neg else parsed
