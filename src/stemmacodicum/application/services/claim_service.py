from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.errors import ClaimError
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.claim import Claim, ClaimSet
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.importers.csv_claim_importer import load_claim_rows_from_csv
from stemmacodicum.infrastructure.importers.json_claim_importer import (
    ClaimJsonError,
    load_claim_rows_from_json,
)
from stemmacodicum.infrastructure.importers.markdown_claim_importer import (
    load_claim_rows_from_markdown,
)


@dataclass(slots=True)
class ClaimImportSummary:
    claim_set_id: str
    claim_set_name: str
    imported: int


class ClaimService:
    def __init__(self, claim_repo: ClaimRepo) -> None:
        self.claim_repo = claim_repo

    def get_or_create_claim_set(self, name: str, description: str | None = None) -> ClaimSet:
        existing = self.claim_repo.get_claim_set_by_name(name)
        if existing is not None:
            return existing

        claim_set = ClaimSet(
            id=new_uuid(),
            name=name,
            description=description,
            created_at=now_utc_iso(),
        )
        self.claim_repo.insert_claim_set(claim_set)
        return claim_set

    def list_claim_sets(self, limit: int = 100) -> list[ClaimSet]:
        return self.claim_repo.list_claim_sets(limit=limit)

    def import_claims(
        self,
        file_path: Path,
        fmt: str,
        claim_set_name: str,
        claim_set_description: str | None = None,
    ) -> ClaimImportSummary:
        path = file_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ClaimError(f"Claim import file not found: {path}")

        claim_set = self.get_or_create_claim_set(claim_set_name, claim_set_description)

        rows = self._load_rows(path, fmt)
        imported = 0
        for row in rows:
            claim = self._build_claim(claim_set.id, row)
            self.claim_repo.insert_claim(claim)
            imported += 1

        return ClaimImportSummary(
            claim_set_id=claim_set.id,
            claim_set_name=claim_set.name,
            imported=imported,
        )

    def list_claims(self, claim_set_name: str | None = None, limit: int = 200) -> list[Claim]:
        claim_set_id: str | None = None
        if claim_set_name:
            claim_set = self.claim_repo.get_claim_set_by_name(claim_set_name)
            if claim_set is None:
                raise ClaimError(f"Claim set not found: {claim_set_name}")
            claim_set_id = claim_set.id

        return self.claim_repo.list_claims(claim_set_id=claim_set_id, limit=limit)

    def _load_rows(self, path: Path, fmt: str) -> list[dict[str, object]]:
        fmt_normalized = fmt.strip().lower()
        if fmt_normalized == "csv":
            return [dict(r) for r in load_claim_rows_from_csv(path)]
        if fmt_normalized == "json":
            try:
                return load_claim_rows_from_json(path)
            except ClaimJsonError as exc:
                raise ClaimError(str(exc)) from exc
        if fmt_normalized in {"md", "markdown"}:
            return [dict(r) for r in load_claim_rows_from_markdown(path)]
        raise ClaimError(f"Unsupported claim import format: {fmt}")

    def _build_claim(self, claim_set_id: str, row: dict[str, object]) -> Claim:
        claim_type = str(row.get("claim_type") or "").strip().lower()
        narrative_text = _opt_str(row.get("narrative_text"))

        if not claim_type:
            claim_type = "narrative" if narrative_text else "quantitative"

        subject = _opt_str(row.get("subject"))
        predicate = _opt_str(row.get("predicate"))
        object_text = _opt_str(row.get("object_text"))
        value_raw = _opt_str(row.get("value_raw"))
        value_parsed = _opt_float(row.get("value_parsed"))
        currency = _opt_str(row.get("currency"))
        scale_factor = _opt_int(row.get("scale_factor"))
        period_label = _opt_str(row.get("period_label"))
        source_cite_id = _opt_str(row.get("source_cite_id"))

        if claim_type == "narrative":
            if not narrative_text:
                raise ClaimError("Narrative claim missing narrative_text")
        elif claim_type == "quantitative":
            if not subject or not predicate:
                raise ClaimError("Quantitative claim requires subject and predicate")
            if value_raw is None and value_parsed is None:
                raise ClaimError("Quantitative claim requires value_raw or value_parsed")
        else:
            raise ClaimError(f"Unsupported claim_type: {claim_type}")

        now = now_utc_iso()
        return Claim(
            id=new_uuid(),
            claim_set_id=claim_set_id,
            claim_type=claim_type,
            subject=subject,
            predicate=predicate,
            object_text=object_text,
            narrative_text=narrative_text,
            value_raw=value_raw,
            value_parsed=value_parsed,
            currency=currency,
            scale_factor=scale_factor,
            period_label=period_label,
            source_cite_id=source_cite_id,
            status="active",
            created_at=now,
            updated_at=now,
        )


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _opt_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _opt_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    return int(text)
