from __future__ import annotations

import json
from dataclasses import dataclass

from stemmacodicum.core.errors import EvidenceBindingError
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.evidence import EvidenceItem, EvidenceSelector
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo


@dataclass(slots=True)
class BindingValidation:
    claim_id: str
    ok: bool
    missing_roles: list[str]
    evidence_with_too_few_selectors: list[str]


class EvidenceBindingService:
    REQUIRED_ROLES_FOR_QUANT = ["value-cell", "row-header", "column-header", "caption"]

    def __init__(
        self,
        claim_repo: ClaimRepo,
        resource_repo: ResourceRepo,
        evidence_repo: EvidenceRepo,
    ) -> None:
        self.claim_repo = claim_repo
        self.resource_repo = resource_repo
        self.evidence_repo = evidence_repo

    def bind_evidence(
        self,
        claim_id: str,
        resource_id: str,
        role: str,
        selectors: list[dict[str, object]],
        page_index: int | None = None,
        note: str | None = None,
    ) -> str:
        claim = self.claim_repo.get_claim_by_id(claim_id)
        if claim is None:
            raise EvidenceBindingError(f"Claim not found: {claim_id}")

        resource = self.resource_repo.get_by_id(resource_id)
        if resource is None:
            raise EvidenceBindingError(f"Resource not found: {resource_id}")

        if len(selectors) < 1:
            raise EvidenceBindingError("At least one selector is required to add evidence")

        item = EvidenceItem(
            id=new_uuid(),
            resource_id=resource_id,
            role=role,
            page_index=page_index,
            note=note,
            created_at=now_utc_iso(),
        )
        self.evidence_repo.insert_evidence_item(item)

        selector_types_seen: set[str] = set()
        for raw_selector in selectors:
            selector_type = str(raw_selector.get("type") or "").strip()
            if not selector_type:
                raise EvidenceBindingError("Each selector must include a non-empty 'type' field")
            selector_types_seen.add(selector_type)

            selector = EvidenceSelector(
                id=new_uuid(),
                evidence_id=item.id,
                selector_type=selector_type,
                selector_json=json.dumps(raw_selector, ensure_ascii=True, sort_keys=True),
                created_at=now_utc_iso(),
            )
            self.evidence_repo.insert_selector(selector)

        self.evidence_repo.bind_claim_to_evidence(claim_id=claim_id, evidence_id=item.id, created_at=now_utc_iso())
        return item.id

    def validate_binding(self, claim_id: str) -> BindingValidation:
        claim = self.claim_repo.get_claim_by_id(claim_id)
        if claim is None:
            raise EvidenceBindingError(f"Claim not found: {claim_id}")

        evidence_bundle = self.evidence_repo.list_evidence_for_claim(claim_id)
        roles_present = [item.role for item, _ in evidence_bundle]

        required_roles: list[str]
        if claim.claim_type == "quantitative":
            required_roles = self.REQUIRED_ROLES_FOR_QUANT
        else:
            required_roles = ["quote"]

        missing_roles = [r for r in required_roles if r not in roles_present]

        weak: list[str] = []
        for item, selectors in evidence_bundle:
            distinct_types = {s.selector_type for s in selectors}
            if len(distinct_types) < 2:
                weak.append(item.id)

        return BindingValidation(
            claim_id=claim_id,
            ok=(not missing_roles and not weak),
            missing_roles=missing_roles,
            evidence_with_too_few_selectors=weak,
        )
