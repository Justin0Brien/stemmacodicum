from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VerificationRun:
    id: str
    claim_set_id: str | None
    policy_profile: str
    created_at: str


@dataclass(slots=True)
class VerificationResult:
    id: str
    run_id: str
    claim_id: str
    status: str
    diagnostics_json: str
    created_at: str
