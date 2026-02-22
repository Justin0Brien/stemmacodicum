from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel

from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.ingestion_policy_service import IngestionPolicyService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.structured_data_service import StructuredDataService
from stemmacodicum.application.services.verification_service import VerificationService
from stemmacodicum.cli.context import CLIContext
from stemmacodicum.core.errors import ProjectNotInitializedError
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.structured_data_repo import StructuredDataRepo
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("verify", help="Run deterministic claim verification")
    verify_subparsers = parser.add_subparsers(dest="verify_command", required=True)

    claim_parser = verify_subparsers.add_parser("claim", help="Verify a single claim")
    claim_parser.add_argument("--claim-id", required=True)
    claim_parser.add_argument("--policy", default="strict")
    claim_parser.set_defaults(handler=run_verify_claim)

    set_parser = verify_subparsers.add_parser("set", help="Verify all claims in a claim set")
    set_parser.add_argument("--claim-set", required=True)
    set_parser.add_argument("--policy", default="strict")
    set_parser.set_defaults(handler=run_verify_set)


def _require_initialized_project(ctx: CLIContext) -> None:
    project_service = ProjectService(ctx.paths)
    if not project_service.is_initialized():
        raise ProjectNotInitializedError(
            f"Project is not initialized. Run 'stemma init' first in {ctx.paths.project_root}"
        )
    project_service.init_project()


def _service(ctx: CLIContext) -> VerificationService:
    claim_repo = ClaimRepo(ctx.paths.db_path)
    resource_repo = ResourceRepo(ctx.paths.db_path)
    evidence_repo = EvidenceRepo(ctx.paths.db_path)
    extraction_repo = ExtractionRepo(ctx.paths.db_path)
    verification_repo = VerificationRepo(ctx.paths.db_path)

    binding_service = EvidenceBindingService(
        claim_repo=claim_repo,
        resource_repo=resource_repo,
        evidence_repo=evidence_repo,
    )
    structured_data_service = StructuredDataService(
        resource_repo=resource_repo,
        structured_repo=StructuredDataRepo(ctx.paths.db_path),
        archive_dir=ctx.paths.archive_dir,
        policy_service=IngestionPolicyService(),
    )

    return VerificationService(
        claim_repo=claim_repo,
        evidence_repo=evidence_repo,
        extraction_repo=extraction_repo,
        verification_repo=verification_repo,
        binding_service=binding_service,
        structured_data_service=structured_data_service,
    )


def run_verify_claim(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _service(ctx)

    outcome = service.verify_claim(claim_id=args.claim_id, policy_profile=args.policy)
    lines = [
        f"Run ID: {outcome.run_id}",
        f"Claim ID: {outcome.claim_id}",
        f"Status: {outcome.status.upper()}",
        f"Diagnostics: {outcome.diagnostics}",
    ]
    ctx.console.print(Panel.fit("\n".join(lines), title="Verify Claim"))
    return 0 if outcome.status == "pass" else 1


def run_verify_set(args: argparse.Namespace, ctx: CLIContext) -> int:
    _require_initialized_project(ctx)
    service = _service(ctx)

    outcome = service.verify_claim_set(claim_set_name=args.claim_set, policy_profile=args.policy)
    lines = [
        f"Run ID: {outcome.run_id}",
        f"Total: {outcome.total}",
        f"Passed: {outcome.passed}",
        f"Failed: {outcome.failed}",
    ]
    ctx.console.print(Panel.fit("\n".join(lines), title="Verify Claim Set"))
    return 0 if outcome.failed == 0 else 1
