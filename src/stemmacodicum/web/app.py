from __future__ import annotations

import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from stemmacodicum.application.services.ceapf_service import CEAPFService
from stemmacodicum.application.services.claim_service import ClaimService
from stemmacodicum.application.services.evidence_binding_service import EvidenceBindingService
from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.health_service import HealthService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.pipeline_service import FinancialPipelineService
from stemmacodicum.application.services.project_service import ProjectService
from stemmacodicum.application.services.reference_service import ReferenceService
from stemmacodicum.application.services.reporting_service import ReportingService
from stemmacodicum.application.services.trace_service import TraceService
from stemmacodicum.application.services.verification_service import VerificationService
from stemmacodicum.core.config import AppPaths
from stemmacodicum.infrastructure.archive.store import ArchiveStore
from stemmacodicum.infrastructure.db.repos.citation_repo import CitationRepo
from stemmacodicum.infrastructure.db.repos.claim_repo import ClaimRepo
from stemmacodicum.infrastructure.db.repos.evidence_repo import EvidenceRepo
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.db.repos.reference_repo import ReferenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.db.repos.verification_repo import VerificationRepo


class IngestPathRequest(BaseModel):
    path: str
    source_uri: str | None = None


class LinkReferenceRequest(BaseModel):
    cite_id: str
    resource_digest: str


class ExtractRunRequest(BaseModel):
    resource_id: str | None = None
    resource_digest: str | None = None
    profile: str = "default"


class ClaimsImportRequest(BaseModel):
    file_path: str
    fmt: str
    claim_set: str
    description: str | None = None


class BindAddRequest(BaseModel):
    claim_id: str
    resource_id: str | None = None
    resource_digest: str | None = None
    role: str
    selectors: list[dict[str, Any]]
    page_index: int | None = None
    note: str | None = None


class VerifyClaimRequest(BaseModel):
    claim_id: str
    policy: str = "strict"


class VerifySetRequest(BaseModel):
    claim_set: str
    policy: str = "strict"


class CEAPFPropositionRequest(BaseModel):
    proposition: dict[str, Any]


class CEAPFAssertionRequest(BaseModel):
    proposition_id: str
    asserting_agent: str
    modality: str
    evidence_id: str | None = None


class CEAPFRelationRequest(BaseModel):
    relation_type: str
    from_node_type: str
    from_node_id: str
    to_node_type: str
    to_node_id: str


class PipelineRequest(BaseModel):
    root: str
    max_files: int | None = None
    skip_extraction: bool = False
    extract_timeout_seconds: int | None = 300


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def create_app(paths: AppPaths) -> FastAPI:
    app = FastAPI(title="Stemma Codicum Web", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_service = ProjectService(paths)
    project_service.init_project()

    def get_resource_repo() -> ResourceRepo:
        return ResourceRepo(paths.db_path)

    def get_extraction_repo() -> ExtractionRepo:
        return ExtractionRepo(paths.db_path)

    def get_claim_repo() -> ClaimRepo:
        return ClaimRepo(paths.db_path)

    def get_evidence_repo() -> EvidenceRepo:
        return EvidenceRepo(paths.db_path)

    def get_verification_repo() -> VerificationRepo:
        return VerificationRepo(paths.db_path)

    def get_ingestion_service() -> IngestionService:
        return IngestionService(resource_repo=get_resource_repo(), archive_store=ArchiveStore(paths.archive_dir))

    def get_extraction_service() -> ExtractionService:
        return ExtractionService(
            resource_repo=get_resource_repo(),
            extraction_repo=get_extraction_repo(),
            archive_dir=paths.archive_dir,
        )

    def get_reference_service() -> ReferenceService:
        return ReferenceService(
            citation_repo=CitationRepo(paths.db_path),
            reference_repo=ReferenceRepo(paths.db_path),
            resource_repo=get_resource_repo(),
        )

    def get_claim_service() -> ClaimService:
        return ClaimService(get_claim_repo())

    def get_binding_service() -> EvidenceBindingService:
        return EvidenceBindingService(
            claim_repo=get_claim_repo(),
            resource_repo=get_resource_repo(),
            evidence_repo=get_evidence_repo(),
        )

    def get_verify_service() -> VerificationService:
        return VerificationService(
            claim_repo=get_claim_repo(),
            evidence_repo=get_evidence_repo(),
            extraction_repo=get_extraction_repo(),
            verification_repo=get_verification_repo(),
            binding_service=get_binding_service(),
        )

    def get_reporting_service() -> ReportingService:
        return ReportingService(get_verification_repo())

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = Path(__file__).resolve().parent / "static" / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.post("/api/init")
    def api_init() -> dict[str, Any]:
        result = project_service.init_project()
        return {
            "ok": True,
            "db_path": str(result.db_path),
            "paths_created": [str(p) for p in result.paths_created],
        }

    @app.post("/api/ingest/path")
    def api_ingest_path(req: IngestPathRequest) -> dict[str, Any]:
        try:
            result = get_ingestion_service().ingest_file(Path(req.path), source_uri=req.source_uri)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "status": result.status, "resource": _jsonable(result.resource)}

    @app.post("/api/ingest/upload")
    async def api_ingest_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "upload.bin").suffix
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = Path(tmp.name)
                content = await file.read()
                tmp.write(content)

            result = get_ingestion_service().ingest_file(
                temp_path,
                source_uri=f"upload:{file.filename or temp_path.name}",
            )
            return {
                "ok": True,
                "status": result.status,
                "resource": _jsonable(result.resource),
                "uploaded_filename": file.filename,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @app.get("/api/resources")
    def api_resources(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        resources = [_jsonable(r) for r in get_resource_repo().list(limit=limit)]
        return {"ok": True, "count": len(resources), "resources": resources}

    @app.post("/api/refs/import-bib")
    def api_refs_import_bib(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        path = payload.get("bib_path")
        if not path:
            raise HTTPException(status_code=400, detail="bib_path is required")
        try:
            summary = get_reference_service().import_bibtex(Path(str(path)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.get("/api/refs")
    def api_refs(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        refs = [_jsonable(r) for r in ReferenceRepo(paths.db_path).list(limit=limit)]
        return {"ok": True, "count": len(refs), "references": refs}

    @app.get("/api/citations")
    def api_citations(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        citations = [_jsonable(c) for c in CitationRepo(paths.db_path).list(limit=limit)]
        return {"ok": True, "count": len(citations), "citations": citations}

    @app.post("/api/refs/link-resource")
    def api_refs_link(req: LinkReferenceRequest) -> dict[str, Any]:
        try:
            get_reference_service().link_reference_to_resource(req.cite_id, req.resource_digest)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/extract/run")
    def api_extract_run(req: ExtractRunRequest) -> dict[str, Any]:
        try:
            if req.resource_id:
                resource_id = req.resource_id
            elif req.resource_digest:
                resource = get_resource_repo().get_by_digest(req.resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {req.resource_digest}")
                resource_id = resource.id
            else:
                raise ValueError("Provide resource_id or resource_digest")

            summary = get_extraction_service().extract_resource(resource_id, parser_profile=req.profile)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.get("/api/extract/tables")
    def api_extract_tables(
        resource_id: str | None = None,
        resource_digest: str | None = None,
        limit: int = Query(default=100, ge=1, le=100000),
    ) -> dict[str, Any]:
        try:
            if resource_id:
                target_id = resource_id
            elif resource_digest:
                resource = get_resource_repo().get_by_digest(resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {resource_digest}")
                target_id = resource.id
            else:
                raise ValueError("Provide resource_id or resource_digest")
            tables = [_jsonable(t) for t in get_extraction_service().list_tables(target_id, limit=limit)]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(tables), "tables": tables}

    @app.post("/api/claims/import")
    def api_claims_import(req: ClaimsImportRequest) -> dict[str, Any]:
        try:
            summary = get_claim_service().import_claims(
                file_path=Path(req.file_path),
                fmt=req.fmt,
                claim_set_name=req.claim_set,
                claim_set_description=req.description,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "summary": _jsonable(summary)}

    @app.get("/api/claims")
    def api_claims(
        claim_set: str | None = None,
        limit: int = Query(default=100, ge=1, le=100000),
    ) -> dict[str, Any]:
        try:
            claims = [_jsonable(c) for c in get_claim_service().list_claims(claim_set_name=claim_set, limit=limit)]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "count": len(claims), "claims": claims}

    @app.get("/api/claim-sets")
    def api_claim_sets(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        sets = [_jsonable(s) for s in get_claim_service().list_claim_sets(limit=limit)]
        return {"ok": True, "count": len(sets), "claim_sets": sets}

    @app.post("/api/bind/add")
    def api_bind_add(req: BindAddRequest) -> dict[str, Any]:
        try:
            if req.resource_id:
                resource_id = req.resource_id
            elif req.resource_digest:
                resource = get_resource_repo().get_by_digest(req.resource_digest)
                if resource is None:
                    raise ValueError(f"Resource digest not found: {req.resource_digest}")
                resource_id = resource.id
            else:
                raise ValueError("Provide resource_id or resource_digest")

            evidence_id = get_binding_service().bind_evidence(
                claim_id=req.claim_id,
                resource_id=resource_id,
                role=req.role,
                selectors=req.selectors,
                page_index=req.page_index,
                note=req.note,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "evidence_id": evidence_id}

    @app.post("/api/bind/validate")
    def api_bind_validate(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        claim_id = payload.get("claim_id")
        if not claim_id:
            raise HTTPException(status_code=400, detail="claim_id is required")
        try:
            result = get_binding_service().validate_binding(str(claim_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "result": _jsonable(result)}

    @app.post("/api/verify/claim")
    def api_verify_claim(req: VerifyClaimRequest) -> dict[str, Any]:
        try:
            outcome = get_verify_service().verify_claim(req.claim_id, policy_profile=req.policy)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "outcome": _jsonable(outcome)}

    @app.post("/api/verify/set")
    def api_verify_set(req: VerifySetRequest) -> dict[str, Any]:
        try:
            outcome = get_verify_service().verify_claim_set(req.claim_set, policy_profile=req.policy)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "outcome": _jsonable(outcome)}

    @app.get("/api/report/verification")
    def api_report_verification(
        run_id: str,
        json_out: str | None = None,
        md_out: str | None = None,
    ) -> dict[str, Any]:
        service = get_reporting_service()
        try:
            summary = service.build_run_summary(run_id)
            exports: dict[str, str] = {}
            if json_out:
                exports["json_out"] = str(service.export_json_report(run_id, Path(json_out)))
            if md_out:
                exports["md_out"] = str(service.export_markdown_report(run_id, Path(md_out)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"ok": True, "summary": _jsonable(summary), "exports": exports}

    @app.get("/api/trace/claim")
    def api_trace_claim(claim_id: str) -> dict[str, Any]:
        try:
            trace = TraceService(paths.db_path).trace_claim(claim_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "trace": _jsonable(trace)}

    @app.get("/api/trace/resource")
    def api_trace_resource(
        resource_id: str | None = None,
        resource_digest: str | None = None,
    ) -> dict[str, Any]:
        try:
            trace = TraceService(paths.db_path).trace_resource(
                resource_id=resource_id,
                digest_sha256=resource_digest,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "trace": _jsonable(trace)}

    @app.get("/api/trace/citation")
    def api_trace_citation(cite_id: str) -> dict[str, Any]:
        try:
            trace = TraceService(paths.db_path).trace_citation(cite_id)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "trace": _jsonable(trace)}

    @app.get("/api/doctor")
    def api_doctor() -> dict[str, Any]:
        report = HealthService(paths.db_path, paths.archive_dir).run_doctor()
        return {
            "ok": report.ok,
            "checks_run": report.checks_run,
            "issues": [_jsonable(i) for i in report.issues],
        }

    @app.post("/api/ceapf/proposition")
    def api_ceapf_proposition(req: CEAPFPropositionRequest) -> dict[str, Any]:
        try:
            proposition_id = CEAPFService(paths.db_path).create_proposition(req.proposition)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "proposition_id": proposition_id}

    @app.post("/api/ceapf/assertion")
    def api_ceapf_assertion(req: CEAPFAssertionRequest) -> dict[str, Any]:
        try:
            assertion_id = CEAPFService(paths.db_path).create_assertion_event(
                proposition_id=req.proposition_id,
                asserting_agent=req.asserting_agent,
                modality=req.modality,
                evidence_id=req.evidence_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "assertion_id": assertion_id}

    @app.post("/api/ceapf/relation")
    def api_ceapf_relation(req: CEAPFRelationRequest) -> dict[str, Any]:
        try:
            relation_id = CEAPFService(paths.db_path).add_argument_relation(
                relation_type=req.relation_type,
                from_node_type=req.from_node_type,
                from_node_id=req.from_node_id,
                to_node_type=req.to_node_type,
                to_node_id=req.to_node_id,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "relation_id": relation_id}

    @app.get("/api/ceapf/propositions")
    def api_ceapf_list(limit: int = Query(default=100, ge=1, le=100000)) -> dict[str, Any]:
        props = CEAPFService(paths.db_path).list_propositions(limit=limit)
        return {
            "ok": True,
            "count": len(props),
            "propositions": [
                {
                    "id": p.id,
                    "proposition": p.proposition,
                }
                for p in props
            ],
        }

    @app.post("/api/pipeline/financial-pass")
    def api_pipeline(req: PipelineRequest) -> dict[str, Any]:
        pipeline = FinancialPipelineService(
            ingestion_service=get_ingestion_service(),
            extraction_service=get_extraction_service(),
            extraction_repo=get_extraction_repo(),
            state_path=paths.stemma_dir / "financial_pass_state.json",
            log_path=paths.stemma_dir / "financial_pass.log.jsonl",
        )
        try:
            stats = pipeline.run(
                root=Path(req.root),
                max_files=req.max_files,
                run_extraction=not req.skip_extraction,
                extract_timeout_seconds=req.extract_timeout_seconds,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {"ok": True, "stats": _jsonable(stats)}

    return app
