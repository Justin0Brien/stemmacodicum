from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo


@dataclass(slots=True)
class PipelineStats:
    candidates: int
    processed: int
    ingested: int
    duplicates: int
    extracted: int
    skipped_extraction: int
    failed: int


class FinancialPipelineService:
    FINANCIAL_KEYWORDS = (
        "financial",
        "annual report",
        "report and accounts",
        "accounts",
        "account",
        "statement",
        "statements",
        "funding",
        "audit",
    )

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".csv",
        ".txt",
        ".html",
        ".htm",
        ".md",
    }

    EXTRACTABLE_MEDIA_TYPES = {
        "application/pdf",
        "text/markdown",
        "text/plain",
        "text/csv",
    }

    def __init__(
        self,
        ingestion_service: IngestionService,
        extraction_service: ExtractionService,
        extraction_repo: ExtractionRepo,
        state_path: Path,
        log_path: Path,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.extraction_service = extraction_service
        self.extraction_repo = extraction_repo
        self.state_path = state_path
        self.log_path = log_path

    def find_financial_candidates(self, root: Path) -> list[Path]:
        root = root.expanduser().resolve()
        files = [p for p in root.rglob("*") if p.is_file()]

        matched: list[Path] = []
        for path in files:
            if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                continue
            probe = str(path.relative_to(root)).lower().replace("_", " ").replace("-", " ")
            if any(k in probe for k in self.FINANCIAL_KEYWORDS):
                matched.append(path)

        return sorted(matched)

    def load_state(self) -> set[str]:
        if not self.state_path.exists():
            return set()
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        paths = payload.get("processed_paths", [])
        if not isinstance(paths, list):
            return set()
        return set(str(x) for x in paths)

    def save_state(self, processed_paths: set[str]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"processed_paths": sorted(processed_paths)}
        self.state_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def append_log(self, row: dict[str, object]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def run(
        self,
        root: Path,
        *,
        max_files: int | None = None,
        run_extraction: bool = True,
        extract_timeout_seconds: int | None = 300,
    ) -> PipelineStats:
        candidates = self.find_financial_candidates(root)
        processed_set = self.load_state()

        ingested = 0
        duplicates = 0
        extracted = 0
        skipped_extraction = 0
        failed = 0
        processed = 0

        for candidate in candidates:
            candidate_key = str(candidate)
            if candidate_key in processed_set:
                continue
            if max_files is not None and processed >= max_files:
                break

            try:
                ingest = self.ingestion_service.ingest_file(candidate)
                if ingest.status == "ingested":
                    ingested += 1
                else:
                    duplicates += 1

                extract_status = "skipped"
                if run_extraction and ingest.resource.media_type in self.EXTRACTABLE_MEDIA_TYPES:
                    already = self.extraction_repo.list_recent_runs(ingest.resource.id, limit=1)
                    if already:
                        skipped_extraction += 1
                        extract_status = "already-extracted"
                    else:
                        with _alarm_timeout(extract_timeout_seconds):
                            summary = self.extraction_service.extract_resource(ingest.resource.id)
                        extracted += 1
                        extract_status = f"extracted:{summary.tables_found}"
                else:
                    skipped_extraction += 1

                self.append_log(
                    {
                        "path": candidate_key,
                        "ingest_status": ingest.status,
                        "resource_id": ingest.resource.id,
                        "digest": ingest.resource.digest_sha256,
                        "extract_status": extract_status,
                    }
                )
            except Exception as exc:
                failed += 1
                self.append_log(
                    {
                        "path": candidate_key,
                        "error": str(exc),
                    }
                )

            processed += 1
            processed_set.add(candidate_key)
            self.save_state(processed_set)

        self.save_state(processed_set)
        return PipelineStats(
            candidates=len(candidates),
            processed=processed,
            ingested=ingested,
            duplicates=duplicates,
            extracted=extracted,
            skipped_extraction=skipped_extraction,
            failed=failed,
        )


@contextmanager
def _alarm_timeout(seconds: int | None):
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - signal handler
        raise TimeoutError(f"Extraction timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
