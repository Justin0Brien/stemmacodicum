from __future__ import annotations

import json
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stemmacodicum.application.services.extraction_service import ExtractionService
from stemmacodicum.application.services.ingestion_policy_service import IngestionPolicyService
from stemmacodicum.application.services.ingestion_service import IngestionService
from stemmacodicum.application.services.structured_data_service import StructuredDataService
from stemmacodicum.infrastructure.db.repos.extraction_repo import ExtractionRepo
from stemmacodicum.infrastructure.parsers.docling_adapter import DoclingAdapter


@dataclass(slots=True)
class PipelineStats:
    candidates: int
    already_processed: int
    processed: int
    ingested: int
    duplicates: int
    extracted: int
    structured_profiled: int
    structured_profile_failed: int
    structured_profile_skipped: int
    skipped_extraction: int
    failed: int
    remaining_unprocessed: int
    state_entries_before: int
    state_entries_after: int


class BatchImportService:

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".csv",
        ".txt",
        ".rst",
        ".adoc",
        ".tex",
        ".html",
        ".htm",
        ".xml",
        ".xhtml",
        ".svg",
        ".fb2",
        ".dita",
        ".dbk",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".odg",
        ".epub",
        ".oxps",
        ".3mf",
        ".md",
    }

    EXTRACTABLE_MEDIA_TYPES = set(DoclingAdapter.SUPPORTED_MEDIA_TYPES)

    def __init__(
        self,
        ingestion_service: IngestionService,
        extraction_service: ExtractionService,
        extraction_repo: ExtractionRepo,
        policy_service: IngestionPolicyService | None,
        structured_data_service: StructuredDataService | None,
        state_path: Path,
        log_path: Path,
    ) -> None:
        self.ingestion_service = ingestion_service
        self.extraction_service = extraction_service
        self.extraction_repo = extraction_repo
        self.policy_service = policy_service or IngestionPolicyService()
        self.structured_data_service = structured_data_service
        self.state_path = state_path
        self.log_path = log_path

    def find_candidates(self, root: Path) -> list[Path]:
        root = root.expanduser().resolve()
        files = [p for p in root.rglob("*") if p.is_file()]

        matched: list[Path] = []
        for path in files:
            if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                continue
            matched.append(path)

        return sorted(matched)

    # Backward-compatible alias while external callers migrate naming.
    def find_financial_candidates(self, root: Path) -> list[Path]:
        return self.find_candidates(root)

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
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> PipelineStats:
        candidates = self.find_candidates(root)
        processed_set = self.load_state()
        state_entries_before = len(processed_set)

        pending = [candidate for candidate in candidates if str(candidate) not in processed_set]
        already_processed = len(candidates) - len(pending)
        worklist = pending[:max_files] if max_files is not None else pending
        planned_total = len(worklist)

        self._emit_progress(
            progress_callback,
            {
                "event": "scan_complete",
                "candidates": len(candidates),
                "already_processed": already_processed,
                "planned_total": planned_total,
                "run_extraction": run_extraction,
            },
        )

        for skipped in (candidate for candidate in candidates if str(candidate) in processed_set):
            self._emit_progress(
                progress_callback,
                {
                    "event": "state_skip",
                    "path": str(skipped),
                },
            )

        ingested = 0
        duplicates = 0
        extracted = 0
        structured_profiled = 0
        structured_profile_failed = 0
        structured_profile_skipped = 0
        skipped_extraction = 0
        failed = 0
        processed = 0

        for index, candidate in enumerate(worklist, start=1):
            candidate_key = str(candidate)
            started = time.perf_counter()

            self._emit_progress(
                progress_callback,
                {
                    "event": "file_start",
                    "index": index,
                    "total": planned_total,
                    "path": candidate_key,
                },
            )

            try:
                ingest = self.ingestion_service.ingest_file(candidate)
                if ingest.status == "ingested":
                    ingested += 1
                else:
                    duplicates += 1

                extract_status = "skipped"
                tables_found = 0
                parse_elapsed_seconds: float | None = None
                page_count: int | None = None
                pages_per_second: float | None = None
                policy = self.policy_service.decide(
                    media_type=ingest.resource.media_type,
                    original_filename=ingest.resource.original_filename,
                )

                if policy.resource_kind == "structured_data":
                    if not run_extraction:
                        structured_profile_skipped += 1
                        skipped_extraction += 1
                        extract_status = "structured-skip:processing-disabled"
                    elif self.structured_data_service is None:
                        structured_profile_skipped += 1
                        skipped_extraction += 1
                        extract_status = "structured-skip:no-profiler"
                    else:
                        profile = self.structured_data_service.profile_resource(ingest.resource.id, force=False)
                        if profile.status == "success":
                            structured_profiled += 1
                            extract_status = f"structured-profiled:{profile.table_count}"
                        elif profile.status == "skipped":
                            structured_profile_skipped += 1
                            extract_status = "structured-profile-skip"
                        else:
                            structured_profile_failed += 1
                            extract_status = f"structured-profile-failed:{profile.error or 'unknown'}"
                elif run_extraction and policy.should_extract_auto and DoclingAdapter.supports(
                    ingest.resource.media_type,
                    ingest.resource.original_filename,
                ):
                    already = self.extraction_repo.list_recent_runs(ingest.resource.id, limit=1)
                    if already:
                        skipped_extraction += 1
                        extract_status = "already-extracted"
                    else:
                        with _alarm_timeout(extract_timeout_seconds):
                            summary = self.extraction_service.extract_resource(ingest.resource.id)
                        extracted += 1
                        extract_status = f"extracted:{summary.tables_found}"
                        tables_found = summary.tables_found
                        parse_elapsed_seconds = summary.elapsed_seconds
                        page_count = summary.page_count
                        pages_per_second = summary.pages_per_second
                else:
                    skipped_extraction += 1

                elapsed_seconds = time.perf_counter() - started
                self.append_log(
                    {
                        "path": candidate_key,
                        "ingest_status": ingest.status,
                        "resource_id": ingest.resource.id,
                        "digest": ingest.resource.digest_sha256,
                        "extract_status": extract_status,
                        "elapsed_seconds": round(elapsed_seconds, 4),
                        "parse_elapsed_seconds": round(parse_elapsed_seconds, 4)
                        if parse_elapsed_seconds is not None
                        else None,
                        "page_count": page_count,
                        "pages_per_second": round(pages_per_second, 4) if pages_per_second is not None else None,
                    }
                )
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "file_done",
                        "index": index,
                        "total": planned_total,
                        "path": candidate_key,
                        "ingest_status": ingest.status,
                        "extract_status": extract_status,
                        "tables_found": tables_found,
                        "elapsed_seconds": elapsed_seconds,
                        "parse_elapsed_seconds": parse_elapsed_seconds,
                        "page_count": page_count,
                        "pages_per_second": pages_per_second,
                    },
                )
            except Exception as exc:
                failed += 1
                elapsed_seconds = time.perf_counter() - started
                self.append_log(
                    {
                        "path": candidate_key,
                        "error": str(exc),
                        "elapsed_seconds": round(elapsed_seconds, 4),
                    }
                )
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "file_error",
                        "index": index,
                        "total": planned_total,
                        "path": candidate_key,
                        "error": str(exc),
                        "elapsed_seconds": elapsed_seconds,
                    },
                )

            processed += 1
            processed_set.add(candidate_key)
            self.save_state(processed_set)

        self.save_state(processed_set)
        remaining_unprocessed = max(0, len(candidates) - already_processed - processed)
        return PipelineStats(
            candidates=len(candidates),
            already_processed=already_processed,
            processed=processed,
            ingested=ingested,
            duplicates=duplicates,
            extracted=extracted,
            structured_profiled=structured_profiled,
            structured_profile_failed=structured_profile_failed,
            structured_profile_skipped=structured_profile_skipped,
            skipped_extraction=skipped_extraction,
            failed=failed,
            remaining_unprocessed=remaining_unprocessed,
            state_entries_before=state_entries_before,
            state_entries_after=len(processed_set),
        )

    @staticmethod
    def _emit_progress(
        callback: Callable[[dict[str, object]], None] | None,
        payload: dict[str, object],
    ) -> None:
        if callback is None:
            return
        callback(payload)


# Backward-compatible export.
FinancialPipelineService = BatchImportService


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
