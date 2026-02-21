from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from stemmacodicum.application.services.extraction_service import ExtractionCancelledError
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.infrastructure.db.sqlite import get_connection

logger = logging.getLogger(__name__)


class BackgroundImportQueueService:
    _MAX_PROGRESS_EVENTS = 4000
    _TERMINAL_STATUSES = {"done", "failed", "skipped", "cancelled"}
    _DEFAULT_JOB_TIMEOUT_SECONDS = 600

    def __init__(
        self,
        *,
        db_path: Path,
        queue_dir: Path,
        run_import_callback: Callable[..., dict[str, Any]],
    ) -> None:
        self.db_path = db_path
        self.queue_dir = queue_dir
        self._run_import_callback = run_import_callback
        self._wakeup = threading.Event()
        self._stop = threading.Event()
        self._queue_name = "default"
        self.job_timeout_seconds = self._env_non_negative_int(
            "STEMMA_IMPORT_JOB_TIMEOUT_SECONDS",
            default=self._DEFAULT_JOB_TIMEOUT_SECONDS,
        )
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._recover_inflight_jobs()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="background-import-queue")
        self._worker.start()

    def shutdown(self) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def enqueue_upload(
        self,
        *,
        uploaded_filename: str,
        content_bytes: bytes,
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        if not uploaded_filename:
            uploaded_filename = "upload.bin"
        if content_bytes is None:
            raise ValueError("Upload content is missing.")
        suffix = Path(uploaded_filename).suffix
        if len(suffix) > 32:
            suffix = suffix[:32]
        job_id = new_uuid()
        staged_name = f"{job_id}{suffix}"
        staged_path = self.queue_dir / staged_name
        staged_path.write_bytes(content_bytes)
        now = now_utc_iso()
        source = source_uri or f"upload:{uploaded_filename}"

        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO import_jobs (
                    id,
                    queue_name,
                    source_kind,
                    source_uri,
                    original_filename,
                    staged_relpath,
                    size_bytes,
                    status,
                    attempts,
                    detail,
                    progress_json,
                    payload_json,
                    resource_id,
                    error_message,
                    created_at,
                    updated_at,
                    started_at,
                    finished_at
                ) VALUES (?, ?, 'upload', ?, ?, ?, ?, 'queued', 0, ?, NULL, NULL, NULL, NULL, ?, ?, NULL, NULL)
                """,
                (
                    job_id,
                    self._queue_name,
                    source,
                    uploaded_filename,
                    staged_name,
                    len(content_bytes),
                    "Queued for background import.",
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (job_id,)).fetchone()
        self._wakeup.set()
        return self._serialize_job_row(row) if row else {"id": job_id}

    def request_control(self, *, job_id: str, action: str) -> dict[str, Any]:
        safe_action = self._validate_control_action(action)
        now = now_utc_iso()
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"Import job not found: {job_id}")

            status = str(row["status"] or "").strip().lower()
            if status in self._TERMINAL_STATUSES:
                return self._serialize_job_row(row)

            if status == "queued":
                terminal_status = self._terminal_status_for_action(safe_action)
                detail = (
                    "Skipped before processing by user request."
                    if terminal_status == "skipped"
                    else "Cancelled before processing by user request."
                )
                conn.execute(
                    """
                    UPDATE import_jobs
                    SET status = ?,
                        detail = ?,
                        cancel_requested = 1,
                        cancel_action = ?,
                        cancel_requested_at = ?,
                        updated_at = ?,
                        finished_at = ?
                    WHERE id = ?
                    """,
                    (terminal_status, detail, safe_action, now, now, now, job_id),
                )
            elif status == "processing":
                detail = self._build_control_pending_detail(safe_action, automatic=False)
                conn.execute(
                    """
                    UPDATE import_jobs
                    SET cancel_requested = 1,
                        cancel_action = ?,
                        cancel_requested_at = COALESCE(cancel_requested_at, ?),
                        detail = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (safe_action, now, detail, now, job_id),
                )
            conn.commit()
            updated = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (job_id,)).fetchone()
        self._wakeup.set()
        if updated is None:
            raise ValueError(f"Import job not found after update: {job_id}")
        return self._serialize_job_row(updated)

    def status(self, *, limit: int = 200) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 50000))
        with get_connection(self.db_path) as conn:
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
                    SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) AS skipped,
                    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled
                FROM import_jobs
                WHERE queue_name = ?
                """,
                (self._queue_name,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM import_jobs
                WHERE queue_name = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (self._queue_name, safe_limit),
            ).fetchall()

        jobs = [self._serialize_job_row(row) for row in reversed(rows)]
        queue_counts = {
            "total": int(counts["total"] or 0),
            "queued": int(counts["queued"] or 0),
            "processing": int(counts["processing"] or 0),
            "done": int(counts["done"] or 0),
            "failed": int(counts["failed"] or 0),
            "skipped": int(counts["skipped"] or 0),
            "cancelled": int(counts["cancelled"] or 0),
        }
        return {
            "ok": True,
            "queue": queue_counts,
            "jobs": jobs,
        }

    def _ensure_schema(self) -> None:
        with get_connection(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS import_jobs (
                    id TEXT PRIMARY KEY,
                    queue_name TEXT NOT NULL DEFAULT 'default',
                    source_kind TEXT NOT NULL,
                    source_uri TEXT,
                    original_filename TEXT NOT NULL,
                    staged_relpath TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    detail TEXT,
                    progress_json TEXT,
                    payload_json TEXT,
                    resource_id TEXT,
                    error_message TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    cancel_action TEXT,
                    cancel_requested_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_import_jobs_queue_status_created
                ON import_jobs(queue_name, status, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_import_jobs_queue_updated
                ON import_jobs(queue_name, updated_at DESC);
                """
            )
            columns = {
                str(row["name"]).strip().lower()
                for row in conn.execute("PRAGMA table_info(import_jobs)").fetchall()
            }
            if "cancel_requested" not in columns:
                conn.execute(
                    "ALTER TABLE import_jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                )
            if "cancel_action" not in columns:
                conn.execute("ALTER TABLE import_jobs ADD COLUMN cancel_action TEXT")
            if "cancel_requested_at" not in columns:
                conn.execute("ALTER TABLE import_jobs ADD COLUMN cancel_requested_at TEXT")
            conn.commit()

    def _recover_inflight_jobs(self) -> None:
        with get_connection(self.db_path) as conn:
            now = now_utc_iso()
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'skipped',
                    detail = 'Skip request recovered after service restart.',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?)
                WHERE queue_name = ?
                  AND status = 'processing'
                  AND COALESCE(cancel_requested, 0) = 1
                  AND LOWER(COALESCE(cancel_action, '')) = 'skip'
                """,
                (now, now, self._queue_name),
            )
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'cancelled',
                    detail = 'Cancel request recovered after service restart.',
                    updated_at = ?,
                    finished_at = COALESCE(finished_at, ?)
                WHERE queue_name = ?
                  AND status = 'processing'
                  AND COALESCE(cancel_requested, 0) = 1
                  AND LOWER(COALESCE(cancel_action, '')) != 'skip'
                """,
                (now, now, self._queue_name),
            )
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'queued',
                    detail = 'Recovered after service restart.',
                    updated_at = ?
                WHERE queue_name = ?
                  AND status = 'processing'
                  AND COALESCE(cancel_requested, 0) = 0
                """,
                (now, self._queue_name),
            )
            conn.commit()

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            row = self._claim_next_job()
            if row is None:
                self._wakeup.wait(timeout=1.0)
                self._wakeup.clear()
                continue
            self._process_job(row)

    def _claim_next_job(self):
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM import_jobs
                WHERE queue_name = ?
                  AND status = 'queued'
                  AND COALESCE(cancel_requested, 0) = 0
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (self._queue_name,),
            ).fetchone()
            if row is None:
                return None
            now = now_utc_iso()
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'processing',
                    attempts = attempts + 1,
                    detail = ?,
                    updated_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE id = ?
                """,
                ("Starting background import...", now, now, row["id"]),
            )
            conn.commit()
            claimed = conn.execute("SELECT * FROM import_jobs WHERE id = ?", (row["id"],)).fetchone()
        return claimed

    def _process_job(self, row) -> None:
        job_id = str(row["id"])
        staged_relpath = str(row["staged_relpath"] or "")
        staged_path = self.queue_dir / staged_relpath
        source_uri = str(row["source_uri"] or f"upload:{row['original_filename']}")
        uploaded_filename = str(row["original_filename"] or staged_path.name)
        resume_progress = self._parse_json_obj(row["progress_json"]) or {}
        resume_resource_id = str(row["resource_id"]).strip() if row["resource_id"] else None
        started_mono = time.monotonic()
        cancel_state: dict[str, str | None] = {"action": self._load_cancel_action(job_id)}
        last_cancel_poll = 0.0
        auto_kill_requested = False

        def progress_callback(update: dict[str, object]) -> None:
            self._update_job_progress(job_id, update)

        def cancellation_check() -> bool:
            nonlocal last_cancel_poll
            nonlocal auto_kill_requested
            now_mono = time.monotonic()
            if (
                self.job_timeout_seconds > 0
                and not auto_kill_requested
                and (now_mono - started_mono) >= float(self.job_timeout_seconds)
            ):
                timeout_detail = self._build_control_pending_detail("kill", automatic=True)
                requested = self._request_processing_control(
                    job_id=job_id,
                    action="kill",
                    detail=timeout_detail,
                )
                auto_kill_requested = True
                if requested:
                    self._wakeup.set()
            if (now_mono - last_cancel_poll) < 0.25:
                return cancel_state["action"] is not None
            last_cancel_poll = now_mono
            requested = self._load_cancel_action(job_id)
            if requested is not None:
                cancel_state["action"] = requested
                return True
            return False

        try:
            payload = self._run_import_callback(
                file_path=staged_path,
                source_uri=source_uri,
                uploaded_filename=uploaded_filename,
                progress_callback=progress_callback,
                resume_resource_id=resume_resource_id,
                resume_progress=resume_progress,
                cancellation_check=cancellation_check,
            )
            self._mark_job_done(job_id, payload)
        except ExtractionCancelledError as exc:
            action = cancel_state["action"] or self._load_cancel_action(job_id) or "kill"
            if action == "skip":
                self._mark_job_skipped(job_id, str(exc))
            else:
                self._mark_job_cancelled(job_id, str(exc))
        except Exception as exc:
            logger.exception("Background import job failed: %s", job_id)
            self._mark_job_failed(job_id, str(exc))
        finally:
            if staged_path.exists():
                staged_path.unlink(missing_ok=True)

    def _update_job_progress(self, job_id: str, update: dict[str, object]) -> None:
        if not isinstance(update, dict):
            return
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                "SELECT progress_json, detail, status FROM import_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return
            if str(row["status"] or "").strip().lower() in self._TERMINAL_STATUSES:
                return
            progress = self._parse_json_obj(row["progress_json"]) or {}
            step_map = progress.get("step_map")
            if not isinstance(step_map, dict):
                step_map = {}
            stage = str(update.get("stage") or "").strip()
            if stage:
                existing_step = step_map.get(stage)
                if not isinstance(existing_step, dict):
                    existing_step = {}
                page_count = update.get("page_count")
                safe_page_count = self._safe_positive_int(page_count)
                existing_page_count = self._safe_positive_int(existing_step.get("page_count"))
                page_current = update.get("page_current")
                if page_current is None:
                    page_current = update.get("current_page")
                safe_page_current = self._safe_positive_int(page_current)
                existing_page_current = self._safe_positive_int(existing_step.get("page_current"))
                resolved_page_count = (
                    safe_page_count if safe_page_count is not None else existing_page_count
                )
                resolved_page_current = (
                    safe_page_current if safe_page_current is not None else existing_page_current
                )
                resolved_page_current = self._clamp_page_current(
                    resolved_page_current,
                    page_count=resolved_page_count,
                )
                safe_preview = self._safe_preview_payload(
                    update.get("preview"),
                    page_count_hint=resolved_page_count,
                )
                existing_preview = existing_step.get("preview")
                if not isinstance(existing_preview, dict):
                    existing_preview = None
                step_map[stage] = {
                    "state": str(update.get("state") or "pending"),
                    "progress": int(update.get("progress") or 0),
                    "detail": str(update.get("detail") or existing_step.get("detail") or ""),
                    "stats": str(update.get("stats") or existing_step.get("stats") or ""),
                    "page_count": (
                        safe_page_count
                        if safe_page_count is not None
                        else existing_page_count
                    ),
                    "page_current": resolved_page_current,
                    "preview": safe_preview if safe_preview is not None else existing_preview,
                }
                if safe_preview is not None:
                    progress["latest_preview"] = safe_preview
            progress["step_map"] = step_map
            detail = str(update.get("detail") or "")
            if detail:
                progress["status_line"] = detail
            stats = str(update.get("stats") or "")
            if stats:
                progress["counters_line"] = stats
            progress_event = self._build_progress_event(update, step_map=step_map)
            if progress_event is not None:
                events = progress.get("events")
                if not isinstance(events, list):
                    events = []
                events.append(progress_event)
                if len(events) > self._MAX_PROGRESS_EVENTS:
                    events = events[-self._MAX_PROGRESS_EVENTS :]
                progress["events"] = events
                progress["last_event"] = progress_event
            resource_id_update = str(update.get("resource_id") or "").strip() or None
            now = now_utc_iso()
            conn.execute(
                """
                UPDATE import_jobs
                SET progress_json = ?, detail = ?, resource_id = COALESCE(?, resource_id), updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(progress, ensure_ascii=True, sort_keys=True),
                    detail or row["detail"],
                    resource_id_update,
                    now,
                    job_id,
                ),
            )
            conn.commit()

    def _mark_job_done(self, job_id: str, payload: dict[str, Any]) -> None:
        now = now_utc_iso()
        resource_id = None
        vector_status = "n/a"
        if isinstance(payload, dict):
            resource = payload.get("resource")
            if isinstance(resource, dict):
                resource_id = resource.get("id")
            extraction = payload.get("extraction")
            if isinstance(extraction, dict):
                summary = extraction.get("summary")
                if isinstance(summary, dict):
                    vector_status = str(summary.get("vector_status") or vector_status)
                vector_obj = extraction.get("vector")
                if isinstance(vector_obj, dict) and vector_status == "n/a":
                    vector_status = str(vector_obj.get("status") or vector_status)
                if vector_status == "n/a":
                    vector_status = str(extraction.get("status") or vector_status)
        detail = (
            f"Complete • resource {resource_id} • vector {vector_status}"
            if resource_id
            else f"Complete • vector {vector_status}"
        )
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'done',
                    detail = ?,
                    payload_json = ?,
                    resource_id = ?,
                    error_message = NULL,
                    cancel_requested = 0,
                    cancel_action = NULL,
                    cancel_requested_at = NULL,
                    updated_at = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (
                    detail,
                    json.dumps(payload or {}, ensure_ascii=True),
                    resource_id,
                    now,
                    now,
                    job_id,
                ),
            )
            conn.commit()

    def _mark_job_failed(self, job_id: str, error: str) -> None:
        now = now_utc_iso()
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'failed',
                    detail = ?,
                    error_message = ?,
                    cancel_requested = 0,
                    cancel_action = NULL,
                    cancel_requested_at = NULL,
                    updated_at = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (
                    f"Import failed: {error}",
                    error,
                    now,
                    now,
                    job_id,
                ),
            )
            conn.commit()

    def _mark_job_skipped(self, job_id: str, reason: str | None = None) -> None:
        now = now_utc_iso()
        suffix = f": {reason}" if reason else ""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'skipped',
                    detail = ?,
                    error_message = NULL,
                    updated_at = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (f"Import skipped{suffix}", now, now, job_id),
            )
            conn.commit()

    def _mark_job_cancelled(self, job_id: str, reason: str | None = None) -> None:
        now = now_utc_iso()
        suffix = f": {reason}" if reason else ""
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'cancelled',
                    detail = ?,
                    error_message = NULL,
                    updated_at = ?,
                    finished_at = ?
                WHERE id = ?
                """,
                (f"Import cancelled{suffix}", now, now, job_id),
            )
            conn.commit()

    def _request_processing_control(self, *, job_id: str, action: str, detail: str) -> bool:
        safe_action = self._validate_control_action(action)
        now = now_utc_iso()
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE import_jobs
                SET cancel_requested = 1,
                    cancel_action = ?,
                    cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    detail = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status = 'processing'
                """,
                (safe_action, now, detail, now, job_id),
            )
            conn.commit()
            return int(cursor.rowcount or 0) > 0

    def _load_cancel_action(self, job_id: str) -> str | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT status, cancel_requested, cancel_action
                FROM import_jobs
                WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        status = str(row["status"] or "").strip().lower()
        if status in self._TERMINAL_STATUSES:
            return None
        requested = int(row["cancel_requested"] or 0) > 0
        if not requested:
            return None
        normalized = str(row["cancel_action"] or "kill").strip().lower()
        if normalized not in {"skip", "kill"}:
            normalized = "kill"
        return normalized

    @staticmethod
    def _validate_control_action(action: str) -> str:
        normalized = str(action or "").strip().lower()
        if normalized in {"skip", "kill"}:
            return normalized
        raise ValueError(f"Unsupported control action: {action}")

    @staticmethod
    def _terminal_status_for_action(action: str) -> str:
        return "skipped" if action == "skip" else "cancelled"

    @staticmethod
    def _build_control_pending_detail(action: str, *, automatic: bool) -> str:
        if action == "skip":
            return "Skip requested; waiting for current import stage to stop."
        if automatic:
            return "Automatic kill requested after timeout; waiting for parser worker shutdown."
        return "Kill requested; waiting for parser worker shutdown."

    @staticmethod
    def _parse_json_obj(raw: str | None) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _safe_short_text(value: object, *, max_len: int = 600) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        if len(text) <= max_len:
            return text
        return text[: max_len - 1].rstrip() + "…"

    def _build_progress_event(
        self,
        update: dict[str, object],
        *,
        step_map: dict[str, object],
    ) -> dict[str, Any] | None:
        stage = self._safe_short_text(update.get("stage"), max_len=64)
        if stage is None:
            return None
        event: dict[str, Any] = {
            "emitted_at": self._safe_short_text(update.get("emitted_at"), max_len=64) or now_utc_iso(),
            "stage": stage,
        }
        state = self._safe_short_text(update.get("state"), max_len=48)
        if state is not None:
            event["state"] = state
        progress_value = update.get("progress")
        if isinstance(progress_value, (int, float)):
            event["progress"] = max(0, min(100, int(progress_value)))
        detail = self._safe_short_text(update.get("detail"))
        if detail is not None:
            event["detail"] = detail
        stats = self._safe_short_text(update.get("stats"))
        if stats is not None:
            event["stats"] = stats
        component = self._safe_short_text(update.get("component"), max_len=80)
        if component is not None:
            event["component"] = component
        event_name = self._safe_short_text(update.get("event"), max_len=80)
        if event_name is not None:
            event["event"] = event_name
        duration_ms = update.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
            event["duration_ms"] = int(duration_ms)
        attempt = update.get("attempt")
        if isinstance(attempt, (int, float)) and int(attempt) > 0:
            event["attempt"] = int(attempt)
        attempts_total = update.get("attempts_total")
        if isinstance(attempts_total, (int, float)) and int(attempts_total) > 0:
            event["attempts_total"] = int(attempts_total)

        step_obj = step_map.get(stage)
        if isinstance(step_obj, dict):
            page_count = self._safe_positive_int(step_obj.get("page_count"))
            if page_count is not None:
                event["page_count"] = page_count
            page_current = self._safe_positive_int(step_obj.get("page_current"))
            if page_current is not None:
                event["page_current"] = page_current
            preview = step_obj.get("preview")
            if isinstance(preview, dict):
                preview_kind = self._safe_short_text(preview.get("kind"), max_len=48)
                if preview_kind is not None:
                    event["preview_kind"] = preview_kind
                preview_label = self._safe_short_text(preview.get("label"), max_len=160)
                if preview_label is not None:
                    event["preview_label"] = preview_label
        return event

    @staticmethod
    def _safe_positive_int(value: object) -> int | None:
        if isinstance(value, (int, float)):
            parsed = int(value)
            if parsed > 0:
                return parsed
        return None

    @staticmethod
    def _clamp_page_current(page_current: int | None, *, page_count: int | None) -> int | None:
        if page_current is None:
            return None
        if page_count is not None and page_count > 0:
            return max(1, min(page_current, page_count))
        return max(1, page_current)

    @staticmethod
    def _safe_preview_payload(
        raw: object,
        *,
        page_count_hint: int | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        preview: dict[str, Any] = {}
        kind = str(raw.get("kind") or "").strip().lower()
        if kind:
            preview["kind"] = kind[:48]
        label = str(raw.get("label") or "").strip()
        if label:
            preview["label"] = label[:160]
        text = str(raw.get("text") or "").strip()
        if text:
            preview["text"] = text[:420]

        page_count_raw = raw.get("page_count")
        raw_page_count = BackgroundImportQueueService._safe_positive_int(page_count_raw)
        resolved_page_count = page_count_hint if page_count_hint is not None else raw_page_count
        if resolved_page_count is None:
            resolved_page_count = raw_page_count
        elif raw_page_count is not None:
            resolved_page_count = min(resolved_page_count, raw_page_count)
        if resolved_page_count is not None:
            preview["page_count"] = resolved_page_count
        page_current_raw = raw.get("page_current")
        raw_page_current = BackgroundImportQueueService._safe_positive_int(page_current_raw)
        resolved_page_current = BackgroundImportQueueService._clamp_page_current(
            raw_page_current,
            page_count=resolved_page_count,
        )
        if resolved_page_current is not None:
            preview["page_current"] = resolved_page_current

        bbox_raw = raw.get("bbox")
        if isinstance(bbox_raw, dict):
            x0 = bbox_raw.get("x0")
            y0 = bbox_raw.get("y0")
            x1 = bbox_raw.get("x1")
            y1 = bbox_raw.get("y1")
            try:
                fx0 = float(x0)
                fy0 = float(y0)
                fx1 = float(x1)
                fy1 = float(y1)
                if fx1 < fx0:
                    fx0, fx1 = fx1, fx0
                if fy1 < fy0:
                    fy0, fy1 = fy1, fy0
                preview["bbox"] = {"x0": fx0, "y0": fy0, "x1": fx1, "y1": fy1}
            except Exception:
                pass

        return preview if preview else None

    def _serialize_job_row(self, row) -> dict[str, Any]:
        payload = self._parse_json_obj(row["payload_json"])
        progress = self._parse_json_obj(row["progress_json"])
        return {
            "id": row["id"],
            "queue_name": row["queue_name"],
            "source_kind": row["source_kind"],
            "source_uri": row["source_uri"],
            "original_filename": row["original_filename"],
            "size_bytes": int(row["size_bytes"] or 0),
            "status": row["status"],
            "attempts": int(row["attempts"] or 0),
            "detail": row["detail"],
            "progress": progress,
            "resource_id": row["resource_id"],
            "error_message": row["error_message"],
            "cancel_requested": bool(int(row["cancel_requested"] or 0)),
            "cancel_action": row["cancel_action"],
            "cancel_requested_at": row["cancel_requested_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "payload": payload,
        }

    @staticmethod
    def _env_non_negative_int(name: str, *, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            parsed = int(str(raw).strip())
        except (TypeError, ValueError):
            return default
        return max(0, parsed)
