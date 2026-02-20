from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.infrastructure.db.sqlite import get_connection

logger = logging.getLogger(__name__)


class BackgroundImportQueueService:
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

    def status(self, *, limit: int = 200) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 5000))
        with get_connection(self.db_path) as conn:
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued,
                    SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
                    SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS done,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
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
            conn.commit()

    def _recover_inflight_jobs(self) -> None:
        with get_connection(self.db_path) as conn:
            now = now_utc_iso()
            conn.execute(
                """
                UPDATE import_jobs
                SET status = 'queued',
                    detail = 'Recovered after service restart.',
                    updated_at = ?
                WHERE queue_name = ? AND status = 'processing'
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
                WHERE queue_name = ? AND status = 'queued'
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

        def progress_callback(update: dict[str, object]) -> None:
            self._update_job_progress(job_id, update)

        try:
            payload = self._run_import_callback(
                file_path=staged_path,
                source_uri=source_uri,
                uploaded_filename=uploaded_filename,
                progress_callback=progress_callback,
                resume_resource_id=resume_resource_id,
                resume_progress=resume_progress,
            )
            self._mark_job_done(job_id, payload)
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
                "SELECT progress_json, detail FROM import_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
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
                safe_page_count = None
                if isinstance(page_count, int) and page_count > 0:
                    safe_page_count = int(page_count)
                elif isinstance(page_count, float) and page_count > 0:
                    safe_page_count = int(page_count)
                page_current = update.get("page_current")
                if page_current is None:
                    page_current = update.get("current_page")
                safe_page_current = None
                if isinstance(page_current, int) and page_current > 0:
                    safe_page_current = int(page_current)
                elif isinstance(page_current, float) and page_current > 0:
                    safe_page_current = int(page_current)
                step_map[stage] = {
                    "state": str(update.get("state") or "pending"),
                    "progress": int(update.get("progress") or 0),
                    "detail": str(update.get("detail") or existing_step.get("detail") or ""),
                    "stats": str(update.get("stats") or existing_step.get("stats") or ""),
                    "page_count": (
                        safe_page_count
                        if safe_page_count is not None
                        else existing_step.get("page_count")
                    ),
                    "page_current": (
                        safe_page_current
                        if safe_page_current is not None
                        else existing_step.get("page_current")
                    ),
                }
            progress["step_map"] = step_map
            detail = str(update.get("detail") or "")
            if detail:
                progress["status_line"] = detail
            stats = str(update.get("stats") or "")
            if stats:
                progress["counters_line"] = stats
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

    @staticmethod
    def _parse_json_obj(raw: str | None) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

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
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "payload": payload,
        }
