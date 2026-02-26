#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.config import load_paths
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.infrastructure.db.sqlite import get_connection, initialize_schema


@dataclass(slots=True)
class PdfResource:
    resource_id: str
    archived_relpath: str
    original_filename: str
    extraction_run_id: str | None


def require_optional_dependencies() -> tuple[object, object]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf") from exc
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Pillow is required. Install with: pip install pillow pillow-avif-plugin") from exc
    try:
        import pillow_avif  # type: ignore  # noqa: F401
    except Exception:
        # Plugin is optional here; ffmpeg fallback is used when unavailable.
        pass
    return fitz, Image


def load_pdf_resources(db_path: Path, only_missing: bool = True) -> list[PdfResource]:
    where_clause = """
      WHERE (lower(r.original_filename) LIKE '%.pdf' OR lower(r.media_type) = 'application/pdf')
    """
    if only_missing:
        where_clause += " AND NOT EXISTS (SELECT 1 FROM resource_images ri WHERE ri.resource_id = r.id)"
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT r.id, r.archived_relpath, r.original_filename,
                   (SELECT er.id FROM extraction_runs er
                    WHERE er.resource_id = r.id
                    ORDER BY er.created_at DESC LIMIT 1) AS extraction_run_id
            FROM resources r
            {where_clause}
            ORDER BY r.ingested_at ASC
            """
        ).fetchall()
    return [
        PdfResource(
            resource_id=str(row["id"]),
            archived_relpath=str(row["archived_relpath"]),
            original_filename=str(row["original_filename"] or ""),
            extraction_run_id=str(row["extraction_run_id"]) if row["extraction_run_id"] else None,
        )
        for row in rows
    ]


def ensure_image_archive_dir(paths) -> Path:
    image_dir = paths.stemma_dir / "image_archive"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def save_as_avif(image, output_path: Path, quality: int) -> bytes:
    """Save image as AVIF and return the encoded bytes."""
    try:
        buf = io.BytesIO()
        image.save(buf, format="AVIF", quality=quality, speed=8, dpi=(72, 72))
        raw = buf.getvalue()
        output_path.write_bytes(raw)
        return raw
    except Exception:
        pass

    with tempfile.TemporaryDirectory(prefix="stemma-avif-") as tmpdir:
        source_png = Path(tmpdir) / "source.png"
        image.save(source_png, format="PNG", optimize=True, dpi=(72, 72))
        ffmpeg = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_png),
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-crf",
            str(max(10, min(63, int(quality)) + 18)),
            "-b:v",
            "0",
            "-cpu-used",
            "8",
            str(output_path),
        ]
        result = subprocess.run(ffmpeg, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Unable to encode AVIF. ffmpeg failed: {result.stderr.strip()}")
    return output_path.read_bytes()


def describe_with_moondream(image_path: Path, model_name: str) -> str | None:
    try:
        import ollama  # type: ignore
    except Exception:
        return None
    try:
        client = ollama.Client()
        response = client.generate(
            model=model_name,
            prompt=(
                "Describe this image in 1-3 concise sentences for archival search. "
                "Focus on entities, chart/table content, and any visible text."
            ),
            images=[str(image_path)],
            options={"temperature": 0.1},
        )
    except Exception:
        return None
    if isinstance(response, dict):
        text = str(response.get("response", "")).strip()
    else:
        text = str(getattr(response, "response", "") or "").strip()
    return text or None


def batch_insert_resource_images(db_path: Path, rows: list[dict]) -> None:
    now = now_utc_iso()
    params = []
    for row in rows:
        description = row["description"]
        params.append((
            str(uuid.uuid4()),
            row["resource_id"],
            row["extraction_run_id"],
            int(row["page_index"]),
            int(row["image_index"]),
            int(row["source_xref"]),
            row["source_name"],
            row["source_format"],
            int(row["source_width_px"]),
            int(row["source_height_px"]),
            float(row["rendered_width_mm"]),
            float(row["rendered_height_mm"]),
            int(row["output_width_px"]),
            int(row["output_height_px"]),
            row["output_file_relpath"],
            row["output_file_sha256"],
            description,
            row["description_model"],
            now_utc_iso() if description else None,
            json.dumps(row["bbox_json"], ensure_ascii=True),
            json.dumps(row["metadata_json"], ensure_ascii=True, sort_keys=True),
            now,
        ))
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO resource_images (
              id,
              resource_id,
              extraction_run_id,
              page_index,
              image_index,
              source_xref,
              source_name,
              source_format,
              source_width_px,
              source_height_px,
              rendered_width_mm,
              rendered_height_mm,
              output_width_px,
              output_height_px,
              output_file_relpath,
              output_file_sha256,
              description_text,
              description_model,
              description_generated_at,
              bbox_json,
              metadata_json,
              created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        conn.commit()


def process_resource(
    *,
    db_path: Path,
    archive_dir: Path,
    image_archive_dir: Path,
    resource: PdfResource,
    avif_quality: int,
    describe_images: bool,
    moondream_model: str,
    fitz,
    image_cls,
) -> tuple[int, int, list[str]]:
    log: list[str] = []
    pdf_path = (archive_dir / resource.archived_relpath).resolve()
    if not pdf_path.exists():
        log.append(f"  missing archive file: {pdf_path}")
        return 0, 0, log
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        log.append(f"  could not open PDF ({type(exc).__name__}: {exc}); skipping.")
        return 0, 0, log

    resample_lanczos = getattr(getattr(image_cls, "Resampling", image_cls), "LANCZOS", image_cls.LANCZOS)
    pending_rows: list[dict] = []

    try:
        page_count = doc.page_count
        log.append(f"  pages: {page_count}")
        for page_index, page in enumerate(doc):
            page_area = page.rect.width * page.rect.height
            images = page.get_images(full=True)
            if not images:
                continue
            log.append(f"  page {page_index + 1}/{page_count}: {len(images)} image(s) found")
            image_counter = 0
            for info in images:
                xref = int(info[0])
                source_width = int(info[2] or 0)
                source_height = int(info[3] or 0)
                source_name = str(info[7] or "")
                source_format = str(info[8] or "")
                extracted = doc.extract_image(xref)
                if not extracted:
                    continue
                raw_bytes = extracted.get("image")
                if not isinstance(raw_bytes, (bytes, bytearray)):
                    continue
                try:
                    pil_image = image_cls.open(io.BytesIO(raw_bytes)).convert("RGB")
                except Exception:
                    continue
                rects = page.get_image_rects(xref)
                if not rects:
                    rects = [fitz.Rect(0, 0, max(1, source_width), max(1, source_height))]

                for rect in rects:
                    rendered_width_mm = float(rect.width) * 25.4 / 72.0
                    rendered_height_mm = float(rect.height) * 25.4 / 72.0

                    if rendered_width_mm < 20.0 or rendered_height_mm < 20.0:
                        continue

                    image_area = rect.width * rect.height
                    if image_area / page_area <= 0.02:
                        continue

                    image_counter += 1

                    dpi_x = (source_width / rect.width) * 72.0 if rect.width > 0 else 72.0
                    dpi_y = (source_height / rect.height) * 72.0 if rect.height > 0 else 72.0

                    if dpi_x > 140.0 or dpi_y > 140.0:
                        output_width_px = max(1, int(round((rect.width / 72.0) * 140.0)))
                        output_height_px = max(1, int(round((rect.height / 72.0) * 140.0)))
                    else:
                        output_width_px = source_width
                        output_height_px = source_height

                    resized = pil_image.resize((output_width_px, output_height_px), resample_lanczos)

                    filename = f"{uuid.uuid4()}.avif"
                    output_path = image_archive_dir / filename
                    log.append(f"    saving image {image_counter} -> {filename}")
                    avif_bytes = save_as_avif(resized, output_path, avif_quality)
                    sha256 = hashlib.sha256(avif_bytes).hexdigest()
                    relpath = str(output_path.relative_to(image_archive_dir.parent))

                    pending_rows.append({
                        "output_path": output_path,
                        "resource_id": resource.resource_id,
                        "extraction_run_id": resource.extraction_run_id,
                        "page_index": page_index,
                        "image_index": image_counter,
                        "source_xref": xref,
                        "source_name": source_name,
                        "source_format": source_format,
                        "source_width_px": max(0, source_width),
                        "source_height_px": max(0, source_height),
                        "rendered_width_mm": rendered_width_mm,
                        "rendered_height_mm": rendered_height_mm,
                        "output_width_px": output_width_px,
                        "output_height_px": output_height_px,
                        "output_file_relpath": relpath,
                        "output_file_sha256": sha256,
                        "description": None,
                        "description_model": None,
                        "bbox_json": {
                            "x0": float(rect.x0),
                            "y0": float(rect.y0),
                            "x1": float(rect.x1),
                            "y1": float(rect.y1),
                        },
                        "metadata_json": {
                            "resource_original_filename": resource.original_filename,
                            "page_number": page_index + 1,
                            "source_extracted_ext": str(extracted.get("ext") or ""),
                            "source_colorspace": str(extracted.get("colorspace") or ""),
                            "source_bpc": extracted.get("bpc"),
                            "source_size_bytes": int(extracted.get("size") or 0),
                            "target_dpi": 140 if (dpi_x > 140.0 or dpi_y > 140.0) else "original",
                            "compression": "lossy_avif",
                            "quality_preference": "small_size",
                        },
                    })
    finally:
        doc.close()

    # Dispatch all moondream calls concurrently (pure I/O — HTTP to Ollama)
    if describe_images and pending_rows:
        with ThreadPoolExecutor(max_workers=min(4, len(pending_rows))) as pool:
            futures = [
                pool.submit(describe_with_moondream, row["output_path"], moondream_model)
                for row in pending_rows
            ]
            for row, future in zip(pending_rows, futures):
                desc = future.result()
                if desc:
                    row["description"] = desc
                    row["description_model"] = moondream_model

    described = sum(1 for r in pending_rows if r["description"])

    if pending_rows:
        batch_insert_resource_images(db_path, pending_rows)

    return len(pending_rows), described, log


def run(args: argparse.Namespace) -> int:
    fitz, image_cls = require_optional_dependencies()
    paths = load_paths(Path(args.project_root))

    if getattr(args, "refresh", False):
        print("Refreshing archive: clearing existing images and database records...", flush=True)
        image_archive_dir = paths.stemma_dir / "image_archive"
        if image_archive_dir.exists():
            shutil.rmtree(image_archive_dir, ignore_errors=True)
        with get_connection(paths.db_path) as conn:
            conn.execute("DROP TRIGGER IF EXISTS block_delete_resource_images")
            conn.execute("DELETE FROM resource_images")
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS block_delete_resource_images
                BEFORE DELETE ON resource_images
                BEGIN
                    SELECT RAISE(ABORT, 'DELETE disabled for resource_images');
                END
                """
            )
            conn.commit()

    schema_path = Path(__file__).parent.parent / "src" / "stemmacodicum" / "infrastructure" / "db" / "schema.sql"
    initialize_schema(paths.db_path, schema_path)
    resources = load_pdf_resources(paths.db_path, only_missing=not args.force)
    if not resources:
        print("No PDF resources need image extraction.")
        return 0
    image_archive_dir = ensure_image_archive_dir(paths)
    total_saved = 0
    total_described = 0
    workers = min(args.workers, len(resources))

    print(f"Processing {len(resources)} PDF resources (workers: {workers})...", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {
            executor.submit(
                process_resource,
                db_path=paths.db_path,
                archive_dir=paths.archive_dir,
                image_archive_dir=image_archive_dir,
                resource=resource,
                avif_quality=args.avif_quality,
                describe_images=args.describe,
                moondream_model=args.moondream_model,
                fitz=fitz,
                image_cls=image_cls,
            ): (idx, resource)
            for idx, resource in enumerate(resources, start=1)
        }
        for future in as_completed(future_to_meta):
            idx, resource = future_to_meta[future]
            print(f"[{idx}/{len(resources)}] {resource.resource_id} {resource.original_filename}", flush=True)
            try:
                saved, described, log_lines = future.result()
            except Exception as exc:
                print(f"  ERROR: {exc}", flush=True)
                continue
            for line in log_lines:
                print(line, flush=True)
            total_saved += saved
            total_described += described
            print(f"  extracted images: {saved}, described: {described}", flush=True)

    print("")
    print("Image extraction summary:")
    print(f"  images stored:      {total_saved}")
    print(f"  descriptions saved: {total_described}")
    print(f"  archive:            {image_archive_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract embedded PDF images to AVIF archive and persist metadata.",
    )
    parser.add_argument("--project-root", default=str(Path.cwd()), help="Project root containing .stemma")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Reprocess PDFs even if images exist")
    parser.add_argument("--refresh", action=argparse.BooleanOptionalAction, default=False, help="Clear existing images and database records")
    parser.add_argument("--avif-quality", type=int, default=50, help="AVIF quality (lower is smaller)")
    parser.add_argument(
        "--describe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate image descriptions using Ollama moondream",
    )
    parser.add_argument("--moondream-model", default="moondream:latest", help="Ollama model name for image descriptions")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Number of parallel PDF workers (default: min(4, cpu_count))",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
