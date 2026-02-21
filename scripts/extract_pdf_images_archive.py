#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
import uuid
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
            SELECT r.id, r.archived_relpath, r.original_filename
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
        )
        for row in rows
    ]


def ensure_image_archive_dir(paths) -> Path:
    image_dir = paths.stemma_dir / "image_archive"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def latest_extraction_run_id(db_path: Path, resource_id: str) -> str | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id
            FROM extraction_runs
            WHERE resource_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (resource_id,),
        ).fetchone()
    return str(row["id"]) if row else None


def save_as_avif(image, output_path: Path, quality: int) -> None:
    try:
        image.save(output_path, format="AVIF", quality=quality, speed=8, dpi=(72, 72))
        return
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


def insert_resource_image(
    db_path: Path,
    *,
    resource_id: str,
    extraction_run_id: str | None,
    page_index: int,
    image_index: int,
    source_xref: int,
    source_name: str,
    source_format: str,
    source_width_px: int,
    source_height_px: int,
    rendered_width_mm: float,
    rendered_height_mm: float,
    output_width_px: int,
    output_height_px: int,
    output_file_relpath: str,
    output_file_sha256: str,
    description_text: str | None,
    description_model: str | None,
    bbox_json: dict[str, float],
    metadata_json: dict[str, object],
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
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
            (
                str(uuid.uuid4()),
                resource_id,
                extraction_run_id,
                int(page_index),
                int(image_index),
                int(source_xref),
                source_name,
                source_format,
                int(source_width_px),
                int(source_height_px),
                float(rendered_width_mm),
                float(rendered_height_mm),
                int(output_width_px),
                int(output_height_px),
                output_file_relpath,
                output_file_sha256,
                description_text,
                description_model,
                now_utc_iso() if description_text else None,
                json.dumps(bbox_json, ensure_ascii=True),
                json.dumps(metadata_json, ensure_ascii=True, sort_keys=True),
                now_utc_iso(),
            ),
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
) -> tuple[int, int]:
    pdf_path = (archive_dir / resource.archived_relpath).resolve()
    if not pdf_path.exists():
        print(f"  missing archive file: {pdf_path}", flush=True)
        return 0, 0
    extraction_run_id = latest_extraction_run_id(db_path, resource.resource_id)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        print(f"  could not open PDF ({type(exc).__name__}: {exc}); skipping.", flush=True)
        return 0, 0
    saved = 0
    described = 0
    try:
        page_count = doc.page_count
        print(f"  pages: {page_count}", flush=True)
        for page_index, page in enumerate(doc):
            images = page.get_images(full=True)
            if not images:
                continue
            print(f"  page {page_index + 1}/{page_count}: {len(images)} image(s)", flush=True)
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
                    image_counter += 1
                    rendered_width_mm = float(rect.width) * 25.4 / 72.0
                    rendered_height_mm = float(rect.height) * 25.4 / 72.0
                    output_width_px = max(1, int(round(rect.width)))
                    output_height_px = max(1, int(round(rect.height)))
                    resample_lanczos = getattr(getattr(image_cls, "Resampling", image_cls), "LANCZOS", image_cls.LANCZOS)
                    resized = pil_image.resize((output_width_px, output_height_px), resample_lanczos)

                    filename = f"{uuid.uuid4()}.avif"
                    output_path = image_archive_dir / filename
                    print(f"    saving image {image_counter} -> {filename}", flush=True)
                    save_as_avif(resized, output_path, avif_quality)
                    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
                    relpath = str(output_path.relative_to(image_archive_dir.parent))

                    description = describe_with_moondream(output_path, moondream_model) if describe_images else None
                    if description:
                        described += 1
                    insert_resource_image(
                        db_path=db_path,
                        resource_id=resource.resource_id,
                        extraction_run_id=extraction_run_id,
                        page_index=page_index,
                        image_index=image_counter,
                        source_xref=xref,
                        source_name=source_name,
                        source_format=source_format,
                        source_width_px=max(0, source_width),
                        source_height_px=max(0, source_height),
                        rendered_width_mm=rendered_width_mm,
                        rendered_height_mm=rendered_height_mm,
                        output_width_px=output_width_px,
                        output_height_px=output_height_px,
                        output_file_relpath=relpath,
                        output_file_sha256=sha256,
                        description_text=description,
                        description_model=moondream_model if description else None,
                        bbox_json={
                            "x0": float(rect.x0),
                            "y0": float(rect.y0),
                            "x1": float(rect.x1),
                            "y1": float(rect.y1),
                        },
                        metadata_json={
                            "resource_original_filename": resource.original_filename,
                            "page_number": page_index + 1,
                            "source_extracted_ext": str(extracted.get("ext") or ""),
                            "source_colorspace": str(extracted.get("colorspace") or ""),
                            "source_bpc": extracted.get("bpc"),
                            "source_size_bytes": int(extracted.get("size") or 0),
                            "target_dpi": 72,
                            "compression": "lossy_avif",
                            "quality_preference": "small_size",
                        },
                    )
                    saved += 1
    finally:
        doc.close()
    return saved, described


def run(args: argparse.Namespace) -> int:
    fitz, image_cls = require_optional_dependencies()
    paths = load_paths(Path(args.project_root))
    schema_path = Path(__file__).parent.parent / "src" / "stemmacodicum" / "infrastructure" / "db" / "schema.sql"
    initialize_schema(paths.db_path, schema_path)
    resources = load_pdf_resources(paths.db_path, only_missing=not args.force)
    if not resources:
        print("No PDF resources need image extraction.")
        return 0
    image_archive_dir = ensure_image_archive_dir(paths)
    total_saved = 0
    total_described = 0

    print(f"Processing {len(resources)} PDF resources...", flush=True)
    for idx, resource in enumerate(resources, start=1):
        print(f"[{idx}/{len(resources)}] {resource.resource_id} {resource.original_filename}", flush=True)
        saved, described = process_resource(
            db_path=paths.db_path,
            archive_dir=paths.archive_dir,
            image_archive_dir=image_archive_dir,
            resource=resource,
            avif_quality=args.avif_quality,
            describe_images=args.describe,
            moondream_model=args.moondream_model,
            fitz=fitz,
            image_cls=image_cls,
        )
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
    parser.add_argument("--avif-quality", type=int, default=28, help="AVIF quality (lower is smaller)")
    parser.add_argument(
        "--describe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate image descriptions using Ollama moondream",
    )
    parser.add_argument("--moondream-model", default="moondream:latest", help="Ollama model name for image descriptions")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
