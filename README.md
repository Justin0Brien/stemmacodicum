# Stemma Codicum

CLI-first Python implementation for machine-verifiable claim-to-evidence workflows.

## Current Status

Initial foundation is implemented:

- Project bootstrap (`stemma init`)
- Immutable resource ingest (`stemma ingest`)
- Resource listing (`stemma resources`)
- Citation/reference import (`stemma refs import-bib`)
- Extraction runs with deterministic table IDs (`stemma extract run`)
- Claim set + claim import/list (`stemma claims ...`)
- Evidence binding + validation (`stemma bind ...`)
- Deterministic verification runs (`stemma verify ...`)
- Verification report export (`stemma report verification`)
- Trace navigation (`stemma trace ...`)
- Integrity checks (`stemma doctor`)
- CEAPF primitives (`stemma ceapf ...`)
- Resumable production batch pass (`stemma pipeline financial-pass`)
- Web GUI (`stemma web`)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# For local vector indexing + semantic search:
pip install -e ".[vector]"

stemma init
stemma ingest /path/to/report.pdf
stemma resources
stemma refs import-bib /path/to/references.bib
stemma refs citations
stemma refs list
stemma extract run --resource-digest <sha256>
stemma extract tables --resource-digest <sha256>
stemma claims import --file /path/to/claims.csv --format csv --claim-set report-2025
stemma claims sets
stemma claims list --claim-set report-2025
stemma bind add --claim-id <uuid> --resource-digest <sha256> --role value-cell --selectors-file selectors.json
stemma bind validate --claim-id <uuid>
stemma verify claim --claim-id <uuid>
stemma verify set --claim-set report-2025
stemma report verification --run-id <verification-run-uuid> --json-out out.json --md-out out.md
stemma trace claim --claim-id <uuid>
stemma trace resource --resource-digest <sha256>
stemma trace citation --cite-id <AB12>
stemma doctor
stemma ceapf add-proposition --json '{"subject":"org:X","predicate":"ceapf:asserts","object":"..."}'
stemma ceapf add-assertion --proposition-id <uuid> --agent person:AuthorX --modality asserts
stemma ceapf add-relation --type supports --from-type assertion_event --from-id <id> --to-type proposition --to-id <id>
stemma pipeline financial-pass --root /Volumes/X10/data/Institution
stemma vector status
stemma vector search --query "cash at bank"
stemma vector migrate-server --url http://127.0.0.1:6333 --drop-existing
stemma web --host 127.0.0.1 --port 8765
```

By default, project state is created under `.stemma/` in the current project root.

Docling PDF extraction auto-tunes CPU/GPU/batching by default and logs the effective settings per file. You can override this per command, for example:

```bash
stemma extract run --resource-digest <sha256> --docling-device mps --docling-threads 10 --docling-layout-batch-size 40
```

`stemma extract run` now shows live spinner/timing output, and `stemma pipeline financial-pass` shows live overall progress plus verbose per-document logs by default.

SQLite runs in WAL mode with lock wait timeouts, so you can run ingestion/extraction in one terminal while reading or editing data from another terminal or the web GUI.

Extraction now triggers vector indexing by default. By default this uses local Qdrant storage under `.stemma/vector/qdrant`.
For higher-performance/concurrent access, run a Qdrant server and migrate with:

```bash
docker run -d --name stemma-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.16.2
stemma vector migrate-server --url http://127.0.0.1:6333 --drop-existing
```

Migration writes `.stemma/vector/qdrant_url.txt`, which activates server mode for future CLI/web runs.
When server mode is active and the URL is local (`127.0.0.1`/`localhost`), Stemma auto-starts the Docker container if needed
(`stemma-qdrant`, restart policy `unless-stopped` by default).

The web GUI includes a per-card `?` help icon (single popup with Basic then Comprehensive guidance) and a Database Explorer card to inspect tables, schema, and sample rows.

## Helper Scripts

The repository now includes workflow scripts for reference URL recovery, PDF image extraction, and human-readable titles:

```bash
# Issue 1: recover missing external source URLs (xattr -> manifests -> web -> Playwright fallback)
python scripts/recover_reference_urls.py --project-root .

# Issue 4: extract embedded PDF images to AVIF archive + moondream descriptions
python scripts/extract_pdf_images_archive.py --project-root .

# Issue 5: generate and persist human-readable resource titles + candidates
python scripts/generate_readable_titles.py --project-root . --model qwen3:4b
```

Notes:
- `scripts/recover_reference_urls.py` supports browser escalation via Playwright CDP (`--cdp-url http://127.0.0.1:9222`).
- `scripts/extract_pdf_images_archive.py` requires `pymupdf`, `Pillow`, and either AVIF plugin support or `ffmpeg`.
- Title/image description scripts use the Ollama Python client when available.
