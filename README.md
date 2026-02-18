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
stemma web --host 127.0.0.1 --port 8765
```

By default, project state is created under `.stemma/` in the current project root.

Docling PDF extraction auto-tunes CPU/GPU/batching by default and logs the effective settings per file. You can override this per command, for example:

```bash
stemma extract run --resource-digest <sha256> --docling-device mps --docling-threads 10 --docling-layout-batch-size 40
```

`stemma extract run` now shows live spinner/timing output, and `stemma pipeline financial-pass` shows live overall progress plus verbose per-document logs by default.

SQLite runs in WAL mode with lock wait timeouts, so you can run ingestion/extraction in one terminal while reading or editing data from another terminal or the web GUI.

The web GUI includes a per-card `?` help icon (single popup with Basic then Comprehensive guidance) and a Database Explorer card to inspect tables, schema, and sample rows.
