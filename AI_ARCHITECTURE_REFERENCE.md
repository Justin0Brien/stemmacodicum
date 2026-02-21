# Stemma Codicum Architecture Reference (AI-Oriented)

## 1. Purpose of This Document
This is a condensed, implementation-accurate architecture map of the current Stemma Codicum codebase. It is designed so an AI (or engineer) can reason about impact, extension points, and cross-layer behavior without first reading every source file.

Scope source:
- File tree snapshot: `/Users/justin/Downloads/temp_tree.txt`
- Codebase: `/Users/justin/code/stemmacodicum`

---

## 2. High-Level Architecture (DDD/Clean-Layer Mapping)

### 2.1 Layer responsibilities

| Layer | Code location | Responsibility | What it must not do |
|---|---|---|---|
| Domain | `src/stemmacodicum/domain` | Pure business data model (`dataclass` entities for resources, claims, evidence, extraction, vector, verification, citations/references) | No DB/network/framework calls |
| Application | `src/stemmacodicum/application` | Use-case orchestration (ingest, extract, bind, verify, report, trace, CEAPF, pipeline, vector indexing, health, background queue) | No HTTP/CLI parsing; avoid UI concerns |
| Infrastructure | `src/stemmacodicum/infrastructure` | Concrete adapters: SQLite repos/schema/runtime, archive store, parser adapters, importers, vector store (Qdrant), server runtime | No command-line or HTTP routing |
| Core utility | `src/stemmacodicum/core` | Cross-cutting primitives (paths/config, IDs, hashing, timestamps, logging, errors, title derivation, file ops) | No business workflow orchestration |
| Interface/Presentation | `src/stemmacodicum/cli`, `src/stemmacodicum/web` | User-facing entrypoints (CLI commands and FastAPI routes + static web GUI) | Should not hold core business rules |

### 2.2 Dependency direction
Stable direction in practice:
- `cli/web -> application -> infrastructure/domain/core`
- `infrastructure -> domain/core`
- `domain -> (none)`

Notes:
- `interfaces/protocols` exists but is currently placeholder-only (future inversion seam).
- Domain rules modules are placeholders; validation logic is mostly in application services today.

---

## 3. Runtime Composition and Entrypoints

### 3.1 CLI
- Entry: `src/stemmacodicum/__main__.py` -> `src/stemmacodicum/cli/main.py`
- Command groups: `init`, `ingest`, `resources`, `refs`, `extract`, `claims`, `bind`, `verify`, `report`, `trace`, `doctor`, `ceapf`, `pipeline`, `vector`, `web`
- Pattern: each command builds concrete repos/services directly and executes one use case.

### 3.2 Web app
- Entry command: `stemma web` (`src/stemmacodicum/cli/commands/web_cmd.py`)
- App factory: `src/stemmacodicum/web/app.py:create_app`
- FastAPI app initializes project and wires service factories (repos + services + vector service cache).
- Static GUI: `src/stemmacodicum/web/static/index.html` (single-page app with vanilla JS calling `/api/...`).

### 3.3 Project state layout
From `src/stemmacodicum/core/config.py`:
- `.stemma/stemma.db` (SQLite)
- `.stemma/archive/` (immutable files)
- `.stemma/vector/qdrant/` (local Qdrant data)
- `.stemma/vector/qdrant_url.txt` (optional server-mode switch)

`STEMMA_HOME` can relocate `.stemma`.

---

## 4. Deep Dive: Infrastructure Data Plane

## 4.1 SQLite integration
Primary files:
- Runtime: `src/stemmacodicum/infrastructure/db/sqlite.py`
- Schema: `src/stemmacodicum/infrastructure/db/schema.sql`
- Repositories: `src/stemmacodicum/infrastructure/db/repos/*.py`

Runtime behavior:
- `PRAGMA foreign_keys = ON`
- `PRAGMA journal_mode = WAL`
- `PRAGMA synchronous = NORMAL`
- `PRAGMA busy_timeout = 30000` (default; env-overridable)

Schema characteristics:
- 27 tables total, including:
  - Core provenance: `resources`, `resource_digests`, `citations`, `reference_entries`, `reference_resources`
  - Extraction/standoff: `extraction_runs`, `extracted_tables`, `document_texts`, `text_segments`, `text_annotations`, `text_annotation_spans`, `text_annotation_relations`
  - Claims/evidence/verification: `claim_sets`, `claims`, `evidence_items`, `evidence_selectors`, `claim_evidence_bindings`, `verification_runs`, `verification_results`
  - Argument graph (CEAPF): `propositions`, `assertion_events`, `argument_relations`
  - Vector metadata: `vector_index_runs`, `vector_chunks`
  - Background/web extras: `import_jobs`, `resource_images`
- 24 delete-blocking triggers enforce append-only posture (DELETE aborted across critical tables).
- Most FKs are `ON DELETE RESTRICT`.

Migration behavior:
- Lightweight column migration in `sqlite.py` for resource metadata columns.
- `BackgroundImportQueueService` performs additional `import_jobs` column backfills (`cancel_*`) if absent.

Repository model:
- Each aggregate has a concrete repo class (ResourceRepo, ClaimRepo, ExtractionRepo, VectorRepo, etc.)
- Services orchestrate repos; repos perform SQL mapping only.

## 4.2 Qdrant integration
Primary files:
- Store adapter: `src/stemmacodicum/infrastructure/vector/qdrant_store.py`
- Docker/server orchestration: `src/stemmacodicum/infrastructure/vector/server_runtime.py`
- Chunking: `src/stemmacodicum/infrastructure/vector/chunking.py`
- Embeddings: `src/stemmacodicum/infrastructure/vector/embeddings.py`
- App service: `src/stemmacodicum/application/services/vector_service.py`

Modes:
- Local embedded mode: default, storage under `.stemma/vector/qdrant`
- Server mode: active when URL provided by:
  1. explicit argument, else
  2. `STEMMA_QDRANT_URL`, else
  3. `.stemma/vector/qdrant_url.txt`

Server runtime behavior (local URLs only):
- Can auto-start Docker Desktop on macOS and run `qdrant/qdrant:v1.16.2` container.
- Health-check via `/healthz`.
- If server init fails and fallback enabled, switches to local embedded Qdrant.
- If local storage locked by another process, can fall back to isolated per-process path.

SQLite/Qdrant split:
- Qdrant stores vectors + payload for search.
- SQLite stores vector run/chunk metadata (`vector_index_runs`, `vector_chunks`) for auditability and status.
- Search path: embed query -> Qdrant search -> return payload-backed hits.

---

## 5. End-to-End Information Flow

## 5.1 Core provenance flow
1. `IngestionService` hashes file, dedupes by `resources.digest_sha256`, archives immutable file via `ArchiveStore`, persists `resources`.
2. `ExtractionService` parses resource (Docling for PDF when available; format fallbacks otherwise), persists run/tables/text/segments/annotations in SQLite.
3. If vector service attached, extraction triggers `VectorIndexingService.index_extraction`.
4. Claims imported via `ClaimService`; evidence linked via `EvidenceBindingService`.
5. `VerificationService` validates bindings and checks narrative/quantitative rules against extracted data.
6. `ReportingService` exports run summaries (JSON/Markdown).
7. `TraceService` provides reverse/forward provenance traversal by claim/resource/citation.

## 5.2 Business logic to CLI
Call chain shape:
- `cli/main.py` parses args -> selected `cli/commands/*` handler
- handler builds repos/services
- service executes use case
- handler renders Rich tables/panels

Important nuance:
- `stemma extract run` and pipeline paths wire vector indexing by default.
- `stemma ingest` only ingests (no extraction).

## 5.3 Business logic to Web GUI
Call chain shape:
- Browser (`web/static/index.html`) uses `fetch` to `/api/*`
- FastAPI route in `web/app.py` validates input and delegates to same application services used by CLI
- JSON response returned; GUI renders panels/cards

Important nuance:
- Web ingest endpoints (`/api/ingest/path`, `/api/ingest/upload`) call `maybe_extract_after_import`, so web ingest usually attempts extraction (and vector indexing) immediately.
- Streaming ingest endpoints provide SSE progress events.
- Background queue (`BackgroundImportQueueService`) persists import jobs in SQLite and survives app reopen.

---

## 6. Directory Narrative (Module-by-Module)

## 6.1 Repository root
- `README.md`: quick start and operational notes.
- `DOCUMENTATION.md`: long-form product/architecture reference.
- `pyproject.toml`: package metadata, dependencies, entrypoint (`stemma`).
- `scripts/`: operational utilities for URL recovery, PDF image extraction, title generation, icon generation.
- `docs/screenshots/`: tutorial screenshot placeholders.
- Design docs/plans: `CEBF.md`, `Claim–Evidence–Argument Provenance Format (CEAPF) v0.1.md`, `WEB_APP_TUTORIAL.md`, etc.

## 6.2 `src/stemmacodicum/core`
- `config.py`: project path resolution (`AppPaths`, `load_paths`).
- `errors.py`: typed app exceptions.
- `hashing.py`: byte/file digests.
- `ids.py`: UUID4 + deterministic UUID5.
- `citation.py`: deterministic 4-char cite ID allocation via SHA-256 + base62.
- `files.py`: directory creation, atomic copy, read-only flags.
- `time.py`: UTC ISO timestamps.
- `logging.py`: Rich logging config.
- `document_titles.py`: heuristic human-readable title derivation.

## 6.3 `src/stemmacodicum/domain`
Models:
- `resource.py`, `reference.py`, `citation.py`
- `claim.py`, `evidence.py`, `verification.py`
- `extraction.py` (runs, tables, text, segments, annotations)
- `vector.py` (index runs/chunks)
Rules:
- `rules/*.py` currently placeholders.

## 6.4 `src/stemmacodicum/application`
- `project_service.py`: initialize project dirs + schema.
- `ingestion_service.py`: ingest/dedupe/archive + source URL metadata.
- `reference_service.py`: BibTeX import/upsert, cite mapping, resource linking.
- `claim_service.py`: claim-set management + CSV/JSON/Markdown claim import.
- `evidence_binding_service.py`: evidence item/selector creation + role coverage checks.
- `extraction_service.py`: parse + persist standoff layers, deterministic table IDs, progress + subprocess/retry controls.
- `vector_service.py`: chunk/embed/index/search/backfill/status.
- `verification_service.py`: deterministic narrative/quantitative verification.
- `reporting_service.py`: verification summaries + report export.
- `trace_service.py`: claim/resource/citation traversal via SQL queries.
- `health_service.py`: integrity checks (DB pragmas, archive digests, selectors JSON, vector consistency).
- `pipeline_service.py`: resumable bulk import/extract pipeline.
- `ceapf_service.py`: proposition/assertion/relation operations.
- `background_import_queue_service.py`: persisted async import queue with cancel/skip controls.
- `dto/io_models.py`: placeholder.

## 6.5 `src/stemmacodicum/infrastructure`
- `archive/store.py`: content-addressed immutable archive layout.
- `db/sqlite.py`: connection policy + schema initialization + lightweight migration.
- `db/schema.sql`: full relational model + indexes + append-only triggers.
- `db/repos/*.py`: per-aggregate SQL mappers.
- `parsers/docling_adapter.py`: multi-format parsing, PDF Docling runtime tuning/fallback.
- `importers/`: BibTeX, CSV claims, JSON claims, Markdown bullet claims.
- `exporters/`: JSON/Markdown exporter placeholders.
- `resolvers/pdf_selector_resolver.py`: placeholder.
- `vector/`: Qdrant store, docker runtime helper, chunking, embeddings.

## 6.6 `src/stemmacodicum/cli`
- `main.py`: command parser and dispatch.
- `context.py`: shared CLI context (`AppPaths`, `Console`).
- `docling_options.py`: shared docling runtime flags.
- `commands/*.py`: feature commands; each composes services from repos.

## 6.7 `src/stemmacodicum/web`
- `app.py`: FastAPI app factory, dependency wiring, API routes, SSE ingest stream, viewer payload shaping, DB explorer APIs.
- `static/index.html`: single-page web GUI (HTML/CSS/vanilla JS) calling `/api/*`.
- `static/icons/*`, manifest/favicon assets.

## 6.8 `src/stemmacodicum/interfaces/protocols`
- Protocol files are placeholder stubs for future dependency inversion.

## 6.9 `tests/unit`
- 22 unit test files, 56 test functions, ~2716 LOC of tests.
- Coverage spans archive, importers, services, web API/GUI backend behavior, concurrency, vector runtime fallback logic.

---

## 7. Unit Test Suite Map

### 7.1 Coverage by subsystem
- Core utilities: hashing, citation IDs, title derivation.
- Infrastructure: archive sharding, BibTeX parser, Docling adapter behavior, SQLite concurrency pragmas/lock-wait semantics, vector runtime failovers.
- Application services: claims, references, evidence binding, extraction, verification, reporting, trace, health, CEAPF, pipeline, vector service.
- Web: broad end-to-end API smoke and targeted viewer/import queue/source-recovery tests.

### 7.2 Key confidence points
- Deterministic IDs and parsing behavior are explicitly tested.
- DB concurrency expectations (WAL + busy timeout + lock wait) are tested.
- Qdrant server/local fallback behavior is tested via monkeypatch/fakes.
- Web API parity and queue persistence are tested with FastAPI `TestClient`.

### 7.3 Execution note
Tests were not executed in this environment during this documentation step because `pytest` is not installed in the active interpreter (`python -m pytest` -> `No module named pytest`).

---

## 8. External Libraries and Runtime Requirements

From `pyproject.toml`:
- Core runtime: `rich`, `fastapi`, `uvicorn`, `python-multipart`
- Optional `vector`: `qdrant-client`, `sentence-transformers` (and transitively `torch` at runtime)
- Optional `dev`: `pytest`, `httpx`
- Optional `media`: `pymupdf`, `pillow`, `pillow-avif-plugin`, `ollama`, `playwright`

Also used in code paths (not declared in core dependencies):
- `docling` for PDF extraction (optional runtime dependency in parser path)
- Docker CLI for local Qdrant server auto-start features

---

## 9. Important Design Choices (Current System Contract)
1. Immutable/archive-first provenance:
- Content-addressed file storage + read-only archived files.

2. Append-only relational history:
- Delete-blocking triggers and restrictive FKs preserve audit trail.

3. Determinism where possible:
- Stable citation IDs, deterministic table IDs, digest-based extraction/vector identities.

4. Dual storage strategy for vectors:
- SQLite for audit metadata, Qdrant for similarity search vectors.

5. CLI-first, web-parity architecture:
- CLI and web call shared application services rather than duplicating business logic.

6. Progressive robustness for long-running operations:
- Extraction retry/subprocess controls, SSE progress, persisted background queue, resumable batch pipeline.

7. Intentional extension stubs:
- Protocol/rules/exporter/resolver placeholders indicate planned hardening and inversion boundaries.

---

## 10. Fast Change-Impact Guide (for AI edits)
- Change DB schema/constraints:
  - Edit `src/stemmacodicum/infrastructure/db/schema.sql`
  - Update migrations in `sqlite.py` and/or service-level backfills
  - Update repos + relevant tests (`test_web_app.py`, service tests)

- Change parsing/extraction behavior:
  - `infrastructure/parsers/docling_adapter.py` and/or `application/services/extraction_service.py`
  - Re-check tests: `test_docling_adapter.py`, `test_extraction_service.py`, `test_web_app.py`

- Change verification logic:
  - `application/services/verification_service.py`
  - Re-check `test_verification_service.py`, `test_reporting_service.py`, `test_trace_service.py`

- Change vector behavior (chunking/embedding/store/runtime):
  - `infrastructure/vector/*`, `application/services/vector_service.py`, optionally `cli/commands/vector_cmd.py`
  - Re-check `test_vector_chunking.py`, `test_vector_service.py`, `test_vector_runtime.py`, web smoke tests

- Change GUI/API behavior:
  - Backend: `web/app.py`
  - Frontend: `web/static/index.html`
  - Re-check `test_web_app.py`

