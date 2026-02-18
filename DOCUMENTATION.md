# Stemma Codicum — Full Technical Documentation

> *Stemma codicum* (Latin): the family tree of manuscripts — the schematic produced by stemmatological recension in which all surviving manuscripts are mapped with their derivation to a single archetype. Used here as a metaphor for tracing any claim back through citations and evidence to its origin.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation & Configuration](#3-installation--configuration)
4. [Core Concepts](#4-core-concepts)
   - 4.1 [Resources](#41-resources)
   - 4.2 [Citations & References](#42-citations--references)
   - 4.3 [Extractions](#43-extractions)
   - 4.4 [Claims & Claim Sets](#44-claims--claim-sets)
   - 4.5 [Evidence Bindings & Selectors](#45-evidence-bindings--selectors)
   - 4.6 [Verification Runs](#46-verification-runs)
   - 4.7 [CEAPF — Claim–Evidence–Argument Provenance Format](#47-ceapf--claimevidence-argument-provenance-format)
5. [Database Schema](#5-database-schema)
6. [CLI Reference](#6-cli-reference)
   - 6.1 [stemma init](#61-stemma-init)
   - 6.2 [stemma ingest](#62-stemma-ingest)
   - 6.3 [stemma resources](#63-stemma-resources)
   - 6.4 [stemma refs](#64-stemma-refs)
   - 6.5 [stemma extract](#65-stemma-extract)
   - 6.6 [stemma claims](#66-stemma-claims)
   - 6.7 [stemma bind](#67-stemma-bind)
   - 6.8 [stemma verify](#68-stemma-verify)
   - 6.9 [stemma report](#69-stemma-report)
   - 6.10 [stemma trace](#610-stemma-trace)
   - 6.11 [stemma doctor](#611-stemma-doctor)
   - 6.12 [stemma ceapf](#612-stemma-ceapf)
   - 6.13 [stemma pipeline](#613-stemma-pipeline)
   - 6.14 [stemma web](#614-stemma-web)
7. [REST API Reference](#7-rest-api-reference)
8. [Claim Import Formats](#8-claim-import-formats)
9. [Selector Types Reference](#9-selector-types-reference)
10. [Verification Logic Detail](#10-verification-logic-detail)
11. [The Immutable Archive](#11-the-immutable-archive)
12. [Citation ID Allocation](#12-citation-id-allocation)
13. [The Financial Pipeline](#13-the-financial-pipeline)
14. [CEAPF Conceptual Model](#14-ceapf-conceptual-model)
15. [Testing](#15-testing)
16. [End-to-End Worked Example](#16-end-to-end-worked-example)

---

## 1. Project Overview

Stemma Codicum is a **CLI-first, machine-verifiable claim-to-evidence workflow toolkit**. Its purpose is to provide an auditable, deterministic chain linking any factual claim in a report or document back through citations and source material to the specific cell, row, paragraph or quoted text in the original source file from which the claim derives.

The motivating use case is fact-checking in research, journalism, and financial analysis. For example: *"Institution X has £5,631,000 cash at bank as of 31 July 2025"* — Stemma Codicum lets you:

- Store a SHA-256-immutable copy of the source PDF in an on-disk archive.
- Extract all tables from that PDF using the IBM Docling parser.
- Record the precise claim with its value, currency, and period.
- Bind the claim to the exact table cell (identified by table ID, row index, column index) along with contextual spans for the row header, column header, caption, and currency/scale cues.
- Run a deterministic verification pass that re-reads the extracted table data and confirms the cell value matches the claimed value.
- Produce JSON and Markdown verification reports.
- Navigate the entire provenance chain in any direction via the trace commands.

Beyond numeric verification the system also supports:

- **Narrative claims** — verified by confirming that a quoted text fragment (`TextQuoteSelector`) appears in the source document.
- **CEAPF** — a higher-level argument graph layered on top, representing propositions, assertion events, and argument relations (supports, rebuts, undercuts, qualifies).
- **Batch financial pipeline** — a resumable, keyword-filtered pass over an entire directory tree that ingests and extracts every financial document it finds.
- **Web GUI** — a FastAPI-backed web interface exposing every operation via REST, with a built-in HTML frontend.

---

## 2. Architecture Overview

The project follows a layered clean-architecture pattern:

```
stemmacodicum/
├── cli/            # Argument parsing, command handlers — presentation layer
├── web/            # FastAPI app — REST presentation layer
├── application/
│   ├── services/   # Business logic orchestration
│   └── dto/        # Input/output data transfer objects
├── domain/
│   ├── models/     # Pure data structures (dataclasses, no DB logic)
│   └── rules/      # Domain validation rules (stubs ready for extension)
├── infrastructure/
│   ├── db/         # SQLite connection, schema, repository classes
│   ├── archive/    # Immutable content-addressed file store
│   ├── parsers/    # Document parsers (Docling adapter)
│   └── importers/  # File format importers (BibTeX, CSV, JSON, Markdown)
└── core/           # Pure utilities: hashing, IDs, time, config, errors
```

**Data flow summary:**

```
Source file
   │ ingest
   ▼
ArchiveStore (immutable, SHA-256 sharded)
ResourceRepo (SQLite)
   │ extract
   ▼
ExtractionRepo (tables, runs in SQLite)
   │ bind
   ▼
EvidenceRepo (items + selectors in SQLite)
ClaimRepo (claims + claim sets in SQLite)
   │ verify
   ▼
VerificationRepo (runs + results in SQLite)
   │ report / trace
   ▼
Markdown / JSON output
```

All identifiers are UUID4 strings. The SQLite database enforces `ON DELETE RESTRICT` on all foreign keys, making the store append-only in practice.

---

## 3. Installation & Configuration

### Requirements

- Python ≥ 3.11
- Optional: IBM Docling (`docling` package) for PDF table extraction. Without it the Markdown/plain-text parser is used.

### Install

```bash
git clone <repo>
cd stemmacodicum
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development/testing:

```bash
pip install -e ".[dev]"
pytest
```

### Project data directory

By default Stemma Codicum creates a `.stemma/` directory inside whatever directory you run `stemma` from (the *project root*). This contains:

| Path | Purpose |
|---|---|
| `.stemma/stemma.db` | SQLite database holding all metadata |
| `.stemma/archive/` | Immutable content-addressed file archive |
| `.stemma/financial_pass_state.json` | Resumable pipeline state |
| `.stemma/financial_pass.log.jsonl` | Pipeline structured log |

### Environment variable override

Set `STEMMA_HOME` to an absolute path to redirect the `.stemma/` directory to a different location:

```bash
export STEMMA_HOME=/Volumes/X10/data/stemma
stemma init
```

### Global flag

All commands accept `--project-root <path>` to override the project root without changing directory:

```bash
stemma --project-root /path/to/project ingest report.pdf
```

### Verbosity

`-v` / `-vv` increases log output to DEBUG level.

---

## 4. Core Concepts

### 4.1 Resources

A **resource** is any source file (PDF, Word, spreadsheet, Markdown, etc.) that has been ingested into the archive. Each resource is identified primarily by its **SHA-256 digest** — this is a content address, meaning two files with identical bytes share a record (deduplication is automatic).

Key resource fields:

| Field | Description |
|---|---|
| `id` | UUID4 primary key |
| `digest_sha256` | Hex SHA-256 of the original file bytes |
| `media_type` | MIME type (auto-detected from file extension) |
| `original_filename` | Original file name at ingest time |
| `source_uri` | Optional URI recording where the file came from |
| `archived_relpath` | Relative path within `.stemma/archive/` |
| `size_bytes` | File size |
| `ingested_at` | UTC ISO-8601 timestamp |

Files are stored immutably in the archive at a sharded path derived from the digest: `sha256/<ab>/<cd>/<full-digest><extension>`. After writing, the file is made read-only at the OS level.

### 4.2 Citations & References

**Citations** are short 4-character identifiers (e.g. `Ab3X`) that map to bibliographic references. They are derived deterministically from the BibTeX key using a SHA-256-based Base62 encoding. If two keys produce a collision a retry counter is incremented (up to 5000 attempts).

**References** are full bibliographic records (BibTeX entries) and are linked to citations in a 1:1 relationship. A reference can then be linked to one or more resources via the `reference_resources` join table.

This forms the citation chain:

```
cite_id (4-char) → reference_entry → resource (PDF/file in archive)
```

### 4.3 Extractions

An **extraction run** records the result of parsing a resource file to extract its structured content. The current parser is the **Docling adapter** (using IBM Docling for PDFs; a built-in Markdown table parser acts as a fallback).

Each extraction run captures:

- Which resource was parsed
- Parser name, version, and a digest of the parser configuration
- A digest of the output payload (for reproducibility checks)
- The status (`success` / `failed`)

Each run produces zero or more **extracted tables**. Every table gets a **deterministic table ID** computed by hashing its canonical form: caption, page index, row headers, column headers, and bounding box. This means re-running extraction on the same file produces the same table IDs, enabling stable references to specific tables across re-extractions.

Table fields include `row_headers_json`, `col_headers_json`, `cells_json` (array of `{row_index, col_index, value}` objects), `bbox_json` (bounding box in page coordinates), and `page_index`.

### 4.4 Claims & Claim Sets

A **claim set** is a named container for a batch of related claims (e.g. "annual-report-2025", "q3-verification").

A **claim** is an individual assertion of fact. Two claim types are supported:

#### Quantitative claims

Assert that a specific measurable value exists in a source document.

Required fields:
- `claim_type`: `"quantitative"`
- `subject`: what the value is about (e.g. `"Institution X"`)
- `predicate`: what the subject's property is (e.g. `"cash_at_bank"`)
- `value_raw` and/or `value_parsed`: the expected numeric value

Optional fields:
- `currency`: e.g. `"GBP"`, `"USD"`
- `scale_factor`: e.g. `1000` if source values are in thousands
- `period_label`: e.g. `"FY2024/25"`
- `source_cite_id`: 4-char citation ID to link the claim to a reference

#### Narrative claims

Assert that a specific quoted passage appears in a source document.

Required fields:
- `claim_type`: `"narrative"`
- `narrative_text`: the exact (or sufficiently distinctive) text being claimed

Optional fields: `subject`, `predicate`, `object_text`, `source_cite_id`

> **Auto-detection**: if `claim_type` is omitted, claims with a `narrative_text` value are inferred as narrative; all others are inferred as quantitative.

### 4.5 Evidence Bindings & Selectors

**Evidence** is the mechanism by which a claim is linked to a specific location within a specific resource. Each evidence item has:

- A **role** — what contextual function it serves for the claim
- A **resource** — which archived file it is found in
- An optional **page index**
- One or more **selectors** — each identifying the evidence location using a different addressing scheme

#### Evidence roles

For **quantitative claims**, four roles are required:

| Role | Purpose |
|---|---|
| `value-cell` | The cell containing the actual value |
| `row-header` | The row header that names the metric |
| `column-header` | The column header (often a period/date) |
| `caption` | The table caption providing context |

For **narrative claims**, one role is required:

| Role | Purpose |
|---|---|
| `quote` | The text passage being cited |

#### Validation rule

Each evidence item must carry **at least 2 distinct selector types**. This redundancy requirement ensures that evidence is not brittle — if one addressing scheme becomes stale (e.g. a bounding box shifts after PDF re-render), another selector can still locate the fragment.

### 4.6 Verification Runs

A **verification run** is a recorded execution of the automated verification logic against one claim or an entire claim set. The run captures the **policy profile** used (currently `"strict"` is the only operational profile) and produces a **verification result** for each claim evaluated.

Each result carries:
- `status`: `"pass"` or `"fail"`
- `diagnostics_json`: a structured explanation of the pass/fail decision

Results are stored permanently so that historical verification runs can be compared and reported on.

### 4.7 CEAPF — Claim–Evidence–Argument Provenance Format

CEAPF is the higher-level argument graph layer. It is inspired by the W3C Web Annotation model, PROV-O, and argumentation theory (Dung's abstract argumentation framework).

It models three distinct object types:

| Type | What it represents |
|---|---|
| **Proposition** | The content-level statement about the world (a JSON object with `subject`, `predicate`, `object` and optional qualifiers) |
| **Assertion Event** | The act of an agent asserting a proposition with a specific modality, optionally linked to an evidence item |
| **Argument Relation** | A directed relationship connecting two nodes in the argument graph |

Assertion modalities: `asserts`, `denies`, `speculates`, `predicts`, `recommends`.

Argument relation types: `supports`, `rebuts`, `undercuts`, `qualifies`.

This separation is the key architectural insight of CEAPF: a source document can be machine-verified to contain an assertion event (the quote exists, the attribution metadata matches) even when the truth of the proposition itself cannot be determined mechanically.

---

## 5. Database Schema

The SQLite database at `.stemma/stemma.db` contains the following tables. All foreign keys are `ON DELETE RESTRICT` — records cannot be deleted if referenced by other records.

**Concurrency model (multi-process safe):**
- Connections use SQLite WAL mode (`PRAGMA journal_mode=WAL`) so readers (CLI/web) can continue while a writer is active.
- Connections set a busy timeout to wait on transient write locks instead of failing immediately.
- Foreign keys remain enforced on every connection.

Environment variables for tuning lock behavior:
- `STEMMA_SQLITE_CONNECT_TIMEOUT_SECONDS` (default: `30`)
- `STEMMA_SQLITE_BUSY_TIMEOUT_MS` (default: `30000`)

| Table | Purpose |
|---|---|
| `resources` | One row per ingested file |
| `resource_digests` | Additional hash algorithms beyond SHA-256 |
| `provenance_events` | Append-only event log for provenance |
| `citations` | 4-char cite ID ↔ normalised BibTeX key |
| `reference_entries` | Full bibliographic records |
| `reference_resources` | Many-to-many: references ↔ resources |
| `extraction_runs` | One row per extraction execution |
| `extracted_tables` | One row per table found in an extraction run |
| `claim_sets` | Named containers for claim batches |
| `claims` | Individual claim records |
| `evidence_items` | Evidence items linked to claims |
| `evidence_selectors` | Selectors for each evidence item |
| `claim_evidence_bindings` | Many-to-many: claims ↔ evidence items |
| `verification_runs` | One row per verification execution |
| `verification_results` | Per-claim results within a run |
| `propositions` | CEAPF proposition objects (JSON payload) |
| `assertion_events` | CEAPF assertion events |
| `argument_relations` | CEAPF argument graph edges |

---

## 6. CLI Reference

All commands follow the pattern:

```bash
stemma [--project-root <path>] [-v] <command> [subcommand] [options]
```

### 6.1 `stemma init`

Initialises the project database and creates the `.stemma/` directory structure.

```bash
stemma init
```

Safe to run multiple times — idempotent. Required before any other command.

**Output example:**
```
Created /path/to/.stemma
Created /path/to/.stemma/archive
Database ready /path/to/.stemma/stemma.db
```

---

### 6.2 `stemma ingest`

Copies one or more source files into the immutable archive, computing their SHA-256 digest.

```bash
stemma ingest <path> [<path> ...] [--source-uri <uri>]
```

| Argument | Description |
|---|---|
| `paths` | One or more local file paths to ingest |
| `--source-uri` | Optional URI metadata applied to all files (useful for recording download location) |

Deduplication is automatic: if the file's SHA-256 already exists in the archive, the status is `duplicate` and no data is modified.

**Output:** A table with file path, status (`ingested`/`duplicate`/`error`), and SHA-256 digest.

**Example:**
```bash
stemma ingest /downloads/university-accounts-2025.pdf --source-uri "https://example.ac.uk/accounts.pdf"
```

---

### 6.3 `stemma resources`

Lists all ingested resources.

```bash
stemma resources [--limit <n>]
```

| Argument | Description |
|---|---|
| `--limit` | Maximum rows to show (default: 100) |

**Output:** Table of resource ID, digest (truncated), filename, media type, size, ingestion timestamp.

---

### 6.4 `stemma refs`

Manages citations and bibliographic references.

#### `stemma refs import-bib`

Parses a BibTeX `.bib` file and imports all entries. For each entry:
1. The cite key is normalised (lower-cased, stripped).
2. A 4-char `cite_id` is allocated (deterministically derived from the normalised key).
3. The full reference is inserted or updated in `reference_entries`.

```bash
stemma refs import-bib <path-to-.bib>
```

**Output:** Summary panel showing entries seen, mappings created, references inserted, references updated.

#### `stemma refs list`

Lists all reference entries.

```bash
stemma refs list [--limit <n>]
```

Shows cite ID, entry type, year, title, DOI, URL.

#### `stemma refs citations`

Lists all cite ID mappings (4-char ID ↔ original BibTeX key).

```bash
stemma refs citations [--limit <n>]
```

#### `stemma refs link-resource`

Explicitly links a citation to an ingested resource by digest. This records that a particular reference *is* a particular archived file.

```bash
stemma refs link-resource --cite-id <AB12> --resource-digest <sha256>
```

---

### 6.5 `stemma extract`

Runs document parsing and manages extracted tables.

#### `stemma extract run`

Parses a resource and extracts all tables from it. The resource can be identified by ID or digest.

```bash
stemma extract run --resource-digest <sha256> [--profile <name>]
stemma extract run --resource-id <uuid> [--profile <name>]
```

| Argument | Description |
|---|---|
| `--resource-digest` | SHA-256 hex string of the resource |
| `--resource-id` | UUID of the resource |
| `--profile` | Parser profile name (default: `"default"`) |
| `--docling-auto-tune` / `--no-docling-auto-tune` | Enable/disable hardware auto-tuning for docling PDF extraction (default: enabled) |
| `--docling-use-threaded-pipeline` / `--no-docling-use-threaded-pipeline` | Force threaded vs standard docling PDF pipeline (default: auto) |
| `--docling-device` | Override inference device (`auto`, `cpu`, `mps`, `xpu`, `cuda`, `cuda:N`) |
| `--docling-threads` | Override CPU thread count used by docling |
| `--docling-layout-batch-size` | Override layout model batch size |
| `--docling-ocr-batch-size` | Override OCR batch size |
| `--docling-table-batch-size` | Override table extraction batch size |
| `--docling-queue-max-size` | Override threaded pipeline queue size |
| `--docling-log-settings` / `--no-docling-log-settings` | Enable/disable terminal logging of effective docling runtime settings (default: enabled) |

**Output:**
- Live spinner while parsing.
- Summary panel with run ID, resource ID, table count, parser version, parse duration, pages, pages/sec, and top timing buckets (if provided by docling).

**Parser behaviour:**
- PDFs: uses IBM Docling if installed.
- Markdown / plain text: uses the built-in Markdown table parser.
- The extracted tables receive deterministic IDs derived from a hash of their structure (caption + page + headers + bbox), so the same table always gets the same ID across re-runs.
- For PDFs, the adapter auto-detects system resources (CPU cores, RAM, platform) and applies tuned docling runtime settings, then logs the effective settings to the terminal before conversion.

#### `stemma extract tables`

Lists extracted tables for a given resource.

```bash
stemma extract tables --resource-digest <sha256> [--limit <n>]
```

**Output:** Table showing table ID, page index, caption, row count, column count.

---

### 6.6 `stemma claims`

Manages claim sets and individual claims.

#### `stemma claims import`

Imports claims from a file into a claim set. The claim set is created if it does not already exist; if it does, new claims are appended to it.

```bash
stemma claims import --file <path> --format <csv|json|md|markdown> --claim-set <name> [--description <text>]
```

| Argument | Description |
|---|---|
| `--file` | Path to the claims file |
| `--format` | File format: `csv`, `json`, `md`, or `markdown` |
| `--claim-set` | Claim set name (created if needed) |
| `--description` | Optional description for a new claim set |

See [Claim Import Formats](#8-claim-import-formats) for file format details.

#### `stemma claims list`

Lists claims, optionally filtered to a specific claim set.

```bash
stemma claims list [--claim-set <name>] [--limit <n>]
```

**Output:** Table showing ID, type, subject, predicate, narrative text, value, and period.

#### `stemma claims sets`

Lists all claim sets.

```bash
stemma claims sets [--limit <n>]
```

---

### 6.7 `stemma bind`

Binds claims to evidence items, and validates evidence completeness.

#### `stemma bind add`

Adds one evidence item to a claim, with one or more selectors describing where in the resource the evidence is located.

```bash
stemma bind add \
  --claim-id <uuid> \
  --resource-digest <sha256> \
  --role <role> \
  --selectors-file selectors.json \
  [--page-index <n>] \
  [--note "free text"]
```

| Argument | Description |
|---|---|
| `--claim-id` | UUID of the claim being bound |
| `--resource-id` / `--resource-digest` | Resource to bind to (mutually exclusive) |
| `--role` | Evidence role (`value-cell`, `row-header`, `column-header`, `caption`, `quote`) |
| `--selectors-file` | Path to a JSON file containing an array of selector objects |
| `--selectors-json` | JSON selector array as a literal string (alternative to `--selectors-file`) |
| `--page-index` | Optional page number (0-based) |
| `--note` | Optional free-text note |

The selectors JSON file must be an array of objects. Each object must have a `type` field. See [Selector Types Reference](#9-selector-types-reference).

**Example `selectors.json`:**
```json
[
  {
    "type": "PageGeometrySelector",
    "pageIndex": 36,
    "boxes": [{"x0": 92.1, "y0": 301.4, "x1": 140.6, "y1": 315.2}]
  },
  {
    "type": "TableAddressSelector",
    "table_id": "sha256:abcdef...",
    "cell_ref": {"row_index": 7, "col_index": 2},
    "units": {"currency": "GBP", "scale_factor": 1000},
    "period": {"label": "FY2024/25"}
  }
]
```

#### `stemma bind validate`

Validates that a claim has all the required evidence roles and that each evidence item has the required minimum of 2 distinct selector types.

```bash
stemma bind validate --claim-id <uuid>
```

**Output:** Panel showing PASS/FAIL, missing roles, and any evidence items with insufficient selector diversity.

---

### 6.8 `stemma verify`

Runs deterministic verification of claims against their bound evidence.

#### `stemma verify claim`

Verifies a single claim.

```bash
stemma verify claim --claim-id <uuid> [--policy <profile>]
```

Returns exit code `0` if the claim passes, `1` if it fails.

**Output:** Panel showing Run ID, Claim ID, status, and diagnostic JSON.

#### `stemma verify set`

Verifies all claims in a named claim set in a single run.

```bash
stemma verify set --claim-set <name> [--policy <profile>]
```

Returns exit code `0` if all claims pass, `1` if any fail.

**Output:** Panel showing Run ID, total, passed, and failed counts.

**Policy profiles:** Currently `"strict"` (default). The strict profile requires all binding validation checks to pass before proceeding to value-matching logic.

---

### 6.9 `stemma report`

Produces reports from verification run data.

#### `stemma report verification`

Displays and optionally exports a verification run summary.

```bash
stemma report verification \
  --run-id <uuid> \
  [--json-out output.json] \
  [--md-out output.md]
```

| Argument | Description |
|---|---|
| `--run-id` | UUID of the verification run |
| `--json-out` | Write full JSON report to this path |
| `--md-out` | Write Markdown report to this path |

**Output:** Table of policy, created timestamp, total/passed/failed. If there are failures, a snapshot of the first failure's diagnostics is shown.

The exported Markdown report contains a table with one row per claim: claim ID, status, and failure reason.

---

### 6.10 `stemma trace`

Navigates the provenance chain in any direction.

#### `stemma trace claim`

Shows all evidence items bound to a claim, including each evidence item's resource ID, role, and selectors.

```bash
stemma trace claim --claim-id <uuid>
```

#### `stemma trace resource`

Given a resource, shows all bibliographic references it is linked to, and all claims that are backed by evidence from it.

```bash
stemma trace resource --resource-digest <sha256>
stemma trace resource --resource-id <uuid>
```

#### `stemma trace citation`

Given a 4-char cite ID, shows the full reference metadata, any resources linked to it, and any claims whose `source_cite_id` matches.

```bash
stemma trace citation --cite-id <AB12>
```

---

### 6.11 `stemma doctor`

Runs integrity and consistency checks across the entire database and archive.

```bash
stemma doctor
```

Four checks are run:

| Check | Level | Condition |
|---|---|---|
| `db_runtime` | error/warning | SQLite runtime pragmas are unsuitable for concurrency (for example, non-WAL mode, disabled `busy_timeout`, disabled `foreign_keys`) |
| `archive_integrity` | error | An archived file is missing from disk, or its on-disk SHA-256 no longer matches the recorded digest |
| `selector_json` | error | An evidence selector's JSON payload cannot be parsed |
| `quantitative_bindings` | warning | A quantitative claim has zero evidence bindings |

Doctor also prints a `Database Runtime` table showing current SQLite values (journal mode, busy timeout, foreign key enforcement, etc.).

Returns exit code `0` if no errors (warnings are allowed), `1` otherwise.

---

### 6.12 `stemma ceapf`

Manages the CEAPF higher-level argument graph.

#### `stemma ceapf add-proposition`

Creates a new proposition record from a JSON object. The object can include any fields — common fields are `subject`, `predicate`, `object`.

```bash
stemma ceapf add-proposition --json '{"subject":"org:X","predicate":"ceapf:spent","object":{"amount":3.4,"unit":"GBP"}}'
stemma ceapf add-proposition --json-file proposition.json
```

#### `stemma ceapf add-assertion`

Records that an agent has asserted a proposition with a given modality, optionally linking to an evidence item.

```bash
stemma ceapf add-assertion \
  --proposition-id <uuid> \
  --agent person:AuthorX \
  --modality asserts \
  [--evidence-id <uuid>]
```

Modalities: `asserts`, `denies`, `speculates`, `predicts`, `recommends`.

#### `stemma ceapf add-relation`

Adds a directed argument relation between two nodes.

```bash
stemma ceapf add-relation \
  --type supports \
  --from-type assertion_event \
  --from-id <uuid> \
  --to-type proposition \
  --to-id <uuid>
```

Relation types: `supports`, `rebuts`, `undercuts`, `qualifies`.

#### `stemma ceapf list-propositions`

Lists recently created propositions.

```bash
stemma ceapf list-propositions [--limit <n>]
```

---

### 6.13 `stemma pipeline`

#### `stemma pipeline financial-pass`

Recursively scans a root directory for documents that look like financial reports, ingests them, and runs table extraction on each one. The pass is resumable: a state file records which paths have already been processed, so re-running continues from where it left off.

```bash
stemma pipeline financial-pass \
  --root /Volumes/X10/data/Institution \
  [--max-files <n>] \
  [--skip-extraction] \
  [--state-file <path>] \
  [--log-file <path>] \
  [--extract-timeout-seconds <n>] \
  [--verbose-docs|--no-verbose-docs] \
  [--docling-auto-tune|--no-docling-auto-tune] \
  [--docling-use-threaded-pipeline|--no-docling-use-threaded-pipeline] \
  [--docling-device <name>] \
  [--docling-threads <n>] \
  [--docling-layout-batch-size <n>] \
  [--docling-ocr-batch-size <n>] \
  [--docling-table-batch-size <n>] \
  [--docling-queue-max-size <n>] \
  [--docling-log-settings|--no-docling-log-settings]
```

| Argument | Description |
|---|---|
| `--root` | Root directory to scan recursively |
| `--max-files` | Cap the number of files processed in this run |
| `--skip-extraction` | Ingest files but do not run table extraction |
| `--state-file` | Custom path for the resumable state JSON (default: `.stemma/financial_pass_state.json`) |
| `--log-file` | Custom path for the JSONL log (default: `.stemma/financial_pass.log.jsonl`) |
| `--extract-timeout-seconds` | Per-file extraction timeout; defaults to 300 seconds |
| `--verbose-docs` / `--no-verbose-docs` | Print detailed terminal log line for each processed document (default: enabled) |
| `--docling-*` | Same docling performance flags as `stemma extract run`; apply to every PDF extraction in the pass |

**File detection rules:**

A file is included if:
1. Its extension is one of: `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.csv`, `.txt`, `.html`, `.htm`, `.md`
2. Its relative path (with `_` and `-` replaced by spaces) contains any of these keywords: `financial`, `annual report`, `report and accounts`, `accounts`, `account`, `statement`, `statements`, `funding`, `audit`

Extraction is only attempted for files with extractable media types: `application/pdf`, `text/markdown`, `text/plain`, `text/csv`.

**State file format:**
```json
{"processed_paths": ["/path/to/file1.pdf", "/path/to/file2.pdf"]}
```

**Log file format:** One JSON object per line:
```jsonl
{"path": "/path/to/file.pdf", "ingest_status": "ingested", "resource_id": "...", "digest": "...", "extract_status": "extracted:12", "elapsed_seconds": 4.38, "parse_elapsed_seconds": 3.91, "page_count": 87, "pages_per_second": 22.25}
{"path": "/path/to/broken.pdf", "error": "Extraction failed: ...", "elapsed_seconds": 300.01}
```

**Output:**
- Live terminal progress with overall file progress and current file status.
- Per-document verbose lines (default) showing ingest status, extract status, elapsed seconds, and parse speed when available.
- Final summary panel with candidate count, processed count, ingested, duplicates, extracted, skipped, failed.

---

### 6.14 `stemma web`

Starts the web GUI and API server using Uvicorn.

```bash
stemma web [--host 127.0.0.1] [--port 8765] [--reload]
```

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8765` | Port number |
| `--reload` | off | Enable Uvicorn hot reload (for development) |

Once running, open `http://127.0.0.1:8765` in a browser for the HTML UI.

The web UI now includes:
- A per-card `?` help icon (top-right) opening one popup with `Basic` guidance first and `Comprehensive` guidance below.
- A `Database Explorer` card for listing tables, viewing schema, and previewing table rows.

---

## 7. REST API Reference

The web server exposes a complete REST API. All JSON responses include an `"ok": true` field on success, or an HTTP error with a `"detail"` message on failure.

### Health / Init

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/init` | Initialise project (idempotent) |
| `GET` | `/api/doctor` | Run integrity checks |

`/api/doctor` returns `db_runtime` in addition to `checks_run` and `issues`.

### Resources

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/resources?limit=<n>` | List resources |
| `POST` | `/api/ingest/path` | Ingest file by server-side path |
| `POST` | `/api/ingest/upload` | Upload and ingest a file (multipart form) |

### Database Inspection

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/db/tables?limit=<n>` | List user tables with row counts, columns, and create SQL |
| `GET` | `/api/db/table?name=<table>&limit=<n>&offset=<n>` | Return selected table schema + row slice |

**`POST /api/ingest/path` body:**
```json
{"path": "/absolute/path/to/file.pdf", "source_uri": "https://..."}
```

### References & Citations

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/refs?limit=<n>` | List references |
| `GET` | `/api/citations?limit=<n>` | List citation mappings |
| `POST` | `/api/refs/import-bib` | Import BibTeX — body: `{"bib_path": "..."}` |
| `POST` | `/api/refs/link-resource` | Link citation to resource |

### Extraction

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/extract/run` | Run extraction — body: `{"resource_id"|"resource_digest": "...", "profile": "default"}` |
| `GET` | `/api/extract/tables?resource_id=<id>&limit=<n>` | List extracted tables |

### Claims

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/claims?claim_set=<name>&limit=<n>` | List claims |
| `GET` | `/api/claim-sets?limit=<n>` | List claim sets |
| `POST` | `/api/claims/import` | Import claims from server-side file |

**`POST /api/claims/import` body:**
```json
{"file_path": "/path/to/claims.csv", "fmt": "csv", "claim_set": "my-set", "description": "..."}
```

### Evidence Binding

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/bind/add` | Add evidence binding |
| `POST` | `/api/bind/validate` | Validate binding — body: `{"claim_id": "..."}` |

**`POST /api/bind/add` body:**
```json
{
  "claim_id": "...",
  "resource_digest": "...",
  "role": "value-cell",
  "selectors": [...],
  "page_index": 36,
  "note": "optional note"
}
```

### Verification

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/verify/claim` | Verify one claim — body: `{"claim_id": "...", "policy": "strict"}` |
| `POST` | `/api/verify/set` | Verify claim set — body: `{"claim_set": "...", "policy": "strict"}` |

### Reporting

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/report/verification?run_id=<id>[&json_out=<path>][&md_out=<path>]` | Get run summary + optional file export |

### Tracing

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/trace/claim?claim_id=<id>` | Trace claim |
| `GET` | `/api/trace/resource?resource_id=<id>` | Trace resource |
| `GET` | `/api/trace/citation?cite_id=<id>` | Trace citation |

### CEAPF

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/ceapf/propositions?limit=<n>` | List propositions |
| `POST` | `/api/ceapf/proposition` | Create proposition — body: `{"proposition": {...}}` |
| `POST` | `/api/ceapf/assertion` | Create assertion event |
| `POST` | `/api/ceapf/relation` | Create argument relation |

### Pipeline

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/pipeline/financial-pass` | Run financial pipeline |

**`POST /api/pipeline/financial-pass` body:**
```json
{"root": "/Volumes/X10/data", "max_files": 100, "skip_extraction": false, "extract_timeout_seconds": 300}
```

---

## 8. Claim Import Formats

### CSV

Must include a header row. Supported column names:

| Column | Required for | Description |
|---|---|---|
| `claim_type` | optional | `"quantitative"` or `"narrative"` (auto-detected if absent) |
| `subject` | quantitative | What the value is about |
| `predicate` | quantitative | The subject's property |
| `object_text` | optional | Object of the claim as text |
| `narrative_text` | narrative | The quoted/asserted text |
| `value_raw` | quantitative | Raw value string (e.g. `"5,631"` or `"(200)"`) |
| `value_parsed` | optional | Pre-parsed float |
| `currency` | optional | ISO currency code |
| `scale_factor` | optional | Integer scale (e.g. `1000` for £000s) |
| `period_label` | optional | Period identifier (e.g. `"FY2024/25"`) |
| `source_cite_id` | optional | 4-char citation ID |

**Example CSV:**
```csv
claim_type,subject,predicate,value_raw,currency,scale_factor,period_label
quantitative,City St George's University,cash_at_bank,"5,631",GBP,1000,FY2024/25
```

### JSON

A JSON object with a top-level `"claims"` array. Each element is a JSON object with the same fields as the CSV columns.

```json
{
  "claims": [
    {
      "claim_type": "quantitative",
      "subject": "City St George's University",
      "predicate": "cash_at_bank",
      "value_raw": "5631",
      "currency": "GBP",
      "scale_factor": 1000,
      "period_label": "FY2024/25"
    },
    {
      "claim_type": "narrative",
      "narrative_text": "The University's cash position improved significantly during the year."
    }
  ]
}
```

### Markdown

A Markdown table. The first row is the header; subsequent rows are claims. Column names follow the same naming as CSV.

```markdown
| claim_type | subject | predicate | value_raw | currency | period_label |
|---|---|---|---|---|---|
| quantitative | Institution X | cash_at_bank | 5631 | GBP | FY2024/25 |
```

---

## 9. Selector Types Reference

Selectors are JSON objects with a required `"type"` field. Every evidence item must have at least 2 selector objects with distinct types.

### `PageGeometrySelector`

Locates evidence by page index and one or more bounding boxes in page coordinates.

```json
{
  "type": "PageGeometrySelector",
  "pageIndex": 36,
  "boxes": [
    {"x0": 92.1, "y0": 301.4, "x1": 140.6, "y1": 315.2}
  ]
}
```

### `TableAddressSelector`

Locates a cell within an extracted table by stable table ID and cell coordinates. Also carries semantic metadata about units and period for quantitative verification.

```json
{
  "type": "TableAddressSelector",
  "table_id": "sha256:abcdef1234...",
  "cell_ref": {
    "row_index": 7,
    "col_index": 2
  },
  "row_path": ["Cash and cash equivalents"],
  "col_path": ["2025"],
  "units": {
    "currency": "GBP",
    "scale_factor": 1000
  },
  "period": {
    "label": "FY2024/25"
  }
}
```

The `table_id` **must** be the deterministic ID produced by `stemma extract tables`. Either `cell_ref` (direct index) or `row_path` / `col_path` (header-name lookup) can be used; both can be supplied for redundancy.

**Cell resolution order:** `cell_ref` is tried first, then `row_path` / `col_path` lookup by matching the last element of the path against the extracted header list.

### `TextQuoteSelector`

Locates evidence by an exact verbatim text fragment. Used for both narrative claim verification and as cross-locator redundancy for numeric claims.

```json
{
  "type": "TextQuoteSelector",
  "exact": "5,631",
  "prefix": "Cash at bank and in hand ",
  "suffix": " (2024: £"
}
```

The `prefix` and `suffix` fields are context anchors (from the W3C Web Annotation model) and are recorded but not currently used in verification logic — they assist human auditors.

### Custom selector types

Any object with a `"type"` field is accepted and stored. The system does not reject unknown selector types, making the format extensible.

---

## 10. Verification Logic Detail

### Pre-check: binding validation

Regardless of claim type, verification first runs `EvidenceBindingService.validate_binding()`:

1. All required roles must be present (see §4.5).
2. Each evidence item must have ≥ 2 distinct selector types.

If either check fails, the verification result is `"fail"` with `"reason": "binding_validation_failed"` and lists missing roles and/or weak evidence IDs.

### Quantitative claim verification

After the binding validation passes:

1. Locate the first evidence item with `role == "value-cell"`.
2. Find a `TableAddressSelector` within that evidence item's selectors.
3. Extract `table_id` from the selector.
4. Look up the extracted table in the database by `(resource_id, table_id)`.
5. Resolve the cell coordinates from `cell_ref` or by header-name index lookup via `row_path` / `col_path`.
6. Read the cell value from `cells_json`.
7. Parse the cell value as a float using a lenient parser that handles: accounting parentheses `(200)` → `-200`, comma-separated thousands, currency symbols `£ $`, and abbreviations `m M`.
8. Compare expected value (from `claim.value_parsed` or `claim.value_raw`) to actual extracted value.
9. If the absolute difference is greater than `1e-9`, fail with `"value_mismatch"`.
10. If `currency`, `scale_factor`, or `period_label` are set on both the claim and the selector's `units`/`period` fields, these are semantically compared — mismatches produce specific diagnostic reasons: `"currency_mismatch"`, `"scale_factor_mismatch"`, `"period_mismatch"`.

### Narrative claim verification

1. Iterate evidence items with `role == "quote"`.
2. For each item, check selectors of type `TextQuoteSelector`.
3. If the `"exact"` value (case-insensitively) is contained within `claim.narrative_text`, the result is `"pass"`.
4. If no matching text quote selector is found across all evidence items, the result is `"fail"` with `"reason": "no_matching_text_quote_selector"`.

### Diagnostic reasons

| Reason | Meaning |
|---|---|
| `binding_validation_failed` | Missing evidence roles or insufficient selector diversity |
| `missing_value_cell` | No evidence item with role `value-cell` |
| `missing_table_address_selector` | The value-cell evidence item has no `TableAddressSelector` |
| `table_address_missing_table_id` | The `TableAddressSelector` has an empty `table_id` |
| `table_not_found` | No extracted table with that `table_id` exists for the resource |
| `unable_to_resolve_cell_coordinates` | Neither `cell_ref` nor `row_path`/`col_path` resolved to valid indices |
| `cell_not_found` | No cell with the resolved row/col indices exists in `cells_json` |
| `numeric_parse_failed` | Either expected or actual value could not be parsed as a float |
| `value_mismatch` | Expected and actual numeric values differ |
| `currency_mismatch` | Currency declared in claim differs from selector |
| `scale_factor_mismatch` | Scale factor declared in claim differs from selector |
| `period_mismatch` | Period label declared in claim differs from selector |
| `no_matching_text_quote_selector` | No `TextQuoteSelector` text found in narrative claim text |
| `unsupported_claim_type:<type>` | The claim type is not `quantitative` or `narrative` |

---

## 11. The Immutable Archive

The archive at `.stemma/archive/` uses content-addressed, sharded storage.

**Layout:**
```
.stemma/archive/
└── sha256/
    └── <ab>/              ← first 2 hex chars of digest
        └── <cd>/          ← next 2 hex chars of digest
            └── <full-sha256><original-extension>
```

**Immutability guarantees:**
- Files are copied atomically using `shutil.copy2` to a temporary path, then `os.replace` to the final path.
- After writing, the file's permissions are set to read-only (`0o444` on POSIX, or `stat.S_IREAD` on Windows).
- If the destination path already exists, the copy is skipped (the file is already immutable).

**Integrity verification:**
The `ArchiveStore.verify_archived_integrity()` method re-computes the SHA-256 of the archived file and compares it to the recorded digest. This is used by `stemma doctor`.

---

## 12. Citation ID Allocation

Citation IDs are exactly 4 characters drawn from a Base62 alphabet (`A–Z`, `a–z`, `0–9`).

The algorithm:

1. Normalise the BibTeX cite key (lower-case, strip whitespace).
2. Feed `"{normalised_key}:{attempt}"` through SHA-256.
3. Take the first 10 bytes of the digest as a big-endian integer.
4. Encode that integer in Base62 with 4 digits.
5. If the resulting 4-char ID is already assigned to a *different* normalised key, increment `attempt` and retry.
6. If the ID maps to the same normalised key, return the existing ID (idempotent).
7. Up to 5000 attempts are made before raising an error (collision exhaustion is effectively impossible).

This means the same BibTeX key always receives the same cite ID, and IDs survive across reimports of the same database.

---

## 13. The Financial Pipeline

The `FinancialPipelineService` combines ingestion and extraction into a single resumable pass over a directory tree.

**How it works:**

1. `find_financial_candidates(root)` — recursively lists all files, filters by extension and filename keyword matching.
2. `load_state()` — reads `financial_pass_state.json` to find which paths were already processed.
3. For each unprocessed candidate:
   - Call `IngestionService.ingest_file()`. If duplicate, count it but do not skip extraction.
   - If the media type is extractable and extraction hasn't run before: call `ExtractionService.extract_resource()` with a per-file SIGALRM timeout.
   - Write a structured log row to the JSONL log.
   - Add the path to `processed_paths` and save state after every file (so a crash mid-run doesn't lose progress).
4. Return `PipelineStats`.

**Timeout mechanism:** Uses POSIX `SIGALRM` (Unix only). Each file extraction is wrapped in a context manager that sets `signal.alarm(seconds)`; if the extraction exceeds the timeout, a `TimeoutError` is raised, counted as a failure, and the pipeline continues with the next file.

**Idempotency:** Re-running the pipeline on the same root produces no duplicate records because:
- Ingestion deduplicates by SHA-256.
- Extraction checks for existing runs before re-running.
- Processed paths are tracked in the state file.

---

## 14. CEAPF Conceptual Model

CEAPF (Claim–Evidence–Argument Provenance Format) is Stemma Codicum's implementation of a higher-level argument graph, designed for cases where machine-verification alone is insufficient.

The core insight: **a claim should not be treated as a single blob**. Three distinct things must be modelled and linked:

```
Proposition
  └── "Institution X spent £3.4m on Y in FY2024/25"
      (the content-level statement about the world)

Assertion Event
  └── "Author Z asserts [Proposition] in [source] with modality: asserts"
      (the speech act — what an agent said and where)

Assessment / Argument Structure
  └── supports(Interpretation → Proposition)
  └── rebuts(Counter-Proposition → Proposition)
  └── undercuts(Counter-Evidence → Inference Step)
  └── qualifies(Proposition → Proposition)
```

**Why this separation matters:**

| Thing | Machine-verifiable? |
|---|---|
| Assertion event occurred | Often yes — the quote exists at the anchored location |
| Quantitative proposition | Sometimes yes — if the table cell matches |
| Predictive/normative proposition | Not directly — awaits future evidence |

By making the inference step a first-class object, counter-evidence can target either the proposition ("the number is wrong") or the bridge from evidence to meaning ("those figures are in £000, not £m") — an *undercutting* attack on the inference rather than a *rebuttal* of the conclusion.

**Relation semantics (Dung-inspired):**

| Type | Meaning |
|---|---|
| `supports` | P1 provides positive evidence for P2 |
| `rebuts` | P1 attacks the conclusion P2 |
| `undercuts` | P1 attacks an inference or interpretation step |
| `qualifies` | P1 narrows the scope or adds conditions to P2 |

---

## 15. Testing

Tests are in `tests/unit/` and use `pytest`.

```bash
pytest
pytest -q          # quiet
pytest -v          # verbose
```

Each test module bootstraps its own isolated SQLite database in `tmp_path` (pytest's temp-dir fixture). No shared state exists between tests.

Significant test modules:

| Module | What it covers |
|---|---|
| `test_verification_service.py` | Full quantitative and narrative claim verification end-to-end |
| `test_evidence_binding_service.py` | Binding validation role and selector-diversity rules |
| `test_claim_service.py` | CSV, JSON, Markdown import; auto-detection of claim type |
| `test_reference_service.py` | BibTeX import, cite ID allocation, resource linking |
| `test_trace_service.py` | Claim, resource, and citation trace queries |
| `test_reporting_service.py` | JSON and Markdown report export |
| `test_pipeline_service.py` | Resumable financial pipeline (candidate detection, state management) |
| `test_archive_store.py` | Immutable file storage and integrity verification |
| `test_hashing.py` | SHA-256 hashing utilities |
| `test_web_app.py` | FastAPI routes via `httpx` test client |
| `test_ceapf_service.py` | CEAPF proposition / assertion / relation operations |
| `test_docling_adapter.py` | Markdown table parser |
| `test_health_service.py` | Doctor check logic |

---

## 16. End-to-End Worked Example

This example follows the complete workflow to verify a single quantitative claim: that a named university had £5,631,000 cash at bank as of 31 July 2025.

### Step 1: Initialise

```bash
mkdir my-project && cd my-project
stemma init
```

### Step 2: Ingest the source PDF

```bash
stemma ingest ~/downloads/university-accounts-2025.pdf \
  --source-uri "https://example.ac.uk/annual-report-2025.pdf"
```

Note the SHA-256 digest from the output, e.g. `a3f9...`.

### Step 3: Import the BibTeX reference

Create `refs.bib`:
```bibtex
@techreport{UniversityAccounts2025,
  title  = {Annual Report and Financial Statements 2025},
  author = {Example University},
  year   = {2025},
  url    = {https://example.ac.uk/annual-report-2025.pdf}
}
```

```bash
stemma refs import-bib refs.bib
stemma refs citations  # note the 4-char cite ID, e.g. "Xk9M"
```

### Step 4: Link the reference to the resource

```bash
stemma refs link-resource --cite-id Xk9M --resource-digest a3f9...
```

### Step 5: Extract tables from the PDF

```bash
stemma extract run --resource-digest a3f9...
stemma extract tables --resource-digest a3f9...
```

Note the `table_id` (e.g. `sha256:7c3b...`) for the table containing cash figures.

### Step 6: Create and import claims

Create `claims.csv`:
```csv
claim_type,subject,predicate,value_raw,currency,scale_factor,period_label,source_cite_id
quantitative,Example University,cash_at_bank,"5,631",GBP,1000,FY2024/25,Xk9M
```

```bash
stemma claims import --file claims.csv --format csv --claim-set accounts-2025
stemma claims list --claim-set accounts-2025
```

Note the claim UUID (e.g. `c1a2...`).

### Step 7: Bind evidence to the claim

Create `value_selectors.json`:
```json
[
  {
    "type": "PageGeometrySelector",
    "pageIndex": 49,
    "boxes": [{"x0": 400.0, "y0": 612.0, "x1": 480.0, "y1": 625.0}]
  },
  {
    "type": "TableAddressSelector",
    "table_id": "sha256:7c3b...",
    "cell_ref": {"row_index": 0, "col_index": 1},
    "units": {"currency": "GBP", "scale_factor": 1000},
    "period": {"label": "FY2024/25"}
  }
]
```

```bash
# Bind the value cell
stemma bind add --claim-id c1a2... --resource-digest a3f9... \
  --role value-cell --selectors-file value_selectors.json --page-index 49

# Bind the row header
stemma bind add --claim-id c1a2... --resource-digest a3f9... \
  --role row-header \
  --selectors-json '[{"type":"TextQuoteSelector","exact":"Cash at bank and in hand"},{"type":"PageGeometrySelector","pageIndex":49,"boxes":[]}]'

# Bind the column header
stemma bind add --claim-id c1a2... --resource-digest a3f9... \
  --role column-header \
  --selectors-json '[{"type":"TextQuoteSelector","exact":"2025"},{"type":"PageGeometrySelector","pageIndex":49,"boxes":[]}]'

# Bind the caption
stemma bind add --claim-id c1a2... --resource-digest a3f9... \
  --role caption \
  --selectors-json '[{"type":"TextQuoteSelector","exact":"Table 23: Liquidity"},{"type":"PageGeometrySelector","pageIndex":49,"boxes":[]}]'
```

Validate completeness:
```bash
stemma bind validate --claim-id c1a2...
```

### Step 8: Verify

```bash
stemma verify claim --claim-id c1a2...
```

Expected output:
```
╭─ Verify Claim ─────────────────────────╮
│ Run ID: <uuid>                          │
│ Claim ID: c1a2...                       │
│ Status: PASS                            │
│ Diagnostics: {'reason': 'quantitative_match', 'table_id': 'sha256:7c3b...', 'row_index': 0, 'col_index': 1, 'value': 5631.0}
╰─────────────────────────────────────────╯
```

### Step 9: Report

```bash
stemma report verification --run-id <run-uuid> --md-out verification-report.md
```

### Step 10: Trace

```bash
stemma trace claim --claim-id c1a2...
stemma trace resource --resource-digest a3f9...
stemma trace citation --cite-id Xk9M
```

### Step 11: Doctor check

```bash
stemma doctor
```

Should produce `Status: PASS` with `Issues: 0`.

---

*End of documentation.*
