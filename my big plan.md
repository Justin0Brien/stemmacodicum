# Stemma Codicum: Big Plan (CLI-first, GUI-ready)

## 1. What I Understand the Project To Be

Stemma Codicum is a machine-verifiable claim-to-evidence system.

Primary objective now:

- Verify every claim in a university financial report by machine, not by manual inspection.
- Keep an auditable chain from claim -> citation/reference -> immutable source artifact -> exact evidence span (for example table cell with row/column/caption/unit/period context).

Long-term objective:

- Become a general claim-evidence-argument epistemology tool (CEBF/CEAPF aligned), including support for competing interpretations, counter-evidence, and argument structure.

Non-negotiables captured from your docs:

- Source identity is content-addressed (hash-led, immutable artifact handling).
- Evidence addressing is redundant (not one brittle locator).
- Table claims must include context envelope (value cell, row header, column header, caption, notes/units/period).
- Verification is deterministic and reproducible with clear failure diagnostics.
- CLI-first implementation with rich-powered UX.
- Architecture must allow adding GUI later without rewriting core logic.

## 2. Delivery Strategy

Use a layered architecture with strict boundaries.

- Domain layer: pure models, validation rules, invariants.
- Application layer: use-cases/services (ingest, extract, bind, verify, report).
- Infrastructure layer: SQLite repos, file archive, docling adapter, filesystem, optional downloads.
- Interface layer: CLI now; GUI adapter later.

Rule:

- CLI commands call application services only.
- Application services call interfaces (protocols), not concrete infra directly.
- Infra implements protocols.

This gives GUI bolt-on capacity because GUI can call the same use-cases as CLI.

## 3. Scope Split (Now vs Later)

### 3.1 MVP (first functional target)

- Ingest one or more source documents (especially the university report PDF).
- Store immutable artifact metadata and digests.
- Parse document with docling adapter and persist structured extraction outputs.
- Create/import claims with explicit value + period + units.
- Bind each claim to evidence bundle (value cell + context roles).
- Run deterministic verifier and generate pass/fail + diagnostics.
- Export verification report (terminal table + JSON + markdown summary).

### 3.2 Near-term expansion

- Citation normalization pipeline (4-char cite IDs + UUID mapping).
- Bibliographic/reference DB and source traceability.
- Bulk claim import from markdown/csv/json.
- Regression verification runs for entire claim sets.

### 3.3 Later (CEAPF breadth)

- Assertion events (speech acts).
- Interpretation and inference nodes as first-class objects.
- Argument graph (supports/rebuts/undercuts/qualifies).
- Attestation/signing wrappers.

## 4. Proposed Repository Layout

```text
stemmacodicum/
  pyproject.toml
  README.md
  my big plan.md
  src/stemmacodicum/
    __init__.py
    cli/
      main.py
      commands/
        init_cmd.py
        ingest_cmd.py
        extract_cmd.py
        claim_cmd.py
        bind_cmd.py
        verify_cmd.py
        report_cmd.py
        trace_cmd.py
    core/
      config.py
      logging.py
      ids.py
      time.py
      hashing.py
      errors.py
      types.py
    domain/
      models/
        resource.py
        citation.py
        reference.py
        claim.py
        evidence.py
        selector.py
        verification.py
        provenance.py
      rules/
        claim_rules.py
        evidence_rules.py
        selector_rules.py
      enums.py
    application/
      services/
        project_service.py
        ingestion_service.py
        extraction_service.py
        claim_service.py
        evidence_binding_service.py
        verification_service.py
        reporting_service.py
        trace_service.py
      dto/
        io_models.py
    infrastructure/
      db/
        sqlite.py
        schema.sql
        migrations/
        repos/
          resource_repo.py
          citation_repo.py
          reference_repo.py
          claim_repo.py
          evidence_repo.py
          verification_repo.py
      archive/
        store.py
      parsers/
        docling_adapter.py
      resolvers/
        pdf_selector_resolver.py
      importers/
        markdown_claim_importer.py
        csv_claim_importer.py
      exporters/
        json_exporter.py
        markdown_exporter.py
    interfaces/
      protocols/
        repositories.py
        parser.py
        archive.py
        verifier.py
  tests/
    unit/
    integration/
    fixtures/
```

## 5. Data Model Plan (SQLite + Domain Objects)

### 5.1 Core entities

- `resources`: immutable source artifacts (UUID, canonical digest, media type, archive path, metadata).
- `resource_digests`: additional digests (sha256 required, sha512 optional).
- `citations`: 4-char cite IDs mapped to canonical reference UUID.
- `references`: bibliography metadata and retrieval data.
- `reference_resources`: mapping from references to resources (many-to-many).
- `claims`: machine-interpretable proposition-like records for MVP quantitative claims.
- `claim_sets`: logical groupings (for one report, one notebook, one run).
- `evidence_items`: role-based evidence anchors tied to a resource.
- `selector_sets`: selectors attached to evidence items.
- `table_addresses`: normalized table selector details (table_id, row_path, col_path, etc).
- `bindings`: links from claim to required evidence roles.
- `verification_runs`: execution metadata for verifier runs.
- `verification_results`: per-claim outcome and diagnostics.
- `provenance_events`: extraction/import/normalization provenance records.

### 5.2 Constraints and invariants

- No destructive deletes on critical evidence lineage tables.
- `resources.digest_sha256` unique.
- For table-quant claims, binding must include required roles:
  - value-cell, row-header, column-header, caption.
- `selector_set` must contain at least 2 selector types.
- Every verification result tied to specific claim version + resource digest + extractor version.

## 6. Canonical Domain Contracts (MVP)

### 6.1 Claim object (MVP quantitative)

Required fields:

- `claim_id` (UUID)
- `subject`
- `predicate`
- `value_raw`
- `value_parsed`
- `currency`
- `scale_factor`
- `period_label` (and optional normalized period interval)
- `human_text`
- `claim_set_id`

### 6.2 Evidence item

Required fields:

- `evidence_id`
- `resource_id`
- `role`
- `selector_set`
- `extraction_provenance`

### 6.3 Selector types for MVP

- `PdfFragmentSelector`
- `PageGeometrySelector`
- `TextQuoteSelector`
- `ContentHashSelector`
- `TableAddressSelector` (required on value-cell for table claims)

### 6.4 Verification output

- `status`: pass | fail | indeterminate
- `selector_matches`: matrix by selector type
- `semantic_checks`: value match, row/col path match, unit and period evidence checks
- `diagnostics`: structured reasons
- `reproducibility`: extractor version, config digest, run timestamp

## 7. CLI Product Surface (Rich-powered)

Top-level command: `stemma`

Planned command set:

- `stemma init`
- `stemma ingest`
- `stemma extract`
- `stemma claims import`
- `stemma claims list`
- `stemma bind add`
- `stemma bind validate`
- `stemma verify run`
- `stemma verify claim`
- `stemma report verification`
- `stemma trace claim`
- `stemma trace resource`

Rich UX standards:

- Structured colored log levels.
- Progress bars for batch ingest/extract/verify.
- Summary panels with pass/fail counts and confidence warnings.
- Failure tables with precise mismatch reason and location hints.
- Optional `--json` output for automation.

## 8. Stage-by-Stage Implementation Plan

## Stage 0: Foundations and Boundaries

Outcome:

- Project scaffolding, dependency setup, config system, logging conventions, error taxonomy.

Build:

- Package structure and dependency injection bootstrap.
- Global config (`XDG` friendly), environment variable overrides.
- `rich` logging and task progress wrapper utilities.
- Baseline test harness.

Acceptance:

- `stemma --help` works.
- Logging format consistent across commands.
- Unit tests run green for core utilities.

## Stage 1: Immutable Resource Registry

Outcome:

- Reliable source artifact ingestion with content addressing.

Build:

- File ingest pipeline: hash -> copy to archive -> metadata persist.
- Optional URL retrieval metadata capture.
- Duplicate handling by digest dedupe.

Acceptance:

- Same file ingested twice points to one resource digest identity.
- Resource trace command can show origin path/url and archive location.

## Stage 2: Citation and Reference Layer (MVP compatible)

Outcome:

- Central citation/reference records linked to resources.

Build:

- 4-char citation mapping store.
- BibTeX parse/import utilities.
- Link references to resource artifacts.

Acceptance:

- Existing citation sets can be imported and normalized.
- A claim can point to citations that map to concrete resources.

## Stage 3: Extraction Layer (Docling Adapter)

Outcome:

- Deterministic extraction artifacts from PDFs, especially tables.

Build:

- Adapter wrapping docling execution and config capture.
- Persist extraction outputs + digest + tool version.
- Table model normalization and stable table_id derivation.

Acceptance:

- Re-running extraction on same bytes with same config yields same table IDs.
- Extracted table model is queryable for row/column/cell references.

## Stage 4: Claim Model + Claim Set Management

Outcome:

- Import and manage machine-verifiable claims.

Build:

- Claim CRUD for MVP quantitative schema.
- Bulk importers (markdown/csv/json).
- Claim set grouping for report-wide verification.

Acceptance:

- Thousands of claims can be loaded and listed by status.
- Invalid claims (missing units/period/value parse) are rejected with clear errors.

## Stage 5: Evidence Binding Workbench (CLI)

Outcome:

- Claims bound to role-complete evidence bundles.

Build:

- Binding commands for role assignments.
- Binding validation logic against required role policy.
- Selector capture and normalization utilities.

Acceptance:

- System blocks verification if mandatory evidence roles are missing.
- Binding validator reports exactly what role/selector is missing.

## Stage 6: Deterministic Verifier

Outcome:

- Machine-verifiable pass/fail for each claim.

Build:

- Selector resolution engine.
- Concordance checks across selectors.
- Table semantic checks: row_path, col_path, value parsing/scaling/unit/period.
- Policy engine for required/optional selector rules.

Acceptance:

- For known-good claims, verifier passes deterministically.
- For mismatched headers/values/units, verifier fails with specific diagnostics.

## Stage 7: Reporting and Audit Exports

Outcome:

- Actionable outputs for auditing and correction loops.

Build:

- Rich terminal summaries and drilldown tables.
- JSON report for automation.
- Markdown report for narrative audit trail.

Acceptance:

- One command yields portfolio-level pass/fail and per-claim diagnostics.
- Report includes reproducibility metadata.

## Stage 8: Traceability and Provenance Navigation

Outcome:

- End-to-end traversal from claim to source and back.

Build:

- `trace claim` and `trace resource` pathways.
- Provenance graph export (JSON-LD-ready structure even if simplified initially).

Acceptance:

- Given claim ID, tool prints full chain to evidence selectors and resource digest.
- Given resource digest, tool lists dependent claims and verification state.

## Stage 9: Hardening for Real Portfolio Runs

Outcome:

- Stable operation on large claim volumes.

Build:

- Batch processing optimizations.
- Retryable jobs and resumable run state.
- Data integrity checks and migration safety.
- Extensive integration tests with realistic report fixtures.

Acceptance:

- End-to-end verification over full report claim set completes reliably.
- Runtime and memory acceptable for repeated use.

## Stage 10: CEAPF Expansion (Post-MVP)

Outcome:

- Move from claim-evidence verification to full argument/provenance framework.

Build:

- Assertion event model.
- Interpretation/inference node modeling.
- Support/rebut/undercut/qualify argument relations.
- Warrant/acceptability evaluation profiles.

Acceptance:

- System can represent and query both supporting and opposing claim structures.

## 9. Function-by-Function Plan (Initial Build)

## 9.1 `core/`

- `load_config()`: Resolve config from file/env/defaults.
- `configure_logging()`: Install rich logger and structured context.
- `new_uuid()`: Generate UUIDs consistently.
- `now_utc_iso()`: Canonical timestamps.
- `compute_file_digest(path, alg='sha256')`: Byte-level digest.
- `compute_bytes_digest(data, alg='sha256')`: In-memory digest helper.
- `safe_copy_atomic(src, dst)`: Atomic copy primitive for archive writes.
- `raise_user_error(message, hint=None)`: Consistent CLI-facing exceptions.

## 9.2 `domain/models/`

- `Resource` model with digest invariants.
- `Reference` and `Citation` models with normalization checks.
- `Claim` model with value/units/period requirements.
- `EvidenceItem` model with role and selector constraints.
- `Selector` union types with type-specific validators.
- `VerificationResult` model for deterministic output schema.

## 9.3 `application/services/project_service.py`

- `init_project(config)`: Initialize DB, directories, baseline metadata.
- `health_check()`: Validate connectivity for DB/archive/parser tooling.

## 9.4 `application/services/ingestion_service.py`

- `ingest_file(path, source_meta)`: Ingest single local file.
- `ingest_batch(paths, source_meta_defaults)`: Batch ingest with progress.
- `ingest_url(url, retrieval_policy)`: Fetch + ingest with provenance metadata.
- `dedupe_resource(digest)`: Resolve existing resource identity.

## 9.5 `application/services/extraction_service.py`

- `extract_resource(resource_id, parser_profile)`: Run parser for one resource.
- `extract_batch(resource_ids, parser_profile)`: Batch extraction runs.
- `derive_table_id(table_obj)`: Stable deterministic table_id.
- `persist_extraction(resource_id, extraction_payload, provenance)`: Save extraction bundle.

## 9.6 `application/services/claim_service.py`

- `create_claim(claim_payload)`: Validate and persist claim.
- `import_claims(file_path, format)`: Bulk import claims.
- `list_claims(filters)`: Query claims by set/status/topic.
- `update_claim(claim_id, patch)`: Controlled claim edits with versioning.
- `create_claim_set(name, metadata)`: Grouping for report-level runs.

## 9.7 `application/services/evidence_binding_service.py`

- `bind_evidence(claim_id, evidence_payloads)`: Attach role-based evidence.
- `add_selector(evidence_id, selector_payload)`: Add locator to evidence item.
- `validate_binding(claim_id, policy)`: Check mandatory roles/selectors.
- `auto_bind_from_table_address(claim_id, extraction_context)`: Optional helper for table claims.

## 9.8 `application/services/verification_service.py`

- `verify_claim(claim_id, policy_profile)`: Full verification for one claim.
- `verify_claim_set(claim_set_id, policy_profile)`: Batch verification.
- `resolve_selector(resource, selector)`: Dispatch selector resolver.
- `check_selector_concordance(resolutions, tolerance)`: Cross-selector consistency.
- `verify_table_semantics(claim, table_address, extraction_model)`: Row/col/value/unit/period checks.
- `compare_claim_value_to_evidence(claim_value, evidence_value, rounding_policy)`: Numeric comparison.
- `build_verification_diagnostics(context)`: Human + machine diagnostics payload.

## 9.9 `application/services/reporting_service.py`

- `build_run_summary(run_id)`: Aggregate counts and error classes.
- `render_rich_summary(summary)`: Terminal dashboard.
- `export_json_report(run_id, out_path)`: Machine pipeline output.
- `export_markdown_report(run_id, out_path)`: Human audit narrative.

## 9.10 `application/services/trace_service.py`

- `trace_claim(claim_id)`: Claim -> evidence -> resource -> reference/citation.
- `trace_resource(resource_id)`: Resource -> dependent claims -> statuses.
- `trace_citation(cite_id)`: Citation -> references -> resources -> claims.

## 9.11 `infrastructure/db/repos/`

Each repo will provide CRUD + query primitives without domain business logic.

- `ResourceRepo`: `insert`, `get_by_digest`, `get_by_id`, `list`.
- `ReferenceRepo`: `insert`, `upsert_bib_entry`, `link_resource`.
- `CitationRepo`: `upsert_mapping`, `resolve_cite_id`.
- `ClaimRepo`: `insert`, `update`, `list_by_set`, `get`.
- `EvidenceRepo`: `insert_item`, `insert_selector`, `list_for_claim`.
- `VerificationRepo`: `create_run`, `insert_result`, `list_results`.

## 9.12 `infrastructure/archive/store.py`

- `ensure_archive_layout(base_dir)`: Create deterministic archive structure.
- `archive_path_for_digest(digest)`: Content-addressed path mapping.
- `store_file_immutable(src, digest)`: Atomic copy + read-only bit.
- `verify_archived_integrity(path, expected_digest)`: Re-hash integrity check.

## 9.13 `infrastructure/parsers/docling_adapter.py`

- `check_docling_available()`: Runtime capability check.
- `parse_pdf(resource_path, profile)`: Execute extraction.
- `normalize_docling_tables(raw_output)`: Canonical table structure.
- `extract_context_spans(table_model, cell_ref)`: Build context envelope candidates.

## 9.14 `infrastructure/resolvers/pdf_selector_resolver.py`

- `resolve_pdf_fragment(selector, resource)`: Fragment-level locate.
- `resolve_geometry(selector, resource)`: Geometry resolution.
- `resolve_text_quote(selector, resource)`: Quote resolution with context.
- `resolve_content_hash(selector, resource, normalizer)`: Region hash validation.
- `resolve_table_address(selector, extraction_model)`: Table semantics locate.

## 9.15 CLI command functions

Each command module exposes:

- `register(subparsers)`
- `run(args, services)`

Key command handlers:

- `run_init`
- `run_ingest`
- `run_extract`
- `run_claims_import`
- `run_bind_add`
- `run_bind_validate`
- `run_verify_claim`
- `run_verify_run`
- `run_report_verification`
- `run_trace_claim`
- `run_trace_resource`

## 10. Verification Policy Profiles

Define named policy profiles to support practical rollout:

- `strict`: all required roles + multi-selector concordance required.
- `balanced`: role-complete; allows one optional selector miss if others agree.
- `lenient-import`: temporary for migration, flags weak bindings as warnings.

Policy object controls:

- Required selector types by role.
- Geometry tolerance.
- Text normalization pipeline.
- Numeric rounding tolerance rules.
- Pass/fail thresholds.

## 11. Testing Strategy

Test layers:

- Unit tests for models, parsers, resolvers, value normalization.
- Integration tests for ingest/extract/bind/verify pipelines on fixture PDFs.
- Snapshot tests for report outputs.
- Regression suite for known failures from real financial-report claims.

Critical fixtures:

- One university report PDF with known-good claims.
- Claims that should fail due to wrong unit/period/row/column.
- Selector drift scenarios.

Definition of done for MVP:

- Can load full claim set for a report and produce deterministic verification report.
- All pass/fail outcomes reproducible on rerun with same inputs.

## 12. Migration and Interoperability Plan

- Provide import adapters for existing citation manager artifacts (`id_mapping.json`, `references.bib`, backlinks).
- Provide optional bridge scripts to read existing SQLite catalogs.
- Maintain stable IDs where possible to avoid rework in historical notes.
- Preserve provenance when importing legacy references and source files.

## 13. GUI-readiness Plan (No GUI build yet)

Prepare now so GUI later is low-risk:

- Service API returns typed DTOs independent of CLI formatting.
- Long operations expose progress callbacks/events.
- Command handlers thin, no domain logic.
- Report generation reusable for API endpoints.

Expected future GUI surfaces:

- Claim review queue.
- Evidence binding visual table explorer.
- Verification failure triage panel.
- Provenance graph view.

## 14. Practical Rollout for Your Immediate Need

Execution sequence for the university report portfolio:

1. Initialize project and ingest the report artifacts.
2. Run extraction and persist table models.
3. Import claims in bulk into one `claim_set` for the report.
4. Bind each claim to role-complete evidence items (manual + helper automation).
5. Run strict verification.
6. Export failing-claim diagnostics and iterate corrections.
7. Repeat until full claim set reaches required verification threshold.

## 15. Open Decisions to Confirm Before Build Starts

I can proceed with sensible defaults, but these decisions affect implementation details:

- Canonical DB location: project-local `.stemma/` vs shared location in home directory.
- Archive mode: copy into managed archive only vs allow immutable external references.
- Claim import format priority: markdown first, csv first, or both equally.
- Initial policy default: `strict` or `balanced`.
- Minimum supported source types in MVP: PDF only or PDF + HTML.

## 16. Initial Success Metrics

- Percentage of claims in target report that are machine-verifiable.
- Mean time to diagnose a failed claim binding.
- Determinism rate across reruns (target 100 percent with fixed inputs).
- Coverage of claims with role-complete evidence bundles.
- Reduction in manual audit effort per report cycle.

## 17. Summary

This plan builds Stemma Codicum as a rigorous, deterministic claim-evidence verification system first (MVP for your financial-report claims), while structuring code and data so CEAPF-scale argument and provenance capabilities can be added without architectural rewrite.
