# Stemma Codicum Web App Tutorial

This guide is for users who are new to Stemma Codicum and want to use the web app for all supported workflows.

It covers:

- project setup and health checks
- ingestion and extraction
- references and citations
- claims, evidence binding, verification, and reports
- trace and CEAPF operations
- financial batch pipeline
- the Extraction Inspector, including "Copy Selector JSON"

---

## 1. Prerequisites

- Python virtual environment is set up in the repo
- Dependencies are installed
- You can run the web app command

Start the web app from the repo root:

```bash
cd /Users/justin/code/stemmacodicum
PYTHONPATH=src ./.venv/bin/python -m stemmacodicum web --host 127.0.0.1 --port 8765
```

Open:

- `http://127.0.0.1:8765`

---

## 2. Recommended tutorial data

Create a folder such as:

- `/Users/justin/code/stemmacodicum/tutorial-data`

Create these files.

### 2.1 Financial source document (`report.md`)

```md
Table 1: Liquidity Summary

| Metric | FY2024/25 |
|---|---:|
| Cash at bank | 5631 |
| Debt | 120 |

Management states that liquidity improved significantly during the period.
```

### 2.2 References file (`refs.bib`)

```bibtex
@report{inst2025,
  title={Institution Annual Report 2025},
  author={Institution Finance Office},
  year={2025},
  url={https://example.org/annual-report-2025}
}
```

### 2.3 Quantitative claims (`claims-quant.json`)

```json
{
  "claims": [
    {
      "claim_type": "quantitative",
      "subject": "Institution",
      "predicate": "cash_at_bank",
      "value_raw": "5631",
      "currency": "GBP",
      "period_label": "FY2024/25"
    }
  ]
}
```

### 2.4 Narrative claims (`claims-narrative.json`)

```json
{
  "claims": [
    {
      "claim_type": "narrative",
      "narrative_text": "The annual report discusses liquidity."
    }
  ]
}
```

---

## 3. UI orientation

The app has a left navigation and one main workspace card at a time.

Cards available:

1. Project and Health
2. Database Explorer
3. Ingest
4. References
5. Extraction
6. Claims
7. Binding
8. Verification and Reports
9. Trace
10. CEAPF
11. Financial Pipeline

Every card has a `?` help icon with Basic and Comprehensive guidance.

---

## 4. Screenshot placeholders

The tutorial includes one screenshot placeholder per card. Replace each image path with your real screenshots.

![Project and Health placeholder](docs/screenshots/01-project-and-health.png)
![Database Explorer placeholder](docs/screenshots/02-database-explorer.png)
![Ingest placeholder](docs/screenshots/03-ingest.png)
![References placeholder](docs/screenshots/04-references.png)
![Extraction placeholder](docs/screenshots/05-extraction.png)
![Claims placeholder](docs/screenshots/06-claims.png)
![Binding placeholder](docs/screenshots/07-binding.png)
![Verification and Reports placeholder](docs/screenshots/08-verification-and-reports.png)
![Trace placeholder](docs/screenshots/09-trace.png)
![CEAPF placeholder](docs/screenshots/10-ceapf.png)
![Financial Pipeline placeholder](docs/screenshots/11-financial-pipeline.png)

---

## 5. Card-by-card tutorial

## 5.1 Project and Health

Purpose:

- initialize project storage and schema
- run integrity/runtime checks

Steps:

1. Open `Project and Health`.
2. Click `Init Project`.
3. Click `Doctor`.
4. Confirm output includes checks and database runtime info.

Expected:

- `.stemma` storage exists
- DB runtime settings are visible
- no blocking issues

Screenshot placeholder:

![Project and Health card](docs/screenshots/01-project-and-health.png)

---

## 5.2 Database Explorer

Purpose:

- inspect table names, schema, and rows directly

Steps:

1. Open `Database Explorer`.
2. Click `List Tables`.
3. Select a table from dropdown.
4. Click `Describe Table` for schema.
5. Click `View Table` for rows.
6. Use filters/sorting in spreadsheet mode.

Useful tables during this tutorial:

- `resources`
- `reference_entries`
- `extraction_runs`
- `extracted_tables`
- `document_texts`
- `text_segments`
- `text_annotations`
- `text_annotation_spans`
- `verification_results`

Screenshot placeholder:

![Database Explorer card](docs/screenshots/02-database-explorer.png)

---

## 5.3 Ingest

Purpose:

- archive source files immutably (digest-addressed)
- obtain resource IDs/digests for downstream steps

Example A: ingest by path

1. Open `Ingest`.
2. Enter:
   `/Users/justin/code/stemmacodicum/tutorial-data/report.md`
3. Click `Ingest Path`.
4. Click `List Resources`.
5. Save:
   - `resource.id`
   - `digest_sha256`

Example B: drag-and-drop upload

1. Drop a file onto the dropzone.
2. Confirm response status (`ingested` or `duplicate`).

Screenshot placeholder:

![Ingest card](docs/screenshots/03-ingest.png)

---

## 5.4 References

Purpose:

- import bibliographic records
- link citations to ingested resources

Steps:

1. Open `References`.
2. `Import BibTeX` using:
   `/Users/justin/code/stemmacodicum/tutorial-data/refs.bib`
3. Click `List Citations` and copy a `cite_id`.
4. In `Link Ref->Resource`, provide:
   - `cite_id`
   - `resource_digest` from Ingest
5. Click `Link Ref->Resource`.
6. Optionally click `List Refs` to verify link chain.

Screenshot placeholder:

![References card](docs/screenshots/04-references.png)

---

## 5.5 Extraction

Purpose:

- run parser extraction
- inspect deterministic table IDs
- inspect extracted text/standoff layers via Extraction Inspector

### A. Run extraction and list tables

1. Open `Extraction`.
2. Enter either:
   - `Resource ID`, or
   - `Resource digest`
3. Click `Run Extract`.
4. In `List Tables` row, provide the same resource selector.
5. Click `List Tables`.
6. Save `table_id` from output.

### B. Use Extraction Inspector

Inputs:

- `Inspector resource ID` or `Inspector resource digest` (required)
- `Inspector run ID` (optional for historical run inspection)

Actions:

1. `Inspect Text`:
   - confirms canonical extracted text and digest
2. `Inspect Segments`:
   - optional `segment_type` filter (for example `layout:paragraph`)
   - renders compact segment cards
3. `Inspect Annotations`:
   - optional `layer` filter (for example `domain_financial`)
   - optional `category` filter (for example `metric`)
   - renders compact annotation cards
4. `Inspect Dump`:
   - full extraction payload summary using limits

### C. Copy selector JSON from cards

When inspecting segments/annotations:

- each compact card has `Copy Selector JSON`

Generated selector types:

- segment card: `TextPositionSelector`
- annotation card: `TextAnnotationSelector`

These are intended for quick paste into Binding selector JSON.

Screenshot placeholder:

![Extraction card](docs/screenshots/05-extraction.png)

---

## 5.6 Claims

Purpose:

- import structured claims into named claim sets

Example A: quantitative claim set

1. Open `Claims`.
2. Set:
   - `Claims file`: `/Users/justin/code/stemmacodicum/tutorial-data/claims-quant.json`
   - `Format`: `json`
   - `Claim set name`: `tutorial-quant`
3. Click `Import Claims`.
4. Click `List Claims` filtered by `tutorial-quant`.
5. Copy `claim_id`.

Example B: narrative claim set

1. Import:
   `/Users/justin/code/stemmacodicum/tutorial-data/claims-narrative.json`
2. Use claim set `tutorial-narrative`.
3. List claims and copy `claim_id`.

Screenshot placeholder:

![Claims card](docs/screenshots/06-claims.png)

---

## 5.7 Binding

Purpose:

- attach evidence selectors to claims
- validate evidence coverage rules

Important:

- selectors field must be valid JSON array
- each evidence item should include more than one selector type for robust validation

### A. Quantitative binding example

For a quantitative claim, add evidence with roles:

1. `value-cell`
2. `row-header`
3. `column-header`
4. `caption`

For `value-cell`, use selectors like:

```json
[
  {"type":"PageGeometrySelector","pageIndex":0,"boxes":[]},
  {
    "type":"TableAddressSelector",
    "table_id":"<TABLE_ID_FROM_EXTRACT_TABLES>",
    "cell_ref":{"row_index":0,"col_index":1},
    "units":{"currency":"GBP"},
    "period":{"label":"FY2024/25"}
  }
]
```

For context roles, combine `TextQuoteSelector` with a copied selector from Extraction Inspector:

```json
[
  {"type":"TextQuoteSelector","exact":"Cash at bank"},
  {"type":"TextPositionSelector","start":42,"end":54}
]
```

After adding roles, click `Validate Binding`.

### B. Narrative binding example

Use role `quote`, selectors such as:

```json
[
  {"type":"TextQuoteSelector","exact":"liquidity improved significantly"},
  {"type":"TextPositionSelector","start":120,"end":152}
]
```

Then click `Validate Binding`.

Screenshot placeholder:

![Binding card](docs/screenshots/07-binding.png)

---

## 5.8 Verification and Reports

Purpose:

- run deterministic verification
- export run reports

### A. Verify one claim

1. Enter `claim_id`.
2. Click `Verify Claim`.

### B. Verify a claim set

1. Enter claim set name.
2. Click `Verify Set`.

### C. Export report

1. Enter verification `run_id`.
2. Optional output paths:
   - JSON output
   - Markdown output
3. Click `Get Report`.

Expected:

- quantitative passes if table cell matches
- narrative passes if quote selector text matches extracted source text

Screenshot placeholder:

![Verification and Reports card](docs/screenshots/08-verification-and-reports.png)

---

## 5.9 Trace

Purpose:

- inspect provenance links in multiple directions

Available traces:

1. `Trace Claim`
2. `Trace Resource`
3. `Trace Citation`

Use this to audit:

- evidence attached to claim
- claims and references connected to resource
- resources and claims linked to a citation

Screenshot placeholder:

![Trace card](docs/screenshots/09-trace.png)

---

## 5.10 CEAPF

Purpose:

- build proposition/assertion/relation structures

Steps:

1. Paste proposition JSON and click `Add Proposition`.
2. Use returned proposition ID in `Add Assertion`.
3. Link nodes with `Add Relation`.
4. Click `List Propositions`.

Example proposition:

```json
{"subject":"org:InstitutionZ","predicate":"ceapf:spent","object":{"amount":3.4,"unit":"GBP"}}
```

Screenshot placeholder:

![CEAPF card](docs/screenshots/10-ceapf.png)

---

## 5.11 Financial Pipeline

Purpose:

- run resumable batch ingest/extract across a directory tree

Steps:

1. Set pipeline root path.
2. Optional `max files`.
3. Set extraction timeout.
4. Choose extract enabled/skip.
5. Click `Run Financial Pass`.

Expected:

- structured pipeline stats in output
- resumable behavior through `.stemma` state/log files

Screenshot placeholder:

![Financial Pipeline card](docs/screenshots/11-financial-pipeline.png)

---

## 6. End-to-end workflows

## 6.1 Workflow A: quantitative verification

1. Init project and run doctor.
2. Ingest `report.md`.
3. Extract resource and list tables.
4. Import `claims-quant.json` as `tutorial-quant`.
5. Bind quantitative claim with required roles.
6. Validate binding.
7. Verify claim/set.
8. Export report.
9. Inspect results in Database Explorer (`verification_runs`, `verification_results`).

---

## 6.2 Workflow B: narrative verification

1. Ingest and extract source.
2. Import `claims-narrative.json` as `tutorial-narrative`.
3. Use Extraction Inspector to locate narrative text spans.
4. Copy selector JSON from segment or annotation cards.
5. Add `quote` evidence with `TextQuoteSelector` plus copied selector.
6. Validate binding.
7. Verify claim.

---

## 6.3 Workflow C: reference-linked claim provenance

1. Import BibTeX.
2. Link citation to resource digest.
3. Import claims with `source_cite_id` where relevant.
4. Bind and verify.
5. Use Trace card:
   - `Trace Citation`
   - `Trace Resource`
   - `Trace Claim`

---

## 6.4 Workflow D: batch financial pass

1. Open Financial Pipeline card.
2. Set root path to your document tree.
3. Run with extraction enabled.
4. Use Ingest/Extraction/Database Explorer cards to spot-check outputs.

---

## 7. Troubleshooting

## 7.1 "Not Found" API response in a card

- Restart the web server from current workspace.
- Confirm URL is `http://127.0.0.1:8765`.

## 7.2 Binding validation fails

- Ensure required roles are present.
- Ensure selector JSON is valid.
- Ensure each evidence item has multiple selector types where expected.

## 7.3 Quantitative verification fails (`table_not_found` or mismatch)

- Re-check `table_id` from latest extraction.
- Re-check row/column indices in `cell_ref`.
- Re-check units/period metadata if used.

## 7.4 Narrative verification fails

- Ensure `TextQuoteSelector.exact` appears in extracted source text.
- Use `Extraction Inspector -> Inspect Text`.

## 7.5 No inspector results

- Confirm resource selector is provided.
- Confirm extraction run exists.
- Increase limits.
- Remove filters (`segment_type`, `layer`, `category`) and retry.

---

## 8. API mapping quick reference

Equivalent API calls behind the UI:

- `/api/init`, `/api/doctor`
- `/api/db/tables`, `/api/db/table`
- `/api/ingest/path`, `/api/ingest/upload`, `/api/resources`
- `/api/refs/import-bib`, `/api/refs`, `/api/citations`, `/api/refs/link-resource`
- `/api/extract/run`, `/api/extract/tables`
- `/api/extract/text`, `/api/extract/segments`, `/api/extract/annotations`, `/api/extract/dump`
- `/api/claims/import`, `/api/claims`, `/api/claim-sets`
- `/api/bind/add`, `/api/bind/validate`
- `/api/verify/claim`, `/api/verify/set`
- `/api/report/verification`
- `/api/trace/claim`, `/api/trace/resource`, `/api/trace/citation`
- `/api/ceapf/proposition`, `/api/ceapf/assertion`, `/api/ceapf/relation`, `/api/ceapf/propositions`
- `/api/pipeline/financial-pass`

---

## 9. Suggested next improvements

1. Replace all screenshot placeholders with real screenshots.
2. Add one "golden sample" project archive for team onboarding.
3. Add card-specific troubleshooting screenshots (error and success states).

