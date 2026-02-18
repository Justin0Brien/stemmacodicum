# Draft Standard: Claim–Evidence Binding Format (CEBF) V0.1

## Purpose and Scope
CEBF defines a machine-readable, verifiable linkage from a claim to specific evidence inside a specific immutable document artefact. It is designed for "final-form" documents such as PDFs where layout, pagination, and rendering are primary, but where the audit trail must still resolve down to a precise evidential item (for example, a particular table cell) together with enough surrounding context (row/column headers, caption, units, period, and relevant notes) to make the meaning unambiguous. CEBF is not a general citation style. It is an evidence-addressing and verification protocol.

CEBF deliberately composes existing standards rather than re-inventing them. It uses the World Wide Web Consortium Web Annotation Data Model as the envelope for linking a body (the claim) to one or more targets (the evidence), and it uses the selector pattern from Selectors and States to provide redundant ways of locating the same target (by fragment identifier, by geometry, by text quote, and by hashes). @SandersonCiccareseYoung2017AnnotationModel; @HermanSandersonCiccareseYoung2017SelectorsStates. Where the target is a PDF, CEBF permits PDF fragment identifiers as one selector type, aligned with the `application/pdf` media type registration maintained by the Internet Engineering Task Force. @HardyJohnsonBailey2017RFC8118.

## Core Design Principles
CEBF assumes that any single locator can fail. Page numbers can change across versions, text extraction can be imperfect, and coordinates can drift depending on renderer. Every CEBF evidence target therefore carries multiple independent selectors and declares a verification policy that defines what constitutes a match. The format is content-addressed: the identity of the evidence document is its cryptographic digest (or a set of digests), not its URL, DOI, or bibliographic metadata. Those metadata remain useful for discovery and display, but they do not establish identity.

CEBF also treats "meaning" as something that must be evidenced explicitly. A numeric string is not meaningful on its own; for a table cell, meaning is established by the row header path, column header path, caption/title, unit/scaling cues (for example "£000"), the accounting period, and, where needed, the nearest explanatory note. CEBF therefore models a table cell evidence item as a bundle of linked regions, not a single highlight rectangle.

## Data Model Overview
A conforming CEBF record is a JSON(-LD) document representing a Web Annotation whose body contains a structured claim, and whose target(s) contain structured evidence pointers.

The required objects are:

1. EvidenceResource: the immutable document artefact.  
	It MUST include at least one strong cryptographic digest of the exact bytes processed, and it SHOULD include retrieval metadata (source URL, retrieval timestamp) and descriptive metadata (title, publisher, year) where available.
	
2. Claim: the asserted proposition.  
	It MUST be machine-interpretable. The minimum requirement is a predicate form with explicit subject, property/predicate, value, units, and period. A free-text rendering MAY be included for human readability but is not sufficient for conformance.
	
3. EvidenceItem(s): one or more targets that jointly support the claim.  
	Each EvidenceItem MUST include redundant selectors. For table-based claims, the EvidenceItem set MUST include at least: the value cell region, the row header region(s), the column header region(s), and the caption/title region. If the unit or period is not fully determined by caption and headers, the relevant unit/period note region(s) MUST also be included.
	
4. VerificationPolicy: declared matching rules and tolerances so an auditor can reproduce the resolution deterministically.

## Identifiers and Immutability Requirements
EvidenceResource identifiers MUST be content-addressed. CEBF defines `urn:cebf:resource:<alg>:<digest>` as the canonical identifier form (for example, `urn:cebf:resource:sha256:…`). The record MAY also include other digests (for example SHA-512) and MAY include signatures, but at least one digest MUST be present.

CEBF treats different byte-identical copies as the same resource even if URLs differ. If the document bytes differ, it is a different EvidenceResource even if title and metadata are the same.

## Selectors (how Targets Are located)
Each EvidenceItem MUST provide a selector set containing at least two selector types, and SHOULD provide three or more. CEBF defines the following selector types, designed to align with the Web Annotation selector pattern. @HermanSandersonCiccareseYoung2017SelectorsStates.

PdfFragmentSelector. A string fragment identifier suitable for `application/pdf` (for example using `page=` plus optional view/region parameters where supported). This is for interoperability and human convenience, not as a sole verifier. @HardyJohnsonBailey2017RFC8118.

PageGeometrySelector. A page index (0-based) plus one or more bounding boxes or polygons in PDF user-space coordinates, with an explicit coordinate system declaration (origin, axis direction, units). This is the primary "point-to-ink" locator.

TextQuoteSelector. An exact quoted text string plus optional prefix/suffix context as stabilisers. This is particularly valuable where the PDF is tagged or text extraction is reliable.

ContentHashSelector. A digest of the normalised content expected within the geometry region (for example, the extracted text in that rectangle after declared normalisation). This provides an objective check that the region still contains the same content.

For table cells, a TableAddressSelector is additionally REQUIRED (next section). This is the semantic locator that binds layout to meaning.

## Table Cell Addressing and Meaning
CEBF defines a TableCellEvidence profile. In this profile, the claim MUST be supported by an EvidenceItem bundle with explicit roles:

- role=value-cell: the rectangle/polygon around the numeric cell itself.
	
- role=row-header: one or more rectangles/polygons covering the row label hierarchy that identifies the row meaning.
	
- role=column-header: one or more rectangles/polygons covering the column header hierarchy that identifies the column meaning (including year/period columns).
	
- role=caption: the table caption/title region that scopes the table.
	
- role=note (optional but often necessary): any nearby note that defines units, scope, exclusions, or accounting basis relevant to interpreting the cell.

CEBF then requires a TableAddressSelector object that is computed from an extracted table model (for example, the table grid and header trees produced by your extraction pipeline). This selector MUST include:

- table_id: a stable identifier for the table instance within the EvidenceResource.
	
- row_path: an ordered array of row header labels from outermost to innermost.
	
- col_path: an ordered array of column header labels from outermost to innermost.
	
- cell_ref: an optional numeric grid coordinate (row index, column index) in the extracted model, included as a convenience but not relied upon alone.
	
- units: the declared unit/scaling for the cell value (for example "GBP", scaling "1e6"), with a pointer to where that unit was evidenced (caption/header/note).
	
- period: an explicit period representation (for example `2024-08-01/2025-07-31` or `FY2024/25`), with a pointer to where it was evidenced.

The table_id MUST be derived deterministically. CEBF specifies the following default derivation, expressed informally: normalise the caption text; normalise the header texts for all header nodes in reading order; concatenate them with the page range and an approximate table bounding box (quantised to a declared tolerance); then hash the resulting canonical string with the declared algorithm. The intent is that a re-extraction of the same bytes yields the same table_id, while different tables on the same page do not collide.

## Claims and Value Normalisation
CEBF claims MUST represent the asserted value in three forms:

- value_raw: exactly as presented (for example "3.4", "3,400", "(3.4)", "£3.4m").
	
- value_parsed: a numeric value in canonical form (for example a JSON number) plus sign.
	
- value_scale: explicit scaling and currency/unit metadata (for example currency "GBP", scale_factor `1000000` where the source shows "£m").

The claim MUST also declare the intended interpretation of negatives (parentheses, minus sign) and rounding where relevant. If the table provides a "£000" cue, the value_scale MUST encode that scaling and MUST link to the specific evidence role that established it (caption/header/note).

## Verification Procedure (normative)
A CEBF verifier MUST be able to take a CEBF record and the EvidenceResource bytes and produce a deterministic result.

Resolution. For each EvidenceItem, the verifier MUST attempt to resolve all provided selectors. If a selector cannot be resolved (for example the viewer does not support a fragment identifier), that does not by itself fail verification unless the VerificationPolicy marks it as required.

Concordance. The verifier MUST check that resolved selectors agree within declared tolerances. For example, the TextQuoteSelector content should occur within the PageGeometrySelector region after declared normalisation, and the ContentHashSelector should match the region's normalised extracted content.

Semantic binding (tables). For TableCellEvidence, the verifier MUST re-extract the table model using the declared extraction provenance (tool name/version/configuration hash) or a declared "compatible extractor class", then locate the cell by TableAddressSelector (row_path, col_path, and table_id). The extracted cell's raw string and parsed value MUST match the claim's declared value within the claim's declared rounding rules. The verifier MUST also confirm that the units and period are evidenced by at least one of the specified evidence roles.

Outputs. Verification MUST return (a) pass/fail, (b) a diagnostic report listing which selectors matched and which failed, and (c) a machine-readable explanation of any mismatch (for example "value cell matched; column header path mismatch at level 2: expected '2024/25', found '2023/24'").

## Reference Serialisation (example)
The following is a minimal but conforming example for a table-cell-based claim. It uses the Web Annotation envelope and adds a CEBF namespace for the claim and evidence typing. This is illustrative; a production deployment would provide a stable JSON-LD context and vocabulary URIs.

```json
{
  "@context": [
    "http://www.w3.org/ns/anno.jsonld",
    {
      "cebf": "urn:cebf:vocab:",
      "EvidenceResource": "cebf:EvidenceResource",
      "Claim": "cebf:Claim",
      "EvidenceItem": "cebf:EvidenceItem",
      "VerificationPolicy": "cebf:VerificationPolicy",
      "selectorSet": "cebf:selectorSet",
      "role": "cebf:role",
      "resource": "cebf:resource",
      "claim": "cebf:claim",
      "evidence": "cebf:evidence",
      "provenance": "cebf:provenance",
      "tableAddress": "cebf:tableAddress"
    }
  ],
  "id": "urn:cebf:annotation:uuid:0d1f8a4d-7bd3-4f6a-9d77-6f2e2b7a6c52",
  "type": "Annotation",
  "motivation": "linking",
  "resource": {
    "type": "EvidenceResource",
    "id": "urn:cebf:resource:sha256:8b1e...f3c2",
    "mediaType": "application/pdf",
    "source": {
      "url": "https://example.ac.uk/annual-report-2025.pdf",
      "retrievedAt": "2026-02-15T10:12:03Z"
    },
    "digest": [
      { "alg": "sha256", "value": "8b1e...f3c2" }
    ],
    "metadata": {
      "title": "Annual Report and Accounts 2024/25",
      "publisher": "Example University",
      "year": 2025
    }
  },
  "claim": {
    "type": "Claim",
    "id": "urn:cebf:claim:uuid:3c4c6b2a-9c1b-4dd0-9b8b-6dfe0f3f2d10",
    "subject": { "type": "Organisation", "id": "urn:example:provider:UKPRN:12345" },
    "predicate": "cebf:spent",
    "object": { "type": "Concept", "id": "urn:example:taxonomy:cost-category:x" },
    "period": { "label": "FY2024/25" },
    "value": {
      "value_raw": "3.4",
      "value_parsed": 3.4,
      "currency": "GBP",
      "scale_factor": 1000000,
      "display": "£3.4m"
    },
    "text": "The provider spent £3.4m on X in FY2024/25."
  },
  "evidence": [
    {
      "type": "EvidenceItem",
      "id": "urn:cebf:evidence:item:uuid:6d61f56b-4a2f-4a88-9c4e-3c4d7f0b7a01",
      "role": "value-cell",
      "selectorSet": [
        { "type": "cebf:PdfFragmentSelector", "value": "page=37&highlight=350,420,420,440" },
        {
          "type": "cebf:PageGeometrySelector",
          "pageIndex": 36,
          "coordSystem": "pdf-user-space-bottom-left",
          "boxes": [ { "x0": 350.2, "y0": 420.1, "x1": 420.6, "y1": 440.3 } ]
        },
        {
          "type": "cebf:TextQuoteSelector",
          "exact": "3.4",
          "prefix": "Total spend ",
          "suffix": " "
        },
        {
          "type": "cebf:ContentHashSelector",
          "alg": "sha256",
          "normalisation": "cebf:norm:v1",
          "value": "2d3c...aa91"
        },
        {
          "type": "cebf:TableAddressSelector",
          "tableAddress": {
            "table_id": "sha256:9a77...c02b",
            "row_path": ["Expenditure", "X"],
            "col_path": ["Year", "2024/25"],
            "units": { "currency": "GBP", "scale_factor": 1000000, "evidencedByRole": "note" },
            "period": { "label": "FY2024/25", "evidencedByRole": "column-header" }
          }
        }
      ]
    },
    {
      "type": "EvidenceItem",
      "id": "urn:cebf:evidence:item:uuid:9a2b7b1b-c2f1-4c86-8a66-5c3a9c5c6c3d",
      "role": "row-header",
      "selectorSet": [
        {
          "type": "cebf:PageGeometrySelector",
          "pageIndex": 36,
          "coordSystem": "pdf-user-space-bottom-left",
          "boxes": [ { "x0": 110.0, "y0": 420.0, "x1": 320.0, "y1": 440.0 } ]
        },
        { "type": "cebf:TextQuoteSelector", "exact": "X" }
      ]
    },
    {
      "type": "EvidenceItem",
      "id": "urn:cebf:evidence:item:uuid:13ef3c1a-7a89-4bfb-9d3e-c2f5d4dfb7e9",
      "role": "column-header",
      "selectorSet": [
        {
          "type": "cebf:PageGeometrySelector",
          "pageIndex": 36,
          "coordSystem": "pdf-user-space-bottom-left",
          "boxes": [ { "x0": 350.0, "y0": 470.0, "x1": 420.0, "y1": 490.0 } ]
        },
        { "type": "cebf:TextQuoteSelector", "exact": "2024/25" }
      ]
    },
    {
      "type": "EvidenceItem",
      "id": "urn:cebf:evidence:item:uuid:8b8e9d1f-3ef0-4d7d-9c8a-0c8a7f4c2a20",
      "role": "caption",
      "selectorSet": [
        {
          "type": "cebf:PageGeometrySelector",
          "pageIndex": 36,
          "coordSystem": "pdf-user-space-bottom-left",
          "boxes": [ { "x0": 110.0, "y0": 510.0, "x1": 520.0, "y1": 540.0 } ]
        },
        { "type": "cebf:TextQuoteSelector", "exact": "Table 7: Expenditure by category" }
      ]
    },
    {
      "type": "EvidenceItem",
      "id": "urn:cebf:evidence:item:uuid:8c0f0fd7-6c1b-4e1a-9df8-67c8c5fd8d13",
      "role": "note",
      "selectorSet": [
        {
          "type": "cebf:PageGeometrySelector",
          "pageIndex": 36,
          "coordSystem": "pdf-user-space-bottom-left",
          "boxes": [ { "x0": 110.0, "y0": 390.0, "x1": 520.0, "y1": 410.0 } ]
        },
        { "type": "cebf:TextQuoteSelector", "exact": "Amounts in £m" }
      ]
    }
  ],
  "provenance": {
    "extractor": {
      "name": "docling",
      "version": "X.Y.Z",
      "configDigest": { "alg": "sha256", "value": "a3b2...91ff" }
    },
    "extractionOutputDigest": { "alg": "sha256", "value": "c01d...55aa" }
  },
  "verificationPolicy": {
    "type": "VerificationPolicy",
    "textNormalisation": "cebf:norm:v1",
    "geometryTolerance": { "units": "pdf-user-space", "maxDelta": 1.0 },
    "requiredRoles": ["value-cell", "row-header", "column-header", "caption"],
    "requiredSelectorTypesForValueCell": ["cebf:PageGeometrySelector", "cebf:TableAddressSelector"]
  }
}
```

What makes this "standard" rather than "a JSON blob you happen to like"  
The normative parts of CEBF are: content-addressed EvidenceResource identity; redundant selector sets with declared verification policy; role-based evidence bundling for tables; deterministic TableAddressSelector derivation; and explicit value normalisation with a link to where units/period are evidenced. Those requirements let an independent verifier reproduce the chain from claim to ink-on-page and then to semantic meaning, while providing useful failure modes when the evidence does not match.

If you want this to be adoptable beyond your own tooling, the next practical step is to publish: a stable JSON-LD context for `urn:cebf:vocab:`; a JSON Schema for the non-JSON-LD subset (many engineering teams will want it); and a small conformance test suite with "known-good" PDFs and expected verification outcomes. The envelope stays aligned with Web Annotation so that existing annotation tooling remains usable, while the CEBF vocabulary handles the PDF/table-specific semantics.

```bibtex
@techreport{SandersonCiccareseYoung2017AnnotationModel,
  author      = {Robert Sanderson and Paolo Ciccarese and Benjamin Young},
  title       = {Web Annotation Data Model},
  institution = {{World Wide Web Consortium}},
  type        = {W3C Recommendation},
  year        = {2017},
  month       = feb,
  day         = {23},
  url         = {https://www.w3.org/TR/annotation-model/},
  urldate     = {2026-02-15}
}

@techreport{HermanSandersonCiccareseYoung2017SelectorsStates,
  author      = {Ivan Herman and Robert Sanderson and Paolo Ciccarese and Benjamin Young},
  title       = {Selectors and States},
  institution = {{World Wide Web Consortium}},
  type        = {W3C Working Group Note},
  year        = {2017},
  month       = feb,
  day         = {23},
  url         = {https://www.w3.org/TR/selectors-states/},
  urldate     = {2026-02-15}
}

@techreport{HardyJohnsonBailey2017RFC8118,
  author      = {Matthew Hardy and Duff Johnson and Martin Bailey},
  title       = {The application/pdf Media Type},
  institution = {{Internet Engineering Task Force}},
  type        = {RFC},
  number      = {8118},
  year        = {2017},
  month       = mar,
  doi         = {10.17487/RFC8118},
  url         = {https://datatracker.ietf.org/doc/html/rfc8118},
  urldate     = {2026-02-15}
}
```
