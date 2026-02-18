---
title: Your Note Title
type: "article-draft        # or: zettel | project | lit-note | meeting"
created: 2024-08-26
aliases:
  - Alias1
  - Alias2
tags:
  - ai
  - higher-education
status: "draft              # draft | active | complete | archived"
priority: "medium           # high | medium | low"
project: Educators-AI-Handbook
area: consulting/ai-he
related: []
summary: One or two sentences on what this note is for.
publish: false
---
# Claim–Evidence–Argument Provenance Format (CEAPF) V0.1
There is not (at present) a single, widely adopted "one-stop" standard that takes you from a claim to a uniquely identified, machine-verifiable _piece of evidence inside an arbitrary document_, _including the surrounding context needed to disambiguate meaning_. What you _can_ do, however, is assemble something that is close to "bulletproof" by composing several well-established building blocks: fragment/segment addressing (the Web Annotation model and selector machinery), provenance (PROV), tamper-evidence and content addressing (hashes; and, if you want a Linked Data-native packaging, nanopublications and trusty identifiers), and an explicit argument layer to handle rebuttals and undercutting counter-evidence.

The fragment/selector side is well covered by the World Wide Web Consortium Web Annotation Data Model and its Selectors/States note (@SandersonCiccareseYoung2017AnnotationModel; @HermanSandersonCiccareseYoung2017SelectorsStates).

Provenance is covered by PROV-O (@LeboSahooMcGuinness2013PROVO). For PDF-specific fragment identifiers, the Internet Engineering Task Force application/pdf media type registration (RFC 8118) summarises the fragment identifier parameters that exist at the PDF level (page, named destinations, structure element IDs, annotation IDs, and so on) (@HardyMasinterMarkovicJohnsonWyatt2017RFC8118).

For serialising graph-shaped claims in a machine-friendly way, JSON-LD is the obvious "pragmatic Linked Data" choice (@KelloggChampinLongley2020JSONLD11).

If you want cryptographic attestation of claim packages, the Verifiable Credentials data model gives you an interoperable wrapping pattern (@SpornyLongleyChadwickHerman2025VCDM20).

If you want the claim packages themselves to be first-class, immutable, independently referenceable micro-publications, nanopublication patterns are a good fit (@KuhnBarbanoNagyKrauthammer2013BroadeningNanopubs; @KuhnDumontier2014TrustyURIs).

Below is a draft standard that composes these pieces into a single, auditable pathway from claim to evidence, while also accommodating semantic claims ("Author X asserts…") and counter-claims/counter-evidence.

## Draft Standard: Claim–Evidence–Argument Provenance Format (CEAPF) V0.1

CEAPF's core idea is that you should not treat "a claim" as one blob. You instead model three distinct things and link them explicitly.

1. A **proposition** is the content-level statement about the world (for example, "Institution Z spent £3.4m on X in FY2024/25", or "Policy Y will cause outcomes A, B, and C").

2. An **assertion** event is the act of some agent asserting (or denying, or speculating) that proposition at a particular place in a source (for example, "Author X asserts proposition P in the annual report").

3. An **assessment** is an agent's stance towards a proposition (for example, "I accept P with high confidence because evidence E supports it" or "I reject P because counter-evidence undermines the inference").

This separation matters because it lets you be strictly machine-verifiable about what is machine-verifiable. You can often verify, mechanically, that an assertion event occurred in a source (the quoted sentence exists at the anchored location, and the attribution metadata matches). You can sometimes verify a quantitative proposition mechanically (the anchored table cell exists, the headers/caption/units are captured, and a declared extraction-and-normalisation pipeline deterministically yields the stated value). You typically cannot verify a predictive proposition ("Policy Y will cause A/B/C") purely mechanically until later evidence arrives, but you can still model it cleanly and attach competing arguments as they appear.

### 1. Evidence Addressing (targets, Selectors, and Context envelopes)
A CEAPF evidence item identifies a _source artefact_ and a _targeted fragment within it_, using a selector stack inspired by Web Annotation (@SandersonCiccareseYoung2017AnnotationModel; @HermanSandersonCiccareseYoung2017SelectorsStates). The evidence item MUST include a content digest of the exact byte stream audited (for example, sha256), plus enough locator redundancy that an auditor can still recover the fragment if one locator becomes brittle.

For PDFs, CEAPF recognises three complementary locator families.

First, PDF-native fragment parameters such as page numbers and named destinations (useful, but usually insufficient for your "row/column/caption" requirement) (@HardyMasinterMarkovicJohnsonWyatt2017RFC8118).

Secondly, layout locators: page index plus one or more bounding boxes in page coordinates for the primary span (the number itself) and for each contextual span.

Thirdly, logical-structure locators: a path into the extracted document structure produced by your parser (for example, a stable table identifier plus cell coordinates r,c plus references to header cells, caption blocks, footnotes, and unit declarations). This is where docling-style table extraction becomes an advantage: you can treat the extracted table model as a first-class "view" of the PDF, with its own internal addressing.

CEAPF makes the "surrounding meaning" explicit by requiring a context envelope. For a numeric table claim, a conforming evidence item MUST identify, as separate anchored spans, the primary value span and the spans that establish interpretation, at minimum the row header and column header that scope the value, and the unit/currency/time-period cues (often found in captions, stub headers, or footnotes). Where these cues are absent, CEAPF forces you to encode that absence explicitly, rather than silently assuming.

### 2. From Evidence to Propositions (interpretations and inferences)
CEAPF distinguishes an anchored fragment from an interpretation of that fragment. An interpretation is a structured representation derived from the evidence target, such as "cell value = 3.4; unit = GBP; scale = million; period = 2024/25; concept = expenditure; category = X". The interpretation MUST declare how it was derived: the extraction tool identity/version/configuration, and any normalisation rules applied (for example, "values are in £000 unless stated otherwise"). Provenance for these derivations SHOULD be expressed in PROV terms (entities, activities, agents, and derivation links), so that the audit trail is not merely narrative but computable (@LeboSahooMcGuinness2013PROVO).

A proposition can then be linked either directly to an interpretation (for simple "this cell equals that value" claims), or via an explicit inference step (for anything that requires transformation, aggregation, disambiguation, or judgement). The critical design decision for counter-evidence is that CEAPF treats an inference step as an addressable object that can be challenged. Counter-evidence often does not attack the raw fragment; it attacks the _bridge_ from fragment to meaning (for example, "those figures are in £000, not £m", or "that row is 'capitalised spend', not 'total spend'"). By making inferences first-class, you can represent undercutting counter-evidence cleanly: it targets the inference node rather than merely shouting "disputed".

### 3. Representing Semantic Claims (speech Acts: "Author X asserts…")
To support "Author X asserts that Policy Y …", CEAPF uses an assertion-event object.

An assertion event MUST identify the asserting agent, the proposition asserted, the evidence target that anchors the act of assertion (typically a quoted sentence/paragraph with robust selectors), and the modality (asserts/denies/speculates/recommends/predicts). The proposition itself can be represented as a small graph (subject–predicate–object plus qualifiers such as time scope and conditions), which serialises naturally as JSON-LD (@KelloggChampinLongley2020JSONLD11). In other words, the evidence anchors the _speech act_, not necessarily the truth of the proposition. Later evidence can then support or refute the proposition without changing the historical fact that the author asserted it.

This is the point at which CEAPF becomes generalisable beyond tables: a "piece of evidence" can be a quote, an image region, a chart element, a paragraph in a policy document, or a segment of audio/video. The anchoring mechanism stays the same (source digest + selector stack + context envelope); only the selector types differ (time segments for media fragments, for example) (@TroncyMannensPfeifferVanDeursen2012MediaFragments).

### 4. Counter-claims, Counter-evidence, and Argument Structure
CEAPF represents disagreement as an argument graph connecting propositions and inference steps.

A conforming CEAPF dataset MAY include relations such as supports(P1,P2), rebuts(P1,P2) (attacks the conclusion), undercuts(P1,InferenceX) (attacks the inferential link), and qualifies(P1,P2) (narrows scope or adds conditions). These relations can be interpreted under standard abstract-argumentation semantics (attack/support graphs and acceptability), which gives you a principled way to compute "which propositions are currently warranted under a chosen policy", rather than merely listing competing quotes (@Dung1995Acceptability).

Counter-evidence is then simply evidence attached to propositions that rebut, or to claims that undercut inferences. You also gain a clean way to represent "both are true but about different scopes", because qualifying relations can encode scope repair ("P is true for 2023/24, not 2024/25", or "P is true for 'staff costs', not 'total spend'").

### 5. Packaging, Immutability, and Attestations
CEAPF is intentionally serialisation-agnostic, but it defines a canonical graph model. For exchange, it RECOMMENDS JSON-LD (because it is implementable in ordinary JSON pipelines while retaining graph semantics) (@KelloggChampinLongley2020JSONLD11).

For immutability, CEAPF RECOMMENDS content-addressed identifiers for source artefacts and for claim packages. If you want a Linked Data-native pattern, nanopublication-style packaging is a good fit: you publish a small assertion graph, a provenance graph, and publication metadata as separable named graphs, and you can use hash-based "trusty" identifiers so that the identifier itself commits to the content (@KuhnBarbanoNagyKrauthammer2013BroadeningNanopubs; @KuhnDumontier2014TrustyURIs). If you additionally want third-party attestation ("this claim package was issued by X and has not been tampered with"), you can wrap or mirror CEAPF claim packages as Verifiable Credentials, which is a standardised pattern for signed sets of claims about subjects (@SpornyLongleyChadwickHerman2025VCDM20). That does not replace your evidence anchoring; it merely signs the bundle.

## A Minimal Illustrative CEAPF Record (schematic)
The following is deliberately schematic rather than exhaustive; the standard's real work is in the mandatory separation between evidence targets, interpretations, propositions, assertion events, and arguments.

```json
{
  "ceapfVersion": "0.1",
  "source": {
    "uri": "file://example-university-annual-report-2025.pdf",
    "mediaType": "application/pdf",
    "digest": { "alg": "sha256", "value": "..." }
  },
  "evidenceTargets": [
    {
      "id": "ev:tableCell",
      "selectors": {
        "pdf": { "page": 37 },
        "layout": { "bbox": [92.1, 301.4, 140.6, 315.2] },
        "structure": { "tableId": "docling:T12", "cell": { "row": 8, "col": 3 } },
        "textQuote": { "exact": "3.4" }
      },
      "contextEnvelope": {
        "rowHeader": { "structure": { "tableId": "docling:T12", "cell": { "row": 8, "col": 0 } } },
        "columnHeader": { "structure": { "tableId": "docling:T12", "cell": { "row": 0, "col": 3 } } },
        "caption": { "structure": { "tableId": "docling:T12", "captionBlock": "cap:1" } },
        "units": { "textQuote": { "exact": "£m" } }
      }
    }
  ],
  "interpretations": [
    {
      "id": "int:expenditureX_2024_25",
      "derivedFrom": "ev:tableCell",
      "value": 3.4,
      "unit": "GBP",
      "scale": "million",
      "period": "2024/25",
      "concept": "expenditure",
      "category": "X",
      "provenance": {
        "prov:wasGeneratedBy": {
          "activity": "act:doclingExtract",
          "software": { "name": "docling", "version": "..." }
        }
      }
    }
  ],
  "propositions": [
    {
      "id": "prop:spent_3_4m_on_X_2024_25",
      "graph": {
        "subject": "org:InstitutionZ",
        "predicate": "ceapf:spent",
        "object": { "amount": 3.4, "unit": "GBP", "scale": "million" },
        "qualifiers": { "category": "X", "period": "2024/25" }
      }
    },
    {
      "id": "prop:author_asserts_policy_effects",
      "graph": {
        "subject": "person:AuthorX",
        "predicate": "ceapf:asserts",
        "object": "prop:policy_Y_causes_A_B_C"
      }
    },
    {
      "id": "prop:policy_Y_causes_A_B_C",
      "graph": {
        "subject": "policy:Y",
        "predicate": "ceapf:willCause",
        "object": ["outcome:A", "outcome:B", "outcome:C"],
        "qualifiers": { "institution": "org:InstitutionZ" }
      }
    }
  ],
  "links": [
    {
      "type": "ceapf:evidenceFor",
      "evidence": "int:expenditureX_2024_25",
      "proposition": "prop:spent_3_4m_on_X_2024_25"
    }
  ],
  "arguments": [
    {
      "id": "arg:support_spend_claim",
      "relation": "ceapf:supports",
      "from": "int:expenditureX_2024_25",
      "to": "prop:spent_3_4m_on_X_2024_25"
    }
  ]
}
```

## What CEAPF Buys You in Practice

It gives you a uniform, machine-auditable chain from a proposition to a specific, anchored fragment, plus the explicit context envelope needed to disambiguate meaning, plus an explicit inference object when "meaning" is not identical to "the glyphs on the page". It also gives you a principled way to represent semantic claims as speech acts anchored to text, while keeping the "truth of the proposition" as a separate object that can later accumulate supporting and opposing arguments. Finally, it gives you a disciplined home for counter-evidence: it is not an awkward footnote; it is a first-class rebuttal or undercutter that targets either a proposition or the inference step that links evidence to proposition.

If you implement CEAPF as a graph store (even if you author it in JSON-LD), you end up with a system where "audit" is reducible to graph traversal plus deterministic re-execution of declared extraction/normalisation steps. That is about as close as you will get to "bulletproof" without requiring every publisher to emit born-semantic PDFs in the first place.

```bibtex
@techreport{SandersonCiccareseYoung2017AnnotationModel,
  title        = {Web Annotation Data Model},
  author       = {Sanderson, Robert and Ciccarese, Paolo and Young, Benjamin},
  institution  = {{World Wide Web Consortium}},
  type         = {W3C Recommendation},
  year         = {2017},
  month        = feb,
  url          = {https://www.w3.org/TR/2017/REC-annotation-model-20170223/},
  urldate      = {2026-02-15}
}

@techreport{HermanSandersonCiccareseYoung2017SelectorsStates,
  title        = {Selectors and States},
  author       = {Herman, Ivan and Sanderson, Robert and Ciccarese, Paolo and Young, Benjamin},
  institution  = {{World Wide Web Consortium}},
  type         = {W3C Working Group Note},
  year         = {2017},
  month        = feb,
  url          = {https://www.w3.org/TR/2017/NOTE-selectors-states-20170223/},
  urldate      = {2026-02-15}
}

@techreport{LeboSahooMcGuinness2013PROVO,
  title        = {{PROV-O: The PROV Ontology}},
  author       = {Lebo, Timothy and Sahoo, Satya and McGuinness, Deborah},
  institution  = {{World Wide Web Consortium}},
  type         = {W3C Recommendation},
  year         = {2013},
  month        = apr,
  url          = {https://www.w3.org/TR/2013/REC-prov-o-20130430/},
  urldate      = {2026-02-15}
}

@techreport{HardyMasinterMarkovicJohnsonWyatt2017RFC8118,
  title        = {{RFC 8118: The application/pdf Media Type}},
  author       = {Hardy, Matthew and Masinter, Larry and Markovic, Dejan and Johnson, Duff and Wyatt, Peter},
  institution  = {{Internet Engineering Task Force}},
  type         = {RFC},
  number       = {8118},
  year         = {2017},
  month        = mar,
  doi          = {10.17487/RFC8118},
  url          = {https://datatracker.ietf.org/doc/html/rfc8118},
  urldate      = {2026-02-15}
}

@techreport{KelloggChampinLongley2020JSONLD11,
  title        = {{JSON-LD 1.1: A JSON-based Serialization for Linked Data}},
  author       = {Kellogg, Gregg and Champin, Pierre-Antoine and Longley, Dave and Sporny, Manu and Lanthaler, Markus and Lindstr{\"o}m, Niklas},
  institution  = {{World Wide Web Consortium}},
  type         = {W3C Recommendation},
  year         = {2020},
  month        = jul,
  url          = {https://www.w3.org/TR/2020/REC-json-ld11-20200716/},
  urldate      = {2026-02-15}
}

@techreport{SpornyLongleyChadwickHerman2025VCDM20,
  title        = {{Verifiable Credentials Data Model v2.0}},
  author       = {Sporny, Manu and Longley, Dave and Chadwick, David and Herman, Ivan},
  institution  = {{World Wide Web Consortium}},
  type         = {W3C Recommendation},
  year         = {2025},
  month        = may,
  url          = {https://www.w3.org/TR/2025/REC-vc-data-model-2.0-20250515/},
  urldate      = {2026-02-15}
}

@techreport{TroncyMannensPfeifferVanDeursen2012MediaFragments,
  title        = {{Media Fragments URI 1.0 (basic)}},
  author       = {Troncy, Rapha{\"e}l and Mannens, Erik and Pfeiffer, Silvia and Van Deursen, Davy},
  institution  = {{World Wide Web Consortium}},
  type         = {W3C Recommendation},
  year         = {2012},
  month        = sep,
  url          = {https://www.w3.org/TR/2012/REC-media-frags-20120925/},
  urldate      = {2026-02-15}
}

@article{Dung1995Acceptability,
  title        = {On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games},
  author       = {Dung, Phan Minh},
  journal      = {Artificial Intelligence},
  volume       = {77},
  number       = {2},
  pages        = {321--357},
  year         = {1995},
  doi          = {10.1016/0004-3702(94)00041-X},
  url          = {https://doi.org/10.1016/0004-3702(94)00041-X},
  urldate      = {2026-02-15},
  issn         = {0004-3702}
}

@incollection{KuhnBarbanoNagyKrauthammer2013BroadeningNanopubs,
  title        = {Broadening the Scope of Nanopublications},
  author       = {Kuhn, Tobias and Barbano, Fabio S. and Nagy, Mario Antonio and Krauthammer, Michael},
  booktitle    = {The Semantic Web: Semantics and Big Data},
  series       = {Lecture Notes in Computer Science},
  volume       = {7882},
  publisher    = {Springer},
  address      = {Berlin, Heidelberg},
  year         = {2013},
  doi          = {10.1007/978-3-642-38288-8_33},
  url          = {https://doi.org/10.1007/978-3-642-38288-8_33},
  urldate      = {2026-02-15}
}

@incollection{KuhnDumontier2014TrustyURIs,
  title        = {Trusty URIs: Verifiable, Immutable, and Permanent Digital Artifacts for Linked Data},
  author       = {Kuhn, Tobias and Dumontier, Michel},
  booktitle    = {The Semantic Web: Trends and Challenges},
  series       = {Lecture Notes in Computer Science},
  publisher    = {Springer},
  year         = {2014},
  doi          = {10.1007/978-3-319-07443-6_27},
  url          = {https://doi.org/10.1007/978-3-319-07443-6_27},
  urldate      = {2026-02-15}
}
```
