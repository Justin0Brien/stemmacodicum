PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS resources (
    id TEXT PRIMARY KEY,
    digest_sha256 TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    source_uri TEXT,
    archived_relpath TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_digests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    digest_value TEXT NOT NULL,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT,
    UNIQUE(resource_id, algorithm),
    UNIQUE(algorithm, digest_value)
);

CREATE TABLE IF NOT EXISTS provenance_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    resource_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS citations (
    cite_id TEXT PRIMARY KEY CHECK(length(cite_id) = 4),
    original_key TEXT NOT NULL,
    normalized_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reference_entries (
    id TEXT PRIMARY KEY,
    cite_id TEXT NOT NULL UNIQUE,
    entry_type TEXT NOT NULL,
    title TEXT,
    author TEXT,
    year TEXT,
    doi TEXT,
    url TEXT,
    raw_bibtex TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    FOREIGN KEY (cite_id) REFERENCES citations(cite_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS reference_resources (
    reference_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY (reference_id, resource_id),
    FOREIGN KEY (reference_id) REFERENCES reference_entries(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL,
    parser_name TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    config_digest TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS extracted_tables (
    id TEXT PRIMARY KEY,
    extraction_run_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    table_id TEXT NOT NULL,
    page_index INTEGER NOT NULL,
    caption TEXT,
    row_headers_json TEXT NOT NULL,
    col_headers_json TEXT NOT NULL,
    cells_json TEXT NOT NULL,
    bbox_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS document_texts (
    id TEXT PRIMARY KEY,
    extraction_run_id TEXT NOT NULL UNIQUE,
    resource_id TEXT NOT NULL,
    text_content TEXT NOT NULL,
    text_digest_sha256 TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS text_segments (
    id TEXT PRIMARY KEY,
    document_text_id TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    segment_type TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    page_index INTEGER,
    order_index INTEGER,
    bbox_json TEXT,
    attrs_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_text_id) REFERENCES document_texts(id) ON DELETE RESTRICT,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS text_annotations (
    id TEXT PRIMARY KEY,
    document_text_id TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    category TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    source TEXT,
    attrs_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_text_id) REFERENCES document_texts(id) ON DELETE RESTRICT,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS text_annotation_spans (
    id TEXT PRIMARY KEY,
    annotation_id TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    span_order INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (annotation_id) REFERENCES text_annotations(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS text_annotation_relations (
    id TEXT PRIMARY KEY,
    document_text_id TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    from_annotation_id TEXT NOT NULL,
    to_annotation_id TEXT NOT NULL,
    attrs_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_text_id) REFERENCES document_texts(id) ON DELETE RESTRICT,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT,
    FOREIGN KEY (from_annotation_id) REFERENCES text_annotations(id) ON DELETE RESTRICT,
    FOREIGN KEY (to_annotation_id) REFERENCES text_annotations(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS claim_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    claim_set_id TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    subject TEXT,
    predicate TEXT,
    object_text TEXT,
    narrative_text TEXT,
    value_raw TEXT,
    value_parsed REAL,
    currency TEXT,
    scale_factor INTEGER,
    period_label TEXT,
    source_cite_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (claim_set_id) REFERENCES claim_sets(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_claims_claim_set_created
ON claims(claim_set_id, created_at DESC);

CREATE TABLE IF NOT EXISTS evidence_items (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL,
    role TEXT NOT NULL,
    page_index INTEGER,
    note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS evidence_selectors (
    id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    selector_type TEXT NOT NULL,
    selector_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (evidence_id) REFERENCES evidence_items(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS claim_evidence_bindings (
    claim_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (claim_id, evidence_id),
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE RESTRICT,
    FOREIGN KEY (evidence_id) REFERENCES evidence_items(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_evidence_selectors_evidence
ON evidence_selectors(evidence_id);

CREATE TABLE IF NOT EXISTS verification_runs (
    id TEXT PRIMARY KEY,
    claim_set_id TEXT,
    policy_profile TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (claim_set_id) REFERENCES claim_sets(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS verification_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES verification_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_verification_results_run
ON verification_results(run_id);

CREATE TABLE IF NOT EXISTS propositions (
    id TEXT PRIMARY KEY,
    proposition_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assertion_events (
    id TEXT PRIMARY KEY,
    proposition_id TEXT NOT NULL,
    asserting_agent TEXT NOT NULL,
    modality TEXT NOT NULL,
    evidence_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (proposition_id) REFERENCES propositions(id) ON DELETE RESTRICT,
    FOREIGN KEY (evidence_id) REFERENCES evidence_items(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS argument_relations (
    id TEXT PRIMARY KEY,
    relation_type TEXT NOT NULL,
    from_node_type TEXT NOT NULL,
    from_node_id TEXT NOT NULL,
    to_node_type TEXT NOT NULL,
    to_node_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vector_index_runs (
    id TEXT PRIMARY KEY,
    resource_id TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,
    vector_backend TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER,
    chunking_version TEXT NOT NULL,
    status TEXT NOT NULL,
    chunks_total INTEGER NOT NULL DEFAULT 0,
    chunks_indexed INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS vector_chunks (
    id TEXT PRIMARY KEY,
    vector_index_run_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    extraction_run_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    vector_point_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    page_index INTEGER,
    start_offset INTEGER,
    end_offset INTEGER,
    token_count_est INTEGER,
    embedding_dim INTEGER NOT NULL,
    vector_backend TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    text_digest_sha256 TEXT NOT NULL,
    text_content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (vector_index_run_id) REFERENCES vector_index_runs(id) ON DELETE RESTRICT,
    FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE RESTRICT,
    FOREIGN KEY (extraction_run_id) REFERENCES extraction_runs(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_extraction_runs_resource_created
ON extraction_runs(resource_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_extracted_tables_resource_table
ON extracted_tables(resource_id, table_id);

CREATE INDEX IF NOT EXISTS idx_document_texts_resource_created
ON document_texts(resource_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_text_segments_resource_type
ON text_segments(resource_id, segment_type, start_offset);

CREATE INDEX IF NOT EXISTS idx_text_annotations_resource_layer
ON text_annotations(resource_id, layer, category);

CREATE INDEX IF NOT EXISTS idx_text_annotation_spans_annotation
ON text_annotation_spans(annotation_id, span_order);

CREATE INDEX IF NOT EXISTS idx_text_annotation_relations_resource_type
ON text_annotation_relations(resource_id, relation_type);

CREATE INDEX IF NOT EXISTS idx_vector_index_runs_resource_created
ON vector_index_runs(resource_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vector_index_runs_extraction_created
ON vector_index_runs(extraction_run_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_resource_created
ON vector_chunks(resource_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_extraction_chunk
ON vector_chunks(extraction_run_id, chunk_id);

CREATE INDEX IF NOT EXISTS idx_vector_chunks_point
ON vector_chunks(vector_point_id);

CREATE TRIGGER IF NOT EXISTS block_delete_resources
BEFORE DELETE ON resources
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for resources');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_resource_digests
BEFORE DELETE ON resource_digests
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for resource_digests');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_citations
BEFORE DELETE ON citations
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for citations');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_reference_entries
BEFORE DELETE ON reference_entries
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for reference_entries');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_extraction_runs
BEFORE DELETE ON extraction_runs
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for extraction_runs');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_extracted_tables
BEFORE DELETE ON extracted_tables
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for extracted_tables');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_document_texts
BEFORE DELETE ON document_texts
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for document_texts');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_text_segments
BEFORE DELETE ON text_segments
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for text_segments');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_text_annotations
BEFORE DELETE ON text_annotations
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for text_annotations');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_text_annotation_spans
BEFORE DELETE ON text_annotation_spans
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for text_annotation_spans');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_text_annotation_relations
BEFORE DELETE ON text_annotation_relations
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for text_annotation_relations');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_claim_sets
BEFORE DELETE ON claim_sets
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for claim_sets');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_claims
BEFORE DELETE ON claims
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for claims');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_evidence_items
BEFORE DELETE ON evidence_items
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for evidence_items');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_evidence_selectors
BEFORE DELETE ON evidence_selectors
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for evidence_selectors');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_claim_evidence_bindings
BEFORE DELETE ON claim_evidence_bindings
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for claim_evidence_bindings');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_verification_runs
BEFORE DELETE ON verification_runs
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for verification_runs');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_verification_results
BEFORE DELETE ON verification_results
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for verification_results');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_propositions
BEFORE DELETE ON propositions
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for propositions');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_assertion_events
BEFORE DELETE ON assertion_events
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for assertion_events');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_argument_relations
BEFORE DELETE ON argument_relations
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for argument_relations');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_vector_index_runs
BEFORE DELETE ON vector_index_runs
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for vector_index_runs');
END;

CREATE TRIGGER IF NOT EXISTS block_delete_vector_chunks
BEFORE DELETE ON vector_chunks
BEGIN
    SELECT RAISE(ABORT, 'DELETE disabled for vector_chunks');
END;
