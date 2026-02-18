from __future__ import annotations

from pathlib import Path

from stemmacodicum.domain.models.extraction import (
    AnnotationRelation,
    AnnotationSpan,
    DocumentText,
    ExtractedTable,
    ExtractionRun,
    TextAnnotation,
    TextSegment,
)
from stemmacodicum.infrastructure.db.sqlite import get_connection


class ExtractionRepo:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def insert_run(self, run: ExtractionRun) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO extraction_runs (
                    id,
                    resource_id,
                    parser_name,
                    parser_version,
                    config_digest,
                    output_digest,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.resource_id,
                    run.parser_name,
                    run.parser_version,
                    run.config_digest,
                    run.output_digest,
                    run.status,
                    run.created_at,
                ),
            )
            conn.commit()

    def insert_table(self, table: ExtractedTable) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO extracted_tables (
                    id,
                    extraction_run_id,
                    resource_id,
                    table_id,
                    page_index,
                    caption,
                    row_headers_json,
                    col_headers_json,
                    cells_json,
                    bbox_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table.id,
                    table.extraction_run_id,
                    table.resource_id,
                    table.table_id,
                    table.page_index,
                    table.caption,
                    table.row_headers_json,
                    table.col_headers_json,
                    table.cells_json,
                    table.bbox_json,
                    table.created_at,
                ),
            )
            conn.commit()

    def insert_document_text(self, document_text: DocumentText) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO document_texts (
                    id,
                    extraction_run_id,
                    resource_id,
                    text_content,
                    text_digest_sha256,
                    char_count,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_text.id,
                    document_text.extraction_run_id,
                    document_text.resource_id,
                    document_text.text_content,
                    document_text.text_digest_sha256,
                    document_text.char_count,
                    document_text.created_at,
                ),
            )
            conn.commit()

    def insert_text_segment(self, segment: TextSegment) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO text_segments (
                    id,
                    document_text_id,
                    extraction_run_id,
                    resource_id,
                    segment_type,
                    start_offset,
                    end_offset,
                    page_index,
                    order_index,
                    bbox_json,
                    attrs_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.id,
                    segment.document_text_id,
                    segment.extraction_run_id,
                    segment.resource_id,
                    segment.segment_type,
                    segment.start_offset,
                    segment.end_offset,
                    segment.page_index,
                    segment.order_index,
                    segment.bbox_json,
                    segment.attrs_json,
                    segment.created_at,
                ),
            )
            conn.commit()

    def insert_text_annotation(self, annotation: TextAnnotation) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO text_annotations (
                    id,
                    document_text_id,
                    extraction_run_id,
                    resource_id,
                    layer,
                    category,
                    label,
                    confidence,
                    source,
                    attrs_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation.id,
                    annotation.document_text_id,
                    annotation.extraction_run_id,
                    annotation.resource_id,
                    annotation.layer,
                    annotation.category,
                    annotation.label,
                    annotation.confidence,
                    annotation.source,
                    annotation.attrs_json,
                    annotation.created_at,
                ),
            )
            conn.commit()

    def insert_annotation_span(self, span: AnnotationSpan) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO text_annotation_spans (
                    id,
                    annotation_id,
                    start_offset,
                    end_offset,
                    span_order,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    span.id,
                    span.annotation_id,
                    span.start_offset,
                    span.end_offset,
                    span.span_order,
                    span.created_at,
                ),
            )
            conn.commit()

    def insert_annotation_relation(self, relation: AnnotationRelation) -> None:
        with get_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO text_annotation_relations (
                    id,
                    document_text_id,
                    extraction_run_id,
                    resource_id,
                    relation_type,
                    from_annotation_id,
                    to_annotation_id,
                    attrs_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.id,
                    relation.document_text_id,
                    relation.extraction_run_id,
                    relation.resource_id,
                    relation.relation_type,
                    relation.from_annotation_id,
                    relation.to_annotation_id,
                    relation.attrs_json,
                    relation.created_at,
                ),
            )
            conn.commit()

    def list_tables_for_resource(self, resource_id: str, limit: int = 100) -> list[ExtractedTable]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM extracted_tables
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (resource_id, limit),
            ).fetchall()
        return [self._to_table(row) for row in rows]

    def list_tables_for_run(self, extraction_run_id: str, limit: int = 1000) -> list[ExtractedTable]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM extracted_tables
                WHERE extraction_run_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (extraction_run_id, limit),
            ).fetchall()
        return [self._to_table(row) for row in rows]

    def list_recent_runs(self, resource_id: str, limit: int = 20) -> list[ExtractionRun]:
        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM extraction_runs
                WHERE resource_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (resource_id, limit),
            ).fetchall()
        return [self._to_run(row) for row in rows]

    def get_latest_run(self, resource_id: str) -> ExtractionRun | None:
        runs = self.list_recent_runs(resource_id=resource_id, limit=1)
        return runs[0] if runs else None

    def get_run_by_id(self, run_id: str) -> ExtractionRun | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM extraction_runs
                WHERE id = ?
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return self._to_run(row) if row else None

    def get_table_by_table_id(self, resource_id: str, table_id: str) -> ExtractedTable | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM extracted_tables
                WHERE resource_id = ? AND table_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (resource_id, table_id),
            ).fetchone()
        return self._to_table(row) if row else None

    def get_document_text_for_run(self, extraction_run_id: str) -> DocumentText | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT * FROM document_texts
                WHERE extraction_run_id = ?
                LIMIT 1
                """,
                (extraction_run_id,),
            ).fetchone()
        return self._to_document_text(row) if row else None

    def get_latest_document_text_for_resource(self, resource_id: str) -> DocumentText | None:
        with get_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT dt.*
                FROM document_texts dt
                INNER JOIN extraction_runs er ON er.id = dt.extraction_run_id
                WHERE dt.resource_id = ?
                ORDER BY er.created_at DESC
                LIMIT 1
                """,
                (resource_id,),
            ).fetchone()
        return self._to_document_text(row) if row else None

    def list_segments_for_document(
        self,
        document_text_id: str,
        *,
        segment_type: str | None = None,
        limit: int = 1000,
    ) -> list[TextSegment]:
        where = "WHERE document_text_id = ?"
        params: list[object] = [document_text_id]
        if segment_type:
            where += " AND segment_type = ?"
            params.append(segment_type)
        params.append(limit)

        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM text_segments
                {where}
                ORDER BY start_offset ASC, end_offset ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._to_text_segment(row) for row in rows]

    def list_segments_for_resource(
        self,
        resource_id: str,
        *,
        extraction_run_id: str | None = None,
        segment_type: str | None = None,
        limit: int = 1000,
    ) -> list[TextSegment]:
        run_id = extraction_run_id
        if run_id is None:
            run = self.get_latest_run(resource_id)
            if run is None:
                return []
            run_id = run.id

        doc_text = self.get_document_text_for_run(run_id)
        if doc_text is None:
            return []
        return self.list_segments_for_document(
            doc_text.id,
            segment_type=segment_type,
            limit=limit,
        )

    def list_annotations_for_document(
        self,
        document_text_id: str,
        *,
        layer: str | None = None,
        category: str | None = None,
        limit: int = 1000,
    ) -> list[tuple[TextAnnotation, list[AnnotationSpan]]]:
        where = "WHERE document_text_id = ?"
        params: list[object] = [document_text_id]
        if layer:
            where += " AND layer = ?"
            params.append(layer)
        if category:
            where += " AND category = ?"
            params.append(category)
        params.append(limit)

        with get_connection(self.db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM text_annotations
                {where}
                ORDER BY created_at ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
            annotations: list[tuple[TextAnnotation, list[AnnotationSpan]]] = []
            for row in rows:
                ann = self._to_text_annotation(row)
                span_rows = conn.execute(
                    """
                    SELECT *
                    FROM text_annotation_spans
                    WHERE annotation_id = ?
                    ORDER BY span_order ASC
                    """,
                    (ann.id,),
                ).fetchall()
                annotations.append((ann, [self._to_annotation_span(s) for s in span_rows]))
        return annotations

    def list_annotations_for_resource(
        self,
        resource_id: str,
        *,
        extraction_run_id: str | None = None,
        layer: str | None = None,
        category: str | None = None,
        limit: int = 1000,
    ) -> list[tuple[TextAnnotation, list[AnnotationSpan]]]:
        run_id = extraction_run_id
        if run_id is None:
            run = self.get_latest_run(resource_id)
            if run is None:
                return []
            run_id = run.id
        doc_text = self.get_document_text_for_run(run_id)
        if doc_text is None:
            return []
        return self.list_annotations_for_document(
            doc_text.id,
            layer=layer,
            category=category,
            limit=limit,
        )

    @staticmethod
    def _to_run(row) -> ExtractionRun:
        return ExtractionRun(
            id=row["id"],
            resource_id=row["resource_id"],
            parser_name=row["parser_name"],
            parser_version=row["parser_version"],
            config_digest=row["config_digest"],
            output_digest=row["output_digest"],
            status=row["status"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_table(row) -> ExtractedTable:
        return ExtractedTable(
            id=row["id"],
            extraction_run_id=row["extraction_run_id"],
            resource_id=row["resource_id"],
            table_id=row["table_id"],
            page_index=row["page_index"],
            caption=row["caption"],
            row_headers_json=row["row_headers_json"],
            col_headers_json=row["col_headers_json"],
            cells_json=row["cells_json"],
            bbox_json=row["bbox_json"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_document_text(row) -> DocumentText:
        return DocumentText(
            id=row["id"],
            extraction_run_id=row["extraction_run_id"],
            resource_id=row["resource_id"],
            text_content=row["text_content"],
            text_digest_sha256=row["text_digest_sha256"],
            char_count=row["char_count"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_text_segment(row) -> TextSegment:
        return TextSegment(
            id=row["id"],
            document_text_id=row["document_text_id"],
            extraction_run_id=row["extraction_run_id"],
            resource_id=row["resource_id"],
            segment_type=row["segment_type"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            page_index=row["page_index"],
            order_index=row["order_index"],
            bbox_json=row["bbox_json"],
            attrs_json=row["attrs_json"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_text_annotation(row) -> TextAnnotation:
        return TextAnnotation(
            id=row["id"],
            document_text_id=row["document_text_id"],
            extraction_run_id=row["extraction_run_id"],
            resource_id=row["resource_id"],
            layer=row["layer"],
            category=row["category"],
            label=row["label"],
            confidence=row["confidence"],
            source=row["source"],
            attrs_json=row["attrs_json"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _to_annotation_span(row) -> AnnotationSpan:
        return AnnotationSpan(
            id=row["id"],
            annotation_id=row["annotation_id"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            span_order=row["span_order"],
            created_at=row["created_at"],
        )
