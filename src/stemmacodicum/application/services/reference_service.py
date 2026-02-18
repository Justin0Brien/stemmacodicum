from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stemmacodicum.core.citation import generate_cite_id, normalize_cite_key
from stemmacodicum.core.errors import ReferenceError
from stemmacodicum.core.ids import new_uuid
from stemmacodicum.core.time import now_utc_iso
from stemmacodicum.domain.models.citation import Citation
from stemmacodicum.domain.models.reference import Reference
from stemmacodicum.infrastructure.db.repos.citation_repo import CitationRepo
from stemmacodicum.infrastructure.db.repos.reference_repo import ReferenceRepo
from stemmacodicum.infrastructure.db.repos.resource_repo import ResourceRepo
from stemmacodicum.infrastructure.importers.bibtex_importer import (
    BibTeXParserError,
    parse_bibtex,
)


@dataclass(slots=True)
class BibImportSummary:
    entries_seen: int
    mappings_created: int
    references_inserted: int
    references_updated: int


class ReferenceService:
    def __init__(
        self,
        citation_repo: CitationRepo,
        reference_repo: ReferenceRepo,
        resource_repo: ResourceRepo,
    ) -> None:
        self.citation_repo = citation_repo
        self.reference_repo = reference_repo
        self.resource_repo = resource_repo

    def import_bibtex(self, bib_path: Path) -> BibImportSummary:
        path = bib_path.expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise ReferenceError(f"BibTeX file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        try:
            entries = parse_bibtex(raw)
        except BibTeXParserError as exc:
            raise ReferenceError(f"Failed to parse BibTeX: {exc}") from exc

        mappings_created = 0
        references_inserted = 0
        references_updated = 0

        for entry in entries:
            original_key = entry.cite_key.strip()
            normalized_key = normalize_cite_key(original_key)

            citation = self.citation_repo.get_by_normalized_key(normalized_key)
            if citation is None:
                citation = self._allocate_and_create_citation(original_key, normalized_key)
                mappings_created += 1

            reference = Reference(
                id=new_uuid(),
                cite_id=citation.cite_id,
                entry_type=entry.entry_type,
                title=entry.fields.get("title"),
                author=entry.fields.get("author"),
                year=entry.fields.get("year"),
                doi=entry.fields.get("doi"),
                url=entry.fields.get("url"),
                raw_bibtex=entry.raw_entry,
                imported_at=now_utc_iso(),
            )
            action = self.reference_repo.upsert(reference)
            if action == "inserted":
                references_inserted += 1
            else:
                references_updated += 1

        return BibImportSummary(
            entries_seen=len(entries),
            mappings_created=mappings_created,
            references_inserted=references_inserted,
            references_updated=references_updated,
        )

    def link_reference_to_resource(self, cite_id: str, resource_digest: str) -> None:
        reference = self.reference_repo.get_by_cite_id(cite_id)
        if reference is None:
            raise ReferenceError(f"Reference not found for cite ID: {cite_id}")

        resource = self.resource_repo.get_by_digest(resource_digest)
        if resource is None:
            raise ReferenceError(f"Resource not found for digest: {resource_digest}")

        self.reference_repo.link_to_resource(reference.id, resource.id, linked_at=now_utc_iso())

    def _allocate_and_create_citation(self, original_key: str, normalized_key: str) -> Citation:
        for attempt in range(5000):
            cite_id = generate_cite_id(normalized_key, attempt)
            existing = self.citation_repo.get_by_cite_id(cite_id)
            if existing is not None:
                if existing.normalized_key == normalized_key:
                    return existing
                continue

            citation = Citation(
                cite_id=cite_id,
                original_key=original_key,
                normalized_key=normalized_key,
                created_at=now_utc_iso(),
            )
            self.citation_repo.insert(citation)
            return citation

        raise ReferenceError(f"Unable to allocate unique citation id for key: {original_key}")
