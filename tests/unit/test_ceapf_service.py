from pathlib import Path

from stemmacodicum.application.services.ceapf_service import CEAPFService
from stemmacodicum.infrastructure.db.sqlite import initialize_schema


def test_ceapf_proposition_assertion_relation_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "stemma.db"
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "stemmacodicum"
        / "infrastructure"
        / "db"
        / "schema.sql"
    )
    initialize_schema(db_path, schema_path)

    service = CEAPFService(db_path)

    proposition_id = service.create_proposition(
        {
            "subject": "org:InstitutionZ",
            "predicate": "ceapf:spent",
            "object": {"amount": 3.4, "unit": "GBP", "scale": "million"},
            "qualifiers": {"period": "FY2024/25"},
        }
    )

    assertion_id = service.create_assertion_event(
        proposition_id=proposition_id,
        asserting_agent="person:AuthorX",
        modality="asserts",
        evidence_id=None,
    )

    relation_id = service.add_argument_relation(
        relation_type="supports",
        from_node_type="assertion_event",
        from_node_id=assertion_id,
        to_node_type="proposition",
        to_node_id=proposition_id,
    )

    props = service.list_propositions()
    assert len(props) == 1
    assert props[0].id == proposition_id
    assert props[0].proposition["predicate"] == "ceapf:spent"
    assert relation_id
