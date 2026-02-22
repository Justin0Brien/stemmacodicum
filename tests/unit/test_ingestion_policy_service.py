from __future__ import annotations

from stemmacodicum.application.services.ingestion_policy_service import IngestionPolicyService


def test_policy_classifies_csv_as_structured_and_skips_auto_extract_vector() -> None:
    policy = IngestionPolicyService(
        structured_auto_extract=False,
        structured_auto_vector=False,
        narrative_auto_vector=True,
    )
    decision = policy.decide(media_type="text/csv", original_filename="table.csv")
    assert decision.resource_kind == "structured_data"
    assert decision.should_extract_auto is False
    assert decision.should_vector_auto is False


def test_policy_classifies_pdf_as_narrative() -> None:
    policy = IngestionPolicyService()
    decision = policy.decide(media_type="application/pdf", original_filename="report.pdf")
    assert decision.resource_kind == "narrative_document"
    assert decision.should_extract_auto is True
