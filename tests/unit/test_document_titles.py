from stemmacodicum.core.document_titles import derive_human_title


def test_derive_human_title_prefers_org_doc_type_year() -> None:
    title = derive_human_title(
        original_filename="bangor_uni_report_2023_24.pdf",
        source_uri="https://www.bangor.ac.uk/finance/reports/annual-report-2023-24.pdf",
        text_preview="Bangor University Annual Report and Financial Statements 2023/24",
    )
    assert "Bangor" in title
    assert "Annual Report" in title
    assert "2023/24" in title


def test_derive_human_title_uses_fallback_when_metadata_is_weak() -> None:
    title = derive_human_title(
        original_filename="3f95fe1e-03d1-4f84-9151-b95fe0c4ab52.pdf",
        source_uri=None,
        text_preview="",
        fallback_id="3f95fe1e-03d1-4f84-9151-b95fe0c4ab52",
    )
    assert title == "Document 3f95fe1e"
