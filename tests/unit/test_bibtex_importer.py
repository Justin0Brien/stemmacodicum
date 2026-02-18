from stemmacodicum.infrastructure.importers.bibtex_importer import parse_bibtex


def test_parse_bibtex_extracts_entries_and_fields() -> None:
    raw = """
@article{KeyOne,
  title={A Good Paper},
  author={Ada Lovelace and Alan Turing},
  year={2024},
  doi={10.1234/example},
  url={https://example.org/paper}
}

@techreport{AnotherKey,
  title = "Report Title",
  year = 2025,
  url = {https://example.org/report}
}
"""

    entries = parse_bibtex(raw)

    assert len(entries) == 2
    assert entries[0].entry_type == "article"
    assert entries[0].cite_key == "KeyOne"
    assert entries[0].fields["title"] == "A Good Paper"
    assert entries[0].fields["year"] == "2024"

    assert entries[1].entry_type == "techreport"
    assert entries[1].cite_key == "AnotherKey"
    assert entries[1].fields["title"] == "Report Title"
    assert entries[1].fields["year"] == "2025"
