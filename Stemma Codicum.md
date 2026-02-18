# Stemma Codicum

Noun. A schema, resulting from the stemmatological process of recension, in which all surviving manuscripts of a given text are represented with their derivation, via intermediary hyparchetypes, from a single archetype.

Etymology: Borrowed from Latin stemma cōdicum, from stemma (“family tree”) + cōdicum (“book”, genitive plural).

## Project

A database and related code to enable the confirmation and visualisation of an audit trail from any claim through citations and sources to the origin of that claim.

The database format (this extends everything it does at the moment)
Simple Citations - connect four character bibtex format citations to UUIDs
References - full reference format (included my own UUID and four character bibtex citation, and link to local as well as remote source)
Claims - Evidence. for each claim, the path to the source (page number, row, column, table, context, etc)

### Required functionality

Maintain, update, improve, and monitor a database of citations, full references, and source material
Parse, deconstruct, analyse and categorise the source material into meaningful, citable parts.
Example usage: I say that City St George's University has £5,631,000 cash at bank as of 31 July 2025, according to their financial statements. This is my claim, and my citation chain should link to the financial statment PDF document I have in my archive, but also to table 23, on page 50 of that document, specifically the last column of the first data row of that table, in which sums are expressed in thousands of GBP. I need to own the verifiable audit trail. If I produce a report that includes this information, I need to be able to verify such claims automatically. Not just as a hedge against LLM hallucination but is a step forward in knowledge verification sciencen and practice.

I therefore need a managed archive of all sources (which could be PDF files, html page, other document types) and the breakdown of all of them using docling or similar.

### Extant Tools (that might help)

These are all the things it can do:
iterate every markdown file in the path `/Users/justin/Obsidian/Panta`. Find all the citations, which may already be my 4 character bibtex format, or may be a different bibtext format (from Google Gemini or from ChatGPT etc). Check if the citation already exists in my citations database. If the format is not four character, allocate a new citation id for it and replace the text in the markdown file, and find the bibtex full reference in the markdown file and add this to the references database. This is already partly done by `/Users/justin/code/util/citation-tools/citation_manager.py` - but the code does not work perfectly. It currently has the goals to:

    Scans markdown files, extracts BibTeX entries and citation-like
    reference material, then maintains three central files in ~/.citation-manager/:
        - id_mapping.json: Maps original cite keys to 4-character IDs
        - references.bib: Central BibTeX database of all citations
        - backlinks.json: Tracks which files cite which references (for tracing)

    Core Features:
    - Scans all .md files in Obsidian vault for citations (@key patterns)
    - Extracts and deduplicates references (DOI/URL matching)
    - Generates stable 4-character citation IDs (e.g., @Ab3X)
    - Optionally rewrites markdown files to use standardized cite IDs
    - Creates auto-placeholders for unresolved citations (with origin metadata)
    - Detects and removes unfenced/fenced BibTeX blocks before rewriting
    - Recognizes ChatGPT-style broken links (chatgpt://...)

    Safety guarantees:
    - Dry-run mode for testing all operations
    - Markdown backups created before any file modifications
    - Placeholder entries marked clearly (not confused with real references)
    - Unresolved cite keys tracked in validation output

Another tool 'source_manager.py' is supposed to take this further to find and manage the actual source document. It is supposed to:
    - Parse all entries in references.bib.
    - Attempt to locate existing local source files that match each entry.
    - Optionally download missing sources.
    - Store artifacts in an immutable archive under /Volumes/X10/data/sources.
    - Track provenance in a durable SQLite catalog (UUIDs, hashes, dates, source URLs/paths).
    - Trace a local file back to known source metadata and BibTeX.

    This is intentionally conservative:
    - No delete operations are exposed.
    - DB tables use ON DELETE RESTRICT.
    - Defensive triggers block accidental DELETE statements.
    - Object payloads are copied atomically and made read-only.

    Example usage:
    python citation-tools/source_manager.py init-db
    python citation-tools/source_manager.py sync-bib --search-dir /Volumes/X10/data
    python citation-tools/source_manager.py sync-bib --search-dir /Volumes/X10/data --download-missing --capture-web
    python citation-tools/source_manager.py scan-local --search-dir /Volumes/X10/data
    python citation-tools/source_manager.py trace-file /path/to/file.pdf --emit-bibtex --link-best-match

## The Resources

There is existing code in `/Users/justin/code/util/citation-tools` for handling citations and converting to my own four character bibtex cite ID format and assigning UUIDs and createing the database of resources.

For opening pdf documents I use IBM docling, and there should be tools for this in eg `/Users/justin/code/util/pdf-tools/extract_pdf_layout_tables_docling.py`

## Existing Data

    /Volumes/X10/data - where all archives are stored, should be manifests for all files (but they're incomplete)
    /Users/justin/Downloads - default (wrongly) for downloaded data. Only useful for finding download source
    /Users/justin/.citation-manager - references in .bib format and sqlite catalogue of references
