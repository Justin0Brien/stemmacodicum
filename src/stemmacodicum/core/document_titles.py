from __future__ import annotations

import re
from urllib.parse import urlparse


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2}(?:/\d{2,4})?)\b")
_DOC_TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Annual Report", ("annual report", "annual accounts", "annual financial statements")),
    ("Financial Statements", ("financial statement", "financial statements", "statement of accounts")),
    ("Strategic Plan", ("strategic plan", "strategy", "five year plan")),
    ("Budget", ("budget", "forecast", "financial plan")),
    ("Policy", ("policy", "guidance", "framework")),
    ("Presentation", ("presentation", "slide deck")),
    ("Research Paper", ("working paper", "journal", "paper", "study")),
]


def looks_like_uuid(value: str | None) -> bool:
    return bool(_UUID_RE.match(str(value or "").strip()))


def clean_title_candidate(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    base = re.sub(r"\?.*$", "", raw)
    base = re.sub(r"^.*[\\/]", "", base)
    base = re.sub(r"\.[a-z0-9]{2,6}$", "", base, flags=re.IGNORECASE)
    if looks_like_uuid(base):
        return ""
    cleaned = re.sub(r"^upload:", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\?.*$", "", cleaned)
    cleaned = re.sub(r"^.*[\\/]", "", cleaned)
    cleaned = re.sub(r"\.[a-z0-9]{2,6}$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or looks_like_uuid(cleaned):
        return ""
    return cleaned


def infer_year_label(*parts: str | None) -> str:
    joined = " ".join(str(p or "") for p in parts)
    matches = [match.group(1) for match in _YEAR_RE.finditer(joined)]
    if not matches:
        return ""
    ranked = sorted(
        matches,
        key=lambda value: ("/" in value, len(value)),
        reverse=True,
    )
    return ranked[0]


def infer_document_type(*parts: str | None) -> str:
    joined = " ".join(str(p or "") for p in parts).lower()
    for label, needles in _DOC_TYPE_RULES:
        if any(needle in joined for needle in needles):
            return label
    return "Document"


def infer_org_label(source_uri: str | None, title_hint: str | None = None) -> str:
    uri = str(source_uri or "").strip()
    if uri and uri.lower().startswith(("http://", "https://")):
        parsed = urlparse(uri)
        host = parsed.hostname or ""
        if host:
            host = re.sub(r"^www\.", "", host, flags=re.IGNORECASE)
            host = host.split(".")[0]
            host = host.replace("-", " ")
            host = re.sub(r"\s+", " ", host).strip()
            if host and not looks_like_uuid(host):
                return host.title()
    hint = clean_title_candidate(title_hint)
    if hint:
        tokens = [token for token in re.split(r"\s+", hint) if token]
        if tokens and len(tokens[0]) > 2:
            return " ".join(tokens[:4]).title()
    return ""


def derive_human_title(
    *,
    original_filename: str | None,
    source_uri: str | None = None,
    text_preview: str | None = None,
    fallback_id: str | None = None,
) -> str:
    filename_hint = clean_title_candidate(original_filename)
    text_hint = str(text_preview or "")[:12000]
    org = infer_org_label(source_uri, filename_hint)
    doc_type = infer_document_type(filename_hint, text_hint)
    year = infer_year_label(filename_hint, text_hint)

    if org and year:
        return f"{org} - {doc_type} - {year}"
    if org:
        return f"{org} - {doc_type}"
    if filename_hint:
        return filename_hint
    if fallback_id:
        short_id = str(fallback_id).strip()[:8]
        if short_id:
            return f"Document {short_id}"
    return "Untitled document"
