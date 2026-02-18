from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BibTeXEntry:
    entry_type: str
    cite_key: str
    fields: dict[str, str]
    raw_entry: str


class BibTeXParserError(ValueError):
    pass


def parse_bibtex(text: str) -> list[BibTeXEntry]:
    entries: list[BibTeXEntry] = []
    idx = 0

    while True:
        start = text.find("@", idx)
        if start == -1:
            break

        entry, end_idx = _parse_entry(text, start)
        if entry is not None:
            entries.append(entry)
        idx = end_idx

    return entries


def _parse_entry(text: str, start_idx: int) -> tuple[BibTeXEntry | None, int]:
    i = start_idx + 1
    n = len(text)

    while i < n and text[i].isspace():
        i += 1

    type_start = i
    while i < n and (text[i].isalpha() or text[i] in "-_"):
        i += 1

    if i == type_start:
        return None, start_idx + 1

    entry_type = text[type_start:i].strip().lower()

    while i < n and text[i].isspace():
        i += 1

    if i >= n or text[i] != "{":
        return None, start_idx + 1

    body_start = i + 1
    i += 1
    depth = 1

    while i < n and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1

    if depth != 0:
        raise BibTeXParserError("Unbalanced braces in BibTeX entry")

    body = text[body_start : i - 1]
    raw_entry = text[start_idx:i]

    comma_idx = body.find(",")
    if comma_idx == -1:
        raise BibTeXParserError("BibTeX entry missing key separator comma")

    cite_key = body[:comma_idx].strip()
    if not cite_key:
        raise BibTeXParserError("BibTeX entry missing cite key")

    fields = _parse_fields(body[comma_idx + 1 :])
    return (
        BibTeXEntry(
            entry_type=entry_type,
            cite_key=cite_key,
            fields=fields,
            raw_entry=raw_entry,
        ),
        i,
    )


def _parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    i = 0
    n = len(text)

    while i < n:
        while i < n and (text[i].isspace() or text[i] == ","):
            i += 1
        if i >= n:
            break

        name_start = i
        while i < n and (text[i].isalnum() or text[i] in "-_"):
            i += 1
        name = text[name_start:i].strip().lower()

        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != "=":
            while i < n and text[i] != ",":
                i += 1
            continue

        i += 1
        while i < n and text[i].isspace():
            i += 1

        value, i = _parse_value(text, i)
        if name:
            fields[name] = value.strip()

    return fields


def _parse_value(text: str, start_idx: int) -> tuple[str, int]:
    i = start_idx
    n = len(text)
    if i >= n:
        return "", i

    ch = text[i]
    if ch == "{":
        i += 1
        depth = 1
        value_start = i
        while i < n and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            raise BibTeXParserError("Unbalanced braces in BibTeX field value")
        return text[value_start : i - 1], i

    if ch == '"':
        i += 1
        value_start = i
        escaped = False
        while i < n:
            current = text[i]
            if current == '"' and not escaped:
                break
            escaped = current == "\\" and not escaped
            if current != "\\":
                escaped = False
            i += 1
        if i >= n:
            raise BibTeXParserError("Unterminated quoted BibTeX field value")
        value = text[value_start:i]
        return value, i + 1

    value_start = i
    while i < n and text[i] not in ",\n":
        i += 1
    return text[value_start:i].strip(), i
