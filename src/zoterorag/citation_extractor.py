"""Citation extraction and bibliography parsing for scientific PDFs.

We target the common numeric bracket citation style used in many arXiv papers:
    "... encoder-decoder structure [5, 2, 35]"
with a References section like:
    "[5] Author. Title. Venue, 2016."

The main entrypoint for integration is :func:`extract_citation_metadata`, which
returns a mapping of sentence text -> resolved citation metadata. The rest of the
app stores these fields alongside sentence embeddings.

This module is deliberately independent from the rest of the pipeline so it's
unit-testable and can be reused from :class:`zoterorag.pdf_processor.PDFProcessor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

_CITATION_GROUP_RE = re.compile(r"\[(?P<body>[^]]{1,80})]")
_CITATION_BODY_OK_RE = re.compile(
    r"^\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*[,;]\s*\d+(?:\s*[-–]\s*\d+)?)*\s*$"
)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _expand_numbers(body: str) -> list[int]:
    """Parse something like '5, 2, 35' or '7-9, 12' into a sorted list."""

    body = (body or "").replace("–", "-")
    out: set[int] = set()
    for part in re.split(r"\s*[,;]\s*", body.strip()):
        if not part:
            continue
        if "-" in part:
            a, b = [p.strip() for p in part.split("-", 1)]
            if a.isdigit() and b.isdigit():
                start, end = int(a), int(b)
                if start <= end and end - start <= 2000:
                    out.update(range(start, end + 1))
            continue
        if part.strip().isdigit():
            out.add(int(part.strip()))
    return sorted(out)


def extract_citation_numbers_from_sentence(sentence: str) -> list[int]:
    """Return sorted unique citation numbers for a single sentence."""

    numbers: set[int] = set()
    for m in _CITATION_GROUP_RE.finditer(sentence or ""):
        body = m.group("body")
        if not _CITATION_BODY_OK_RE.match(body):
            continue
        numbers.update(_expand_numbers(body))
    return sorted(numbers)


def extract_page_lines(doc: fitz.Document, page_index: int) -> list[str]:
    """Extract lines in a stable reading order (handles two columns reasonably)."""

    page = doc.load_page(page_index)
    words = page.get_text("words")
    if not words:
        return []

    width = page.rect.width
    mid_x = width * 0.5
    left = [w for w in words if w[0] < mid_x]
    right = [w for w in words if w[0] >= mid_x]

    def lines_from(words_subset: list[tuple]) -> list[tuple[float, float, str]]:
        grouped: dict[tuple[int, int], list[tuple]] = {}
        for w in words_subset:
            key = (int(w[5]), int(w[6]))
            grouped.setdefault(key, []).append(w)

        lines: list[tuple[float, float, str]] = []
        for ws in grouped.values():
            ws_sorted = sorted(ws, key=lambda x: x[0])
            text = " ".join(w[4] for w in ws_sorted)
            y0 = min(w[1] for w in ws_sorted)
            x0 = min(w[0] for w in ws_sorted)
            lines.append((y0, x0, text))
        return lines

    if len(right) < 0.2 * len(left):
        all_lines = lines_from(words)
        all_lines.sort(key=lambda t: (t[0], t[1]))
        return [_normalize_ws(txt) for _, _, txt in all_lines if _normalize_ws(txt)]

    left_lines = lines_from(left)
    right_lines = lines_from(right)
    left_lines.sort(key=lambda t: (t[0], t[1]))
    right_lines.sort(key=lambda t: (t[0], t[1]))

    merged = [(0, float(y0), float(x0), str(txt)) for (y0, x0, txt) in left_lines] + [
        (1, float(y0), float(x0), str(txt)) for (y0, x0, txt) in right_lines
    ]
    merged.sort(key=lambda t: (t[0], t[1], t[2]))

    return [_normalize_ws(txt) for _col, _y0, _x0, txt in merged if _normalize_ws(txt)]


def find_references_start_page(doc: fitz.Document) -> int:
    """Return 0-based page index for the start of references section."""

    heading_re = re.compile(r"^\s*References\s*$", re.I | re.M)
    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text")
        if heading_re.search(text):
            return i

    refstart_re = re.compile(r"^\s*\[(\d{1,4})]\s+", re.M)
    for i in range(max(0, doc.page_count - 8), doc.page_count):
        text = doc.load_page(i).get_text("text")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        starts = sum(1 for ln in lines if refstart_re.match(ln))
        if lines and starts / len(lines) > 0.25:
            return i

    return max(0, doc.page_count - 1)


def parse_references(doc: fitz.Document, start_page: int) -> dict[int, str]:
    """Parse references of the form '[12] ...' into a mapping."""

    ref_line_re = re.compile(r"^\s*\[(\d{1,4})]\s+(.*)$")

    entries: dict[int, str] = {}
    cur_n: Optional[int] = None
    cur_text = ""

    def flush() -> None:
        nonlocal cur_n, cur_text
        if cur_n is not None:
            t = _normalize_ws(cur_text)
            if t:
                entries[cur_n] = t
        cur_n = None
        cur_text = ""

    for p in range(start_page, doc.page_count):
        page_text = doc.load_page(p).get_text("text")
        for raw_line in page_text.splitlines():
            line = _normalize_ws(raw_line)
            if not line:
                continue
            if line.isdigit():
                continue
            if re.match(r"^\s*References\s*$", line, re.I):
                continue

            m = ref_line_re.match(line)
            if m:
                flush()
                cur_n = int(m.group(1))
                cur_text = m.group(2)
                continue

            if cur_n is not None:
                if cur_text.endswith("-") and line and line[0].isalpha():
                    cur_text = cur_text[:-1] + line
                else:
                    cur_text += " " + line

    flush()
    return entries


def _bibtex_key_from_authors_year(authors: list[str], year: Optional[str]) -> str:
    last = "ref"
    if authors:
        first = authors[0].strip()
        if "," in first:
            last = first.split(",", 1)[0].strip()
        else:
            last = first.split()[-1].strip()
        last = re.sub(r"[^A-Za-z0-9]", "", last) or "ref"
    return f"{last}{year or ''}".lower()


def _split_authors(authors_raw: str) -> list[str]:
    s = (authors_raw or "").strip().rstrip(".")
    if not s:
        return []

    s = s.replace(" & ", " and ")
    s = re.sub(r",\s+and\s+", " and ", s)

    comma_parts = [p.strip() for p in re.split(r",\s+", s) if p.strip()]
    out: list[str] = []
    for part in comma_parts:
        part = re.sub(r"^and\s+", "", part.strip(), flags=re.I)
        if part:
            out.append(part)

    return [p for p in out if p.lower() not in {"et al", "et al."}]


def reference_text_to_bibtex(ref_text: str, number: Optional[int] = None) -> Optional[str]:
    """Convert a reference string into a minimal BibTeX entry (heuristic)."""

    txt = _normalize_ws(ref_text)
    if not txt:
        return None

    doi_m = re.search(r"\b(10\.\d{4,9}/[^\s]+)\b", txt)
    arxiv_m = re.search(r"\barXiv:(\d{4}\.\d{4,5})(?:v\d+)?\b", txt, re.I)

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", txt)
    year = years[-1] if years else None

    parts = [p.strip() for p in txt.split(".") if p.strip()]
    if len(parts) < 2:
        return None

    authors_raw = parts[0]
    title = parts[1]
    authors = _split_authors(authors_raw)

    title = re.sub(r"\s+In\s+.+$", "", title).strip().rstrip(",")

    booktitle: Optional[str] = None
    journal: Optional[str] = None
    publisher: Optional[str] = None
    pages: Optional[str] = None

    pages_m = re.search(r"\bpages\s+([0-9]+\s*[–-]\s*[0-9]+)\b", txt, re.I)
    if pages_m:
        pages = pages_m.group(1).replace(" ", "")

    in_m = re.search(
        r"\bIn\s+(.+?)(?:,\s*pages\s+[0-9]|,\s*(?:ACL|IEEE|Springer|AAAI)|,\s*(?:August|June|July)\s+\d{4}|,\s*\d{4}|\.|$)",
        txt,
    )
    if in_m:
        booktitle = _normalize_ws(in_m.group(1)).rstrip(",")
        booktitle = re.sub(r"\bProc\.\s+of\b", "Proceedings of", booktitle)
    else:
        if re.search(r"\bCoRR\b", txt):
            journal = "CoRR"

    pub_m = re.search(r"\.\s*(ACL|Curran Associates, Inc\.|IEEE)\b", txt)
    if pub_m:
        publisher = pub_m.group(1)

    if arxiv_m or ("arXiv" in txt and not booktitle and not journal):
        entry_type = "misc"
    elif booktitle:
        entry_type = "inproceedings"
    else:
        entry_type = "article"

    key = _bibtex_key_from_authors_year(authors, year)
    if number is not None:
        key = f"{key}{number}"

    fields: list[tuple[str, str]] = []
    if authors:
        fields.append(("author", " and ".join(authors)))
    if title:
        fields.append(("title", title))
    if year:
        fields.append(("year", year))

    if entry_type == "inproceedings" and booktitle:
        fields.append(("booktitle", booktitle))
    if entry_type == "article" and journal:
        fields.append(("journal", journal))

    if pages:
        fields.append(("pages", pages))
    if publisher:
        fields.append(("publisher", publisher))

    if doi_m:
        fields.append(("doi", doi_m.group(1).rstrip(".,;")))

    if arxiv_m:
        fields.append(("eprint", arxiv_m.group(1)))
        fields.append(("archivePrefix", "arXiv"))
        cls_m = re.search(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\s*\[([^]]+)]", txt)
        if cls_m:
            fields.append(("primaryClass", cls_m.group(1).strip()))

    have_author = any(k == "author" for k, _ in fields)
    have_title = any(k == "title" for k, _ in fields)
    if not (have_author and have_title):
        return None

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    body = ",\n  ".join(f"{k} = {{{esc(v)}}}" for k, v in fields)
    return f"@{entry_type}{{{key},\n  {body}\n}}"


@dataclass(frozen=True)
class CitationMetadata:
    citation_numbers: list[int]
    referenced_texts: list[str]
    referenced_bibtex: list[str]


def extract_citation_metadata(pdf_path: str | Path) -> dict[str, CitationMetadata]:
    """Extract citation metadata keyed by *exact* sentence text.

    We purposely key by sentence text because our current PDF pipeline produces
    sentence text from a different extractor (pymupdf4llm). Exact matches won't
    always succeed; the caller should treat this as best-effort.
    """

    doc = fitz.open(str(pdf_path))
    ref_start = find_references_start_page(doc)
    refs = parse_references(doc, ref_start)
    refs_bibtex: dict[int, Optional[str]] = {n: reference_text_to_bibtex(t, number=n) for n, t in refs.items()}

    splitter = re.compile(r"(?<=[.!?])\s+(?=[\"\[(]?[A-Z0-9])")

    sentence_map: dict[str, CitationMetadata] = {}
    for p in range(0, max(0, ref_start)):
        lines = extract_page_lines(doc, p)
        if not lines:
            continue
        filtered = [ln for ln in lines if not (ln.isdigit() and len(ln) <= 3)]
        text = _normalize_ws(" ".join(filtered))
        if not text:
            continue
        for sent in splitter.split(text):
            s = _normalize_ws(sent)
            if not s:
                continue
            nums = extract_citation_numbers_from_sentence(s)
            if not nums:
                continue

            referenced_texts: list[str] = []
            referenced_bibtex: list[str] = []
            for n in nums:
                if n in refs:
                    referenced_texts.append(refs[n])
                    b = refs_bibtex.get(n)
                    if b:
                        referenced_bibtex.append(b)

            sentence_map[s] = CitationMetadata(
                citation_numbers=nums,
                referenced_texts=referenced_texts,
                referenced_bibtex=referenced_bibtex,
            )

    return sentence_map

