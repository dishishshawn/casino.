"""SEC EDGAR filing parser.

Handles modern HTML/inline-XBRL filings and legacy SGML wrappers, extracts
plain text, and detects 8-K item numbers (notably Item 2.02, earnings).
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

# 8-K item number pattern. Captures both "Item 2.02" and "ITEM 2.02".
_ITEM_RE = re.compile(r"Item\s+(\d+\.\d+)", re.IGNORECASE)


def detect_format(raw: bytes) -> str:
    """Return one of {'sgml', 'html'}."""
    head = raw[:512].lstrip().lower()
    if head.startswith(b"<sec-document") or head.startswith(b"<submission"):
        return "sgml"
    return "html"


def extract_text_from_html(html: bytes) -> str:
    """Strip HTML/XBRL tags and normalize whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_sgml(sgml: bytes) -> str:
    """Extract embedded document text from an SGML wrapper.

    Old EDGAR submissions wrap multiple documents in <DOCUMENT>...<TEXT>...</TEXT>
    blocks. We concatenate the <TEXT> blocks and run the HTML stripper over the
    result (the embedded text is sometimes plain, sometimes nested HTML).
    """
    body = sgml.decode("utf-8", errors="replace")
    chunks = re.findall(r"<TEXT>(.*?)</TEXT>", body, flags=re.DOTALL | re.IGNORECASE)
    if not chunks:
        return re.sub(r"\s+", " ", body).strip()
    joined = " ".join(chunks).encode("utf-8", errors="replace")
    return extract_text_from_html(joined)


def parse_filing_document(raw: bytes, form_type: str) -> dict[str, object]:
    """Dispatch parser by detected format and return enriched dict.

    Returned keys:
        - text: normalized plain-text body
        - items: list of detected 8-K item numbers (only meaningful for 8-K)
        - has_item_202: bool
        - format: 'html' | 'sgml'
    """
    fmt = detect_format(raw)
    text = extract_text_from_sgml(raw) if fmt == "sgml" else extract_text_from_html(raw)
    items = detect_8k_items(text) if form_type.upper() == "8-K" else []
    return {
        "text": text,
        "items": items,
        "has_item_202": is_earnings_announcement(items),
        "format": fmt,
    }


def detect_8k_items(text: str) -> list[str]:
    """Return distinct, sorted item numbers found in an 8-K body."""
    found = {m.group(1) for m in _ITEM_RE.finditer(text)}
    return sorted(found)


def is_earnings_announcement(items: list[str]) -> bool:
    """8-K Item 2.02 = "Results of Operations and Financial Condition" (earnings)."""
    return "2.02" in items
