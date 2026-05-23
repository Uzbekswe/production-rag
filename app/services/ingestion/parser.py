import asyncio
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedDocument:
    text: str            # full extracted text (Markdown from Docling)
    title: str | None
    page_count: int
    source_path: str
    file_type: str


class DocumentParser:
    """
    Wraps Docling for PDF parsing and handles plain text/markdown directly.
    Docling's converter is synchronous and CPU-bound, so we run it in a
    thread pool to avoid blocking FastAPI's async event loop.

    HTML files (.htm/.html) bypass Docling entirely — SEC EDGAR HTML dumps
    are raw HTML tables that cause Docling to hang. BeautifulSoup extracts
    clean text in seconds.
    """

    def __init__(self) -> None:
        # Lazy-import so the server starts fast — Docling loads heavy ML models
        # on first use, not at import time.
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    async def parse(self, file_path: Path, file_type: str) -> ParsedDocument:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._parse_sync, file_path, file_type
        )

    def _parse_sync(self, file_path: Path, file_type: str) -> ParsedDocument:
        if file_type in ("txt", "md"):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return ParsedDocument(
                text=text,
                title=file_path.stem,
                page_count=1,
                source_path=str(file_path),
                file_type=file_type,
            )

        if file_type in ("htm", "html"):
            return self._parse_html(file_path)

        # PDF — route through Docling
        converter = self._get_converter()
        result = converter.convert(str(file_path))
        doc = result.document

        text = doc.export_to_markdown()
        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1
        title = getattr(doc, "name", None) or file_path.stem

        return ParsedDocument(
            text=text,
            title=title,
            page_count=page_count,
            source_path=str(file_path),
            file_type=file_type,
        )

    def _parse_html(self, file_path: Path) -> ParsedDocument:
        from bs4 import BeautifulSoup

        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Remove non-content tags
        for tag in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "nav"]):
            tag.decompose()

        # Remove hidden elements — SEC EDGAR iXBRL files store all XBRL metadata
        # in display:none divs. Without this, those divs leak as garbage text chunks
        # containing namespace URLs like "http://fasb.org/us-gaap/2024#...".
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]+", " ", text)

        # Filter lines that are XBRL technical content (namespace URLs, period codes,
        # bare identifiers). These slip through even after removing hidden divs when
        # iXBRL tags are mixed into the visible body.
        clean_lines = []
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            # XBRL namespace URL (e.g. "http://fasb.org/us-gaap/2024#RevenueFromContract")
            if re.match(r"^https?://\S+#\S+$", s):
                continue
            # XBRL duration codes (P1Y, P856D, P2Q, etc.)
            if re.match(r"^P\d+[YMWDQH]$", s):
                continue
            # Bare CIK / ticker-date identifiers with no spaces ("aapl-20240928", "0000320193")
            if re.match(r"^[\w\-]{1,30}$", s) and not any(c.isspace() for c in s) and not s[0].isalpha():
                continue
            clean_lines.append(s)

        text = re.sub(r"\n{3,}", "\n\n", "\n".join(clean_lines)).strip()

        title_tag = soup.find("title")
        title = (title_tag.get_text(strip=True) if title_tag else None) or file_path.stem

        return ParsedDocument(
            text=text,
            title=title,
            page_count=1,
            source_path=str(file_path),
            file_type="html",
        )
