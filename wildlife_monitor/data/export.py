"""
Export of detection and behaviour results (Package P1, FR8).

Converts results into the three formats the proposal promises: CSV for
spreadsheets, JSON for downstream code, and a PDF summary report for sharing.

The PDF is written by hand rather than through a reporting library. The
project's learning guide asks for core functionality to be built from the
ground up, and a summary report needs only text layout — a dependency that
pulls in a whole rendering stack would not earn its place. The writer below
emits a minimal but valid PDF 1.4 document using the standard Helvetica font,
which every reader has built in, so no font has to be embedded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

PAGE_WIDTH, PAGE_HEIGHT = 595, 842          # A4 in PDF points
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 56, 786, 56
TITLE_SIZE, HEADING_SIZE, BODY_SIZE = 18, 12, 9.5
LINE_HEIGHT = 14


def _escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string."""
    return (str(text).replace("\\", r"\\")
            .replace("(", r"\(").replace(")", r"\)"))


class _PdfWriter:
    """A minimal PDF 1.4 writer: pages of left-aligned Helvetica text."""

    def __init__(self) -> None:
        self._pages: list[list[str]] = []
        self._current: list[str] = []
        self._y = MARGIN_TOP

    def _new_page(self) -> None:
        if self._current:
            self._pages.append(self._current)
        self._current = []
        self._y = MARGIN_TOP

    def text(self, content: str, size: float = BODY_SIZE,
             bold: bool = False, gap: float = LINE_HEIGHT) -> None:
        """Write one line, starting a new page when the current one is full."""
        if self._y - gap < MARGIN_BOTTOM:
            self._new_page()
        font = "F2" if bold else "F1"
        self._current.append(
            f"BT /{font} {size} Tf {MARGIN_X} {self._y:.1f} Td "
            f"({_escape(content)}) Tj ET")
        self._y -= gap

    def spacer(self, height: float = LINE_HEIGHT / 2) -> None:
        self._y -= height

    def render(self) -> bytes:
        """Assemble the pages into a complete PDF document."""
        if self._current:
            self._pages.append(self._current)
        if not self._pages:
            self._pages = [[]]

        page_count = len(self._pages)
        # Object layout: 1 catalog, 2 pages tree, 3 font F1, 4 font F2,
        # then one page object and one content stream per page.
        first_page_object = 5
        page_ids = [first_page_object + index * 2 for index in range(page_count)]

        objects: dict[int, bytes] = {
            1: b"<< /Type /Catalog /Pages 2 0 R >>",
            2: ("<< /Type /Pages /Count %d /Kids [%s] >>" % (
                page_count,
                " ".join(f"{pid} 0 R" for pid in page_ids))).encode(),
            3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        }

        for index, lines in enumerate(self._pages):
            page_id = page_ids[index]
            content_id = page_id + 1
            stream = "\n".join(lines).encode("latin-1", "replace")
            objects[page_id] = (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {content_id} 0 R >>").encode()
            objects[content_id] = (
                b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                + stream + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n")
        offsets: dict[int, int] = {}
        for number in sorted(objects):
            offsets[number] = len(out)
            out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"

        xref_offset = len(out)
        highest = max(objects) + 1
        out += f"xref\n0 {highest}\n".encode()
        out += b"0000000000 65535 f \n"
        for number in range(1, highest):
            out += f"{offsets.get(number, 0):010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {highest} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n").encode()
        return bytes(out)


class ExportService:
    """Converts results into CSV, JSON or a PDF report (FR8)."""

    def __init__(self, project_name: str = "Wildlife Monitor") -> None:
        self.project_name = project_name

    # ── Normalisation ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_frame(records: Any) -> pd.DataFrame:
        """Accept a dataframe, a list of dataclasses, or a list of dicts."""
        if isinstance(records, pd.DataFrame):
            return records
        rows = [asdict(record) if is_dataclass(record) else dict(record)
                for record in (records or [])]
        return pd.DataFrame(rows)

    # ── Formats ───────────────────────────────────────────────────────────────
    def to_csv(self, records: Any) -> bytes:
        return self._to_frame(records).to_csv(index=False).encode()

    def to_json(self, records: Any, indent: int = 2) -> bytes:
        frame = self._to_frame(records)
        payload = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "row_count": int(len(frame)),
            "records": json.loads(frame.to_json(orient="records")) if not frame.empty else [],
        }
        return json.dumps(payload, indent=indent).encode()

    def to_pdf(self, records: Any, title: str = "Detection Report",
               summary: Sequence[tuple[str, Any]] = (),
               columns: Sequence[str] | None = None,
               max_rows: int = 40, notes: Iterable[str] = ()) -> bytes:
        """Render a summary report: heading, key figures, then a data table."""
        frame = self._to_frame(records)
        writer = _PdfWriter()

        writer.text(title, size=TITLE_SIZE, bold=True, gap=26)
        writer.text(f"{self.project_name} · generated "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}", size=BODY_SIZE)
        writer.spacer()

        if summary:
            writer.text("Summary", size=HEADING_SIZE, bold=True, gap=18)
            for label, value in summary:
                writer.text(f"{label}: {value}")
            writer.spacer()

        if not frame.empty:
            selected = [c for c in (columns or frame.columns) if c in frame.columns]
            selected = selected[:6]          # keep the table inside the margin
            writer.text(f"Records ({len(frame)} total, showing "
                        f"{min(max_rows, len(frame))})",
                        size=HEADING_SIZE, bold=True, gap=18)
            writer.text(" | ".join(str(c)[:16].ljust(16) for c in selected),
                        bold=True)
            for _, row in frame.head(max_rows).iterrows():
                writer.text(" | ".join(
                    str(row[column])[:16].ljust(16) for column in selected))
            writer.spacer()

        for line in notes:
            writer.text(line, size=BODY_SIZE)

        return writer.render()

    # ── File-writing wrappers ─────────────────────────────────────────────────
    def export_csv(self, records: Any, path: str | Path) -> Path:
        return self._write(path, self.to_csv(records))

    def export_json(self, records: Any, path: str | Path) -> Path:
        return self._write(path, self.to_json(records))

    def export_pdf(self, records: Any, path: str | Path, **kwargs) -> Path:
        return self._write(path, self.to_pdf(records, **kwargs))

    @staticmethod
    def _write(path: str | Path, payload: bytes) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path
