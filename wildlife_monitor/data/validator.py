"""
Image ingestion validation (Package P1, FR1).

Checks that an uploaded camera trap image is usable before it enters the
system: readable, an accepted format, large enough to detect an animal in, and
carrying a capture time. Where the surrounding metadata has no timestamp, the
EXIF block is read instead.

Validation is per file. One unreadable image never stops a batch — the caller
collects results and reports accepted and rejected files separately, which is
what the UC1 alternative flow requires.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

from wildlife_monitor.utils.validation import ValidationResult

# EXIF tag numbers: DateTimeOriginal, then the generic DateTime fallback.
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME = 306
_EXIF_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")


class ImageValidator:
    """Validates camera trap images on ingestion (FR1)."""

    def __init__(self, min_width: int = 640, min_height: int = 480,
                 accepted_formats: tuple[str, ...] = ("JPEG", "PNG")) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.accepted_formats = tuple(fmt.upper() for fmt in accepted_formats)

    @property
    def requirements(self) -> str:
        """One line describing what this validator accepts."""
        formats = ", ".join(self.accepted_formats)
        return (f"Accepted: {formats} · Minimum resolution "
                f"{self.min_width}x{self.min_height} · EXIF timestamps read "
                f"when present.")

    def validate(self, source: str | Path | BinaryIO,
                 name: str = "") -> ValidationResult:
        """Validate one image, given a path or an open file object."""
        label = name or (Path(str(source)).name
                          if isinstance(source, (str, Path)) else "uploaded file")
        try:
            image = Image.open(source)
            width, height = image.size
            image_format = (image.format or "?").upper()
        except Exception as error:
            return ValidationResult.fail(f"Unreadable file: {error}", name=label)

        if image_format not in self.accepted_formats:
            return ValidationResult.fail(
                f"Unsupported format: {image_format}", name=label,
                format=image_format)

        if width < self.min_width or height < self.min_height:
            return ValidationResult.fail(
                f"Resolution {width}x{height} below minimum "
                f"{self.min_width}x{self.min_height}",
                name=label, width=width, height=height, format=image_format)

        timestamp = self.extract_timestamp(image)
        return ValidationResult.ok(
            name=label, width=width, height=height, format=image_format,
            resolution=f"{width}x{height}",
            timestamp=timestamp.isoformat(sep=" ") if timestamp else "",
            has_timestamp=timestamp is not None)

    def extract_timestamp(self, source: str | Path | Image.Image
                          ) -> datetime | None:
        """Capture time from EXIF, or None when the image carries none."""
        try:
            image = (source if isinstance(source, Image.Image)
                     else Image.open(source))
            exif = image.getexif()
        except Exception:
            return None

        raw = exif.get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
        if not raw:
            return None
        for fmt in _EXIF_FORMATS:
            try:
                return datetime.strptime(str(raw).strip(), fmt)
            except (ValueError, TypeError):
                continue
        return None

    def validate_batch(self, files: list[Any]
                        ) -> tuple[list[ValidationResult], list[ValidationResult]]:
        """Validate many files, returning ``(accepted, rejected)``.

        Each file is handled independently so a single bad image cannot abort
        the batch.
        """
        accepted: list[ValidationResult] = []
        rejected: list[ValidationResult] = []
        for file in files:
            name = getattr(file, "name", "")
            result = self.validate(file, name=name)
            (accepted if result.valid else rejected).append(result)
        return accepted, rejected
