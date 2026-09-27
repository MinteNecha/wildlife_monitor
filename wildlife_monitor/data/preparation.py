"""
Image preparation for ingestion (Package P1, extends FR1).

When an image fails validation the user has three honest options, and this
module exists to keep them distinguishable:

    1. replace the image with a better one
    2. let the system fix it, where fixing is actually possible
    3. lower the requirement, and accept the image as it is

The distinction that matters is between adjustments that preserve the
information in the photograph and one that does not.

    Converting HEIC or TIFF to JPEG      lossless in every way that matters
    Applying the EXIF rotation flag      lossless
    Shrinking an oversized image         loses nothing the models can use
    ENLARGING an undersized image        adds no information whatsoever

That last one deserves the emphasis. Enlarging a 320x240 frame to 640x480
interpolates pixels; it does not recover detail that the sensor never
captured. The project's own evaluation makes the point directly — zebra and
leopard score worst because "stripes and rosettes are exactly what gets lost
at low resolution". An enlarged image passes the resolution check and then
performs exactly as badly as before, which is worse than rejecting it,
because now nobody knows.

So upscaling is offered, because a user may genuinely prefer a flagged result
to no result, but it is recorded as cosmetic on every image it touches, and
that flag travels with the image into the database.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image, ImageOps

# Formats Pillow reads without extra packages. HEIC, which newer trail cameras
# and phones produce, needs the optional pillow-heif package; its absence is
# reported rather than raised.
READABLE_FORMATS = {"JPEG", "PNG", "TIFF", "BMP", "WEBP", "GIF", "PPM"}
OUTPUT_FORMAT = "JPEG"
OUTPUT_QUALITY = 92

# Above this, shrinking is pure gain: the vision models resize down anyway.
DEFAULT_MAX_DIMENSION = 2048


@dataclass
class Adjustment:
    """One change made to an image, and whether it preserves information."""

    kind: str
    detail: str
    lossless: bool

    def __str__(self) -> str:
        return self.detail


@dataclass
class PreparationResult:
    """What happened to one image on its way into the system."""

    name: str
    prepared: bool = False
    adjustments: list[Adjustment] = field(default_factory=list)
    width: int = 0
    height: int = 0
    original_width: int = 0
    original_height: int = 0
    image_format: str = ""
    data: bytes | None = None
    error: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.adjustments)

    @property
    def cosmetic_only(self) -> bool:
        """True when this image was enlarged — it gained size, not detail."""
        return any(not adjustment.lossless for adjustment in self.adjustments)

    @property
    def summary(self) -> str:
        if self.error:
            return self.error
        if not self.adjustments:
            return "No changes needed"
        return "; ".join(str(adjustment) for adjustment in self.adjustments)

    def to_details(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}",
            "format": self.image_format,
            "adjusted": self.changed,
            "upscaled": self.cosmetic_only,
            "adjustments": self.summary,
        }


class ImagePreparer:
    """Applies the adjustments a user has explicitly opted into.

    Nothing here runs unless asked. ``allow_upscale`` defaults to False and is
    separated from the other options precisely because it is the one that
    cannot improve a result.
    """

    def __init__(self, min_width: int = 640, min_height: int = 480,
                 convert_format: bool = True, fix_rotation: bool = True,
                 max_dimension: int | None = DEFAULT_MAX_DIMENSION,
                 allow_upscale: bool = False) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.convert_format = convert_format
        self.fix_rotation = fix_rotation
        self.max_dimension = max_dimension
        self.allow_upscale = allow_upscale

    @property
    def description(self) -> str:
        """What this preparer will do, for display next to the opt-in."""
        parts = []
        if self.convert_format:
            parts.append("convert to JPEG")
        if self.fix_rotation:
            parts.append("apply EXIF rotation")
        if self.max_dimension:
            parts.append(f"shrink above {self.max_dimension}px")
        if self.allow_upscale:
            parts.append(f"enlarge below {self.min_width}x{self.min_height} "
                          f"(adds no detail)")
        return ", ".join(parts) if parts else "no adjustments enabled"

    def prepare(self, source: str | Path | BinaryIO,
                name: str = "") -> PreparationResult:
        """Open one image, apply the enabled adjustments, return the bytes."""
        label = name or (Path(str(source)).name
                          if isinstance(source, (str, Path)) else "image")
        result = PreparationResult(name=label)

        try:
            if hasattr(source, "seek"):
                source.seek(0)
            image = Image.open(source)
            image.load()
        except Exception as error:
            result.error = f"Unreadable file: {error}"
            return result

        original_format = (image.format or "").upper()
        result.image_format = original_format
        result.original_width, result.original_height = image.size

        if original_format and original_format not in READABLE_FORMATS:
            result.error = (
                f"Unsupported format {original_format}. Convert it to JPEG "
                f"first, or install pillow-heif for HEIC support.")
            return result

        if self.fix_rotation:
            rotated = ImageOps.exif_transpose(image)
            if rotated is not None and rotated.size != image.size:
                result.adjustments.append(Adjustment(
                    "rotation", "Applied EXIF rotation", lossless=True))
            image = rotated or image

        if self.max_dimension and max(image.size) > self.max_dimension:
            before = image.size
            image.thumbnail((self.max_dimension, self.max_dimension),
                            Image.LANCZOS)
            result.adjustments.append(Adjustment(
                "downscale",
                f"Shrunk {before[0]}x{before[1]} to "
                f"{image.size[0]}x{image.size[1]}", lossless=True))

        width, height = image.size
        if self.allow_upscale and (width < self.min_width
                                   or height < self.min_height):
            scale = max(self.min_width / width, self.min_height / height)
            target = (int(round(width * scale)), int(round(height * scale)))
            image = image.resize(target, Image.LANCZOS)
            result.adjustments.append(Adjustment(
                "upscale",
                f"Enlarged {width}x{height} to {target[0]}x{target[1]} "
                f"— no detail added", lossless=False))

        if self.convert_format and original_format != OUTPUT_FORMAT:
            result.adjustments.append(Adjustment(
                "format", f"Converted {original_format or 'image'} to JPEG",
                lossless=True))

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, OUTPUT_FORMAT, quality=OUTPUT_QUALITY,
                   exif=image.info.get("exif", b""))
        result.data = buffer.getvalue()
        result.width, result.height = image.size
        result.image_format = OUTPUT_FORMAT
        result.prepared = True
        return result

    def meets_resolution(self, width: int, height: int) -> bool:
        return width >= self.min_width and height >= self.min_height

    def would_help(self, width: int, height: int,
                   image_format: str = "") -> bool:
        """True when the enabled adjustments could make this image usable."""
        if image_format and image_format.upper() not in (OUTPUT_FORMAT,):
            if self.convert_format:
                return True
        if self.max_dimension and max(width, height) > self.max_dimension:
            return True
        if not self.meets_resolution(width, height):
            return self.allow_upscale
        return False


def describe_options(min_width: int, min_height: int) -> list[tuple[str, str]]:
    """The choices to offer when images fail the resolution check.

    Returned as (title, explanation) pairs so the dashboard can present them
    without embedding the reasoning in the view layer.
    """
    return [
        ("Replace the images",
         "Upload higher-resolution versions. This is the only option that "
         "actually gives the models more to work with."),
        (f"Lower the requirement below {min_width}x{min_height}",
         "Accept the images as they are and record their true resolution. "
         "Honest, and often the right answer — the threshold is a chosen "
         "default, not a hard limit of the models."),
        ("Let the system enlarge them",
         "The images will pass the check, but enlarging invents pixels rather "
         "than recovering detail, so detection will be no better. Every "
         "enlarged image is flagged so its results can be read with that in "
         "mind."),
    ]
