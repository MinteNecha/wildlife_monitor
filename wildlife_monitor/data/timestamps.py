"""
Capture-time resolution (Package P1, FR1).

Behavioural analysis is entirely built on *when* each photograph was taken. The
hour decides diurnal versus nocturnal; the calendar month decides migratory
versus territorial. An image with no capture time contributes nothing, and an
image with a *wrong* capture time is worse than one with none, because it is
counted.

Section 2.3.2 of the design document specifies "an optional metadata CSV ...
pulls timestamps from EXIF data where the metadata does not provide them". This
module implements that fallback chain, in priority order:

1. **An annotation JSON** — a COCO Camera Traps file such as
   ``SnapshotSerengetiS01.json``. Highest priority because it is a curated
   record: it is how this project obtained Serengeti capture times in the first
   place, through ``scripts/extract_ground_truth.py``. It also carries species
   labels, which :meth:`AnnotationFile.labels` exposes so a user who has such a
   file gets verified accuracy rather than unverified predictions.
2. **A metadata CSV** — ``filename, timestamp`` and optionally a camera column.
   This is what camera management software exports (Camelot, Wildlife Insights,
   TRAPPER, eMammal), so most users who have organised their data already have
   one.
3. **EXIF** — written by the camera itself. Reliable when present, but stripped
   by many editing and transfer tools.
4. **The file name** — parsed for the patterns camera traps actually use, e.g.
   ``IMG_20240315_093000.jpg``. A last resort, and only when the name carries a
   time of day as well as a date (see below).

Two exclusions, both deliberate:

**File modification time is never used.** Copying a folder, extracting a zip or
syncing to cloud storage all reset it to the moment of the copy. It is always
present and almost always wrong, so it would replace "no timestamp" — which the
system reports honestly and the sufficiency check accounts for — with a
confident lie that puts every photograph in the same week.

**A date-only file name is rejected by default.** ``20240315.jpg`` gives a real
calendar month, which movement classification wants, but no time of day. Filling
the time in with midnight would make every such image read as nocturnal and
quietly corrupt activity classification. ``accept_date_only=True`` turns it on
for a user who only cares about seasonality, and the ingestion report says how
many images took that route.

Every resolution reports which source produced it, so the ingestion report can
state where the timestamps came from rather than only how many exist.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from wildlife_monitor.data.validator import ImageValidator

# ── Source names, used in reports and stored nowhere else ────────────────────

ANNOTATIONS = "annotation file"
METADATA = "metadata file"
EXIF = "EXIF"
FILENAME = "file name"
FILENAME_DATE = "file name (date only)"
NONE = "not found"

SOURCE_ORDER = (ANNOTATIONS, METADATA, EXIF, FILENAME, FILENAME_DATE, NONE)

# ── Timestamp text parsing ───────────────────────────────────────────────────

# Ordered by how unambiguous each format is. ISO and the colon-separated EXIF
# form come first because they cannot be misread. Slash-separated dates are
# genuinely ambiguous between day-first and month-first; day-first is tried
# first (the international convention) and month-first only catches strings
# day-first cannot parse, i.e. where the first field exceeds 12.
_TEXT_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y:%m:%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y:%m:%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%Y%m%d %H%M%S",
    "%Y%m%d_%H%M%S",
)

def _plural(count: int, singular: str, plural: str = "") -> str:
    """'1 image' / '2 images', so reports read as English rather than output."""
    word = singular if count == 1 else (plural or singular + "s")
    return f"{count:,} {word}"


def _is(count: int) -> str:
    """'is' / 'are', to agree with a count produced by :func:`_plural`."""
    return "is" if count == 1 else "are"


def _has(count: int) -> str:
    """'has' / 'have', likewise."""
    return "has" if count == 1 else "have"


def parse_timestamp(text: Any) -> datetime | None:
    """Parse a timestamp written in any of the formats cameras and tools use.

    Returns ``None`` rather than raising, because one unparseable row must not
    stop a batch of thousands.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw.lower() in {"nan", "none", "null", "na", ""}:
        return None

    # Trim fractional seconds and timezone suffixes, which no camera trap
    # workflow needs and which would otherwise fail every format above.
    cleaned = raw.replace("T", " ", 1) if "T" in raw[:11] else raw
    cleaned = re.sub(r"\s*(Z|[+-]\d{2}:?\d{2})$", "", cleaned)
    cleaned = re.sub(r"[.,]\d+$", "", cleaned).strip()

    for fmt in _TEXT_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


# ── Key matching ─────────────────────────────────────────────────────────────
#
# A metadata or annotation file names an image one way; the file on disk may be
# named another. Snapshot Serengeti is the clearest case: the JSON says
# 'S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG' and download_subset_images.py writes
# it to disk as 's1_b04_b04_r1_s1_b04_r1_pict0001.jpg'. Rather than guess one
# convention, every record is indexed under each plausible spelling of its
# name, and a lookup tries the same spellings of the file it holds.


def _variants(name: str) -> tuple[list[str], list[str]]:
    """Spellings of an image name worth indexing under.

    Returns ``(shared, scoped_only)``. The shared spellings identify the image
    on their own. The scoped-only ones — the last underscore-delimited token,
    e.g. ``PICT0001`` out of ``S1_B04_R1_PICT0001`` — repeat across sites, so
    they are indexed under the camera and never used to resolve a name without
    one.
    """
    raw = str(name or "").strip().replace("\\", "/")
    if not raw:
        return [], []

    tail = raw.rsplit("/", 1)[-1]
    if "." in tail:
        tail_stem = tail.rsplit(".", 1)[0]
        stem = raw[: len(raw) - len(tail)] + tail_stem
    else:
        tail_stem = tail
        stem = raw

    shared: list[str] = []
    for candidate in (raw, stem, tail, tail_stem,
                      raw.replace("/", "_"), stem.replace("/", "_")):
        key = candidate.lower()
        if key and key not in shared:
            shared.append(key)

    scoped: list[str] = []
    if "_" in tail_stem:
        token = tail_stem.rsplit("_", 1)[-1].lower()
        if token and token not in shared:
            scoped.append(token)
    return shared, scoped


class _Index:
    """Name-keyed lookup that tolerates differing naming conventions.

    Keys are stored both globally and scoped to a camera. The camera-scoped
    table is consulted first and holds the weaker keys as well, because a bare
    ``PICT0001`` repeats across sites. Where a shared key genuinely collides,
    the index marks it ambiguous and refuses to answer on it rather than
    picking one of the records arbitrarily.
    """

    def __init__(self) -> None:
        self._global: dict[str, Any] = {}
        self._ambiguous: set[str] = set()
        self._scoped: dict[tuple[str, str], Any] = {}
        self._collisions: set[str] = set()
        self.records = 0

    def add(self, names: str | Iterable[str], value: Any,
            camera_id: str = "") -> None:
        """Index one record under every spelling of the names given.

        Several names may describe the same record — an annotation file gives
        both an image id and a file name — so they are added together and the
        record is counted once.
        """
        if isinstance(names, str):
            names = [names]
        shared: list[str] = []
        scoped_only: list[str] = []
        for name in names:
            name_shared, name_scoped = _variants(name)
            for key in name_shared:
                if key not in shared:
                    shared.append(key)
            for key in name_scoped:
                if key not in scoped_only:
                    scoped_only.append(key)
        if not shared:
            return

        self.records += 1
        camera = str(camera_id).strip().lower()
        if camera:
            for key in shared + scoped_only:
                self._scoped[(camera, key)] = value
        for key in shared:
            if key in self._global and self._global[key] != value:
                self._ambiguous.add(key)
                self._collisions.add(shared[0])
            else:
                self._global[key] = value

    def get(self, name: str, camera_id: str = "") -> Any:
        shared, scoped_only = _variants(name)
        camera = str(camera_id).strip().lower()
        if camera:
            for key in shared + scoped_only:
                found = self._scoped.get((camera, key))
                if found is not None:
                    return found
        for key in shared:
            if key in self._ambiguous:
                continue
            found = self._global.get(key)
            if found is not None:
                return found
        return None

    def __len__(self) -> int:
        return self.records

    @property
    def ambiguous(self) -> int:
        """How many distinct names collided, not how many keys."""
        return len(self._collisions)


# ── The sources ──────────────────────────────────────────────────────────────


@dataclass
class SourceReport:
    """What one source loaded, for reporting before ingestion runs."""

    source: str
    entries: int = 0
    problems: list[str] = field(default_factory=list)
    labels: int = 0

    @property
    def usable(self) -> bool:
        return self.entries > 0

    def summary_line(self) -> str:
        if not self.entries:
            reason = self.problems[0] if self.problems else "no usable rows"
            return f"{self.source}: nothing loaded — {reason}"
        line = f"{self.source}: {_plural(self.entries, 'capture time')}"
        if self.labels:
            line += f", {_plural(self.labels, 'species label')}"
        if self.problems:
            line += f" ({_plural(len(self.problems), 'problem')})"
        return line


class AnnotationFile:
    """Capture times and species labels from a COCO Camera Traps JSON.

    Reads both shapes the format allows. ``images[]`` entries normally carry
    ``file_name``, ``datetime`` and ``location``; Snapshot Serengeti also
    repeats ``datetime`` on each ``annotations[]`` entry keyed by ``image_id``,
    which is where ``scripts/extract_ground_truth.py`` reads it from. Either
    alone is enough.

    Species labels come from ``annotations[].category_id`` resolved through
    ``categories[]``. They are optional: a user who wants only timestamps
    ignores :meth:`labels`.
    """

    source = ANNOTATIONS

    # Categories that are not species. Serengeti uses all three.
    _NON_SPECIES = {"empty", "human", "vehicle", "unidentifiable", "unknown",
                    "blank", "no animal", "nothing"}

    def __init__(self, document: dict | None = None) -> None:
        self._times = _Index()
        self._labels = _Index()
        self.report = SourceReport(self.source)
        if document is not None:
            self._load(document)

    @classmethod
    def load(cls, source: str | Path | BinaryIO) -> AnnotationFile:
        """Read from a path or an uploaded file object."""
        instance = cls()
        try:
            if hasattr(source, "read"):
                source.seek(0)                                # type: ignore[union-attr]
                raw = source.read()                           # type: ignore[union-attr]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                document = json.loads(text)
            else:
                with open(source, encoding="utf-8") as handle:
                    document = json.load(handle)
        except FileNotFoundError:
            instance.report.problems.append("file not found")
            return instance
        except json.JSONDecodeError as error:
            instance.report.problems.append(f"not valid JSON: {error}")
            return instance
        except Exception as error:                            # pragma: no cover
            instance.report.problems.append(f"could not be read: {error}")
            return instance

        if not isinstance(document, dict):
            instance.report.problems.append(
                "expected a COCO Camera Traps object with 'images' and "
                "'annotations'")
            return instance

        instance._load(document)
        return instance

    def _load(self, document: dict) -> None:
        categories = {}
        for category in document.get("categories") or []:
            if isinstance(category, dict) and "id" in category:
                categories[category["id"]] = str(
                    category.get("name", "")).strip().lower()

        # Pass 1 — images[], which carries file_name and location.
        locations: dict[str, str] = {}
        file_names: dict[str, str] = {}
        for image in document.get("images") or []:
            if not isinstance(image, dict):
                continue
            identifier = str(image.get("id") or image.get("image_id") or "")
            name = str(image.get("file_name") or image.get("filename") or "")
            camera = str(image.get("location") or image.get("camera_id") or "")
            if identifier and camera:
                locations[identifier] = camera
            if identifier and name:
                file_names[identifier] = name
            moment = parse_timestamp(image.get("datetime")
                                     or image.get("date_captured"))
            if moment is None:
                continue
            names = [key for key in (name, identifier) if key]
            if names:
                self._times.add(names, moment, camera)

        # Pass 2 — annotations[], keyed by image_id. Serengeti stores the
        # capture time here as well, and this is the only place species is.
        for annotation in document.get("annotations") or []:
            if not isinstance(annotation, dict):
                continue
            identifier = str(annotation.get("image_id") or "")
            if not identifier:
                continue
            camera = locations.get(identifier,
                                    str(annotation.get("location") or ""))
            names = [key for key in (identifier, file_names.get(identifier, ""))
                     if key]

            moment = parse_timestamp(annotation.get("datetime"))
            if moment is not None and self._times.get(identifier, camera) is None:
                self._times.add(names, moment, camera)

            species = categories.get(annotation.get("category_id"))
            if species and species not in self._NON_SPECIES:
                self._labels.add(names, species.replace(" ", ""), camera)

        self.report.entries = len(self._times)
        self.report.labels = len(self._labels)
        if not self.report.entries:
            self.report.problems.append(
                "no 'datetime' field found on any image or annotation")
        if self._times.ambiguous:
            self.report.problems.append(
                f"{_plural(self._times.ambiguous, 'image name')} "
                f"{_is(self._times.ambiguous)} used for more than one capture "
                f"time and was skipped as ambiguous")

    def lookup(self, name: str, camera_id: str = "",
               path: Path | None = None) -> datetime | None:
        return self._times.get(name, camera_id)

    def label_for(self, name: str, camera_id: str = "") -> str | None:
        """The annotated species for one image, or None when unlabelled."""
        return self._labels.get(name, camera_id)

    @property
    def has_labels(self) -> bool:
        return len(self._labels) > 0


class MetadataFile:
    """Capture times from a ``filename, timestamp`` CSV.

    Column names vary between camera management tools, so a range of spellings
    is accepted for each of the three columns that matter. An optional camera
    column disambiguates file names that repeat across sites.
    """

    source = METADATA

    _NAME_COLUMNS = ("filename", "file_name", "file", "image", "image_id",
                     "image_name", "name", "path", "file_path", "relativepath")
    _TIME_COLUMNS = ("timestamp", "datetime", "date_time", "captured_at",
                     "capture_time", "real_datetime", "datetimeoriginal",
                     "date_captured", "date", "time")
    _CAMERA_COLUMNS = ("camera_id", "camera", "site", "site_name", "location",
                       "station", "deployment", "deployment_id")
    _SPECIES_COLUMNS = ("species", "species_label", "ground_truth",
                        "ground_truth_species", "label", "common_name")

    def __init__(self) -> None:
        self._times = _Index()
        self._labels = _Index()
        self.report = SourceReport(self.source)

    @classmethod
    def load(cls, source: str | Path | BinaryIO) -> MetadataFile:
        instance = cls()
        try:
            if hasattr(source, "read"):
                source.seek(0)                                # type: ignore[union-attr]
                raw = source.read()                           # type: ignore[union-attr]
                text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
                rows = list(csv.DictReader(text.splitlines()))
            else:
                with open(source, encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
        except FileNotFoundError:
            instance.report.problems.append("file not found")
            return instance
        except Exception as error:
            instance.report.problems.append(f"could not be read: {error}")
            return instance

        instance._load(rows)
        return instance

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> MetadataFile:
        instance = cls()
        instance._load(list(rows))
        return instance

    @staticmethod
    def _column(headers: list[str], candidates: tuple[str, ...]) -> str | None:
        normalised = {header.strip().lower().replace(" ", "_"): header
                      for header in headers if header}
        for candidate in candidates:
            if candidate in normalised:
                return normalised[candidate]
        return None

    def _load(self, rows: list[dict]) -> None:
        if not rows:
            self.report.problems.append("the file has no rows")
            return

        headers = [key for key in rows[0].keys() if key]
        name_column = self._column(headers, self._NAME_COLUMNS)
        time_column = self._column(headers, self._TIME_COLUMNS)
        camera_column = self._column(headers, self._CAMERA_COLUMNS)
        species_column = self._column(headers, self._SPECIES_COLUMNS)

        if name_column is None or time_column is None:
            missing = []
            if name_column is None:
                missing.append("a file name column (filename, image, image_id)")
            if time_column is None:
                missing.append("a timestamp column (timestamp, datetime, date)")
            self.report.problems.append("could not find " + " and ".join(missing))
            return

        unparseable = 0
        for row in rows:
            name = str(row.get(name_column) or "").strip()
            if not name:
                continue
            camera = (str(row.get(camera_column) or "").strip()
                      if camera_column else "")
            moment = parse_timestamp(row.get(time_column))
            if moment is None:
                unparseable += 1
            else:
                self._times.add(name, moment, camera)
            if species_column:
                species = str(row.get(species_column) or "").strip().lower()
                if species and species not in {"nan", "unknown", "empty", ""}:
                    self._labels.add(name, species.replace(" ", ""), camera)

        self.report.entries = len(self._times)
        self.report.labels = len(self._labels)
        if unparseable:
            self.report.problems.append(
                f"{_plural(unparseable, 'row')} had a timestamp that could not "
                f"be read")
        if not self.report.entries:
            self.report.problems.append(
                f"no readable timestamps in column '{time_column}'")
        if self._times.ambiguous:
            self.report.problems.append(
                f"{_plural(self._times.ambiguous, 'file name')} "
                f"{_is(self._times.ambiguous)} used for more than one capture "
                f"time and was skipped — add a camera column to tell them "
                f"apart")

    def lookup(self, name: str, camera_id: str = "",
               path: Path | None = None) -> datetime | None:
        return self._times.get(name, camera_id)

    def label_for(self, name: str, camera_id: str = "") -> str | None:
        return self._labels.get(name, camera_id)

    @property
    def has_labels(self) -> bool:
        return len(self._labels) > 0


class ExifTimestamps:
    """Capture time from the image's own EXIF block."""

    source = EXIF

    def __init__(self, validator: ImageValidator | None = None) -> None:
        self.validator = validator or ImageValidator()

    def lookup(self, name: str, camera_id: str = "",
               path: Any = None) -> datetime | None:
        if path is None:
            return None
        try:
            return self.validator.extract_timestamp(path)
        except Exception:
            return None


class FilenameTimestamps:
    """Capture time parsed out of the file name.

    Covers the patterns camera traps and phone cameras actually produce. A name
    must carry a time of day as well as a date unless ``accept_date_only`` is
    set — see the module docstring for why.
    """

    source = FILENAME

    # Each pattern captures year, month, day, then optionally hour, minute and
    # second. Separators are deliberately loose: _ - . or nothing at all.
    _DATETIME = re.compile(
        r"(?<!\d)(20\d{2}|19\d{2})[-_.:/]?(0[1-9]|1[0-2])[-_.:/]?"
        r"(0[1-9]|[12]\d|3[01])"
        r"[-_.:T ]{0,3}"
        r"([01]\d|2[0-3])[-_.:hH]?([0-5]\d)[-_.:mM]?([0-5]\d)?(?!\d)")
    _DATE = re.compile(
        r"(?<!\d)(20\d{2}|19\d{2})[-_.:/]?(0[1-9]|1[0-2])[-_.:/]?"
        r"(0[1-9]|[12]\d|3[01])(?!\d)")

    def __init__(self, accept_date_only: bool = False) -> None:
        self.accept_date_only = accept_date_only
        self.date_only_used = 0

    def lookup(self, name: str, camera_id: str = "",
               path: Any = None) -> datetime | None:
        stem = Path(str(name or "")).stem
        if not stem:
            return None

        match = self._DATETIME.search(stem)
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                return datetime(int(year), int(month), int(day), int(hour),
                                 int(minute), int(second or 0))
            except ValueError:
                return None

        if not self.accept_date_only:
            return None

        match = self._DATE.search(stem)
        if not match:
            return None
        try:
            moment = datetime(int(match.group(1)), int(match.group(2)),
                               int(match.group(3)))
        except ValueError:
            return None
        self.date_only_used += 1
        return moment


# ── The chain ────────────────────────────────────────────────────────────────


@dataclass
class Resolution:
    """One image's capture time and where it came from."""

    timestamp: datetime | None
    source: str

    @property
    def found(self) -> bool:
        return self.timestamp is not None

    @property
    def text(self) -> str:
        """The form stored in ``Image.captured_at``."""
        return self.timestamp.isoformat(sep=" ") if self.timestamp else ""


class TimestampResolver:
    """Tries each source in priority order and reports which one answered.

    Sources are consulted in the order given, first answer wins. EXIF and file
    name parsing are always available; an annotation JSON or metadata CSV is
    added only when the user supplies one, and then takes precedence over both,
    because a curated record beats whatever survived in the file itself.
    """

    def __init__(self, annotations: AnnotationFile | None = None,
                 metadata: MetadataFile | None = None,
                 validator: ImageValidator | None = None,
                 use_exif: bool = True, use_filename: bool = True,
                 accept_date_only: bool = False) -> None:
        self.annotations = annotations
        self.metadata = metadata
        self.filename = (FilenameTimestamps(accept_date_only)
                         if use_filename else None)

        self.sources: list[Any] = []
        if annotations is not None and annotations.report.usable:
            self.sources.append(annotations)
        if metadata is not None and metadata.report.usable:
            self.sources.append(metadata)
        if use_exif:
            self.sources.append(ExifTimestamps(validator))
        if self.filename is not None:
            self.sources.append(self.filename)

        self.counts: dict[str, int] = {}

    # -- resolution --------------------------------------------------------

    def resolve(self, name: str, camera_id: str = "",
                path: Any = None) -> Resolution:
        """The capture time for one image, and which source supplied it."""
        for source in self.sources:
            before = getattr(source, "date_only_used", None)
            moment = source.lookup(name, camera_id, path)
            if moment is None:
                continue
            label = source.source
            if (before is not None
                    and getattr(source, "date_only_used", 0) > before):
                label = FILENAME_DATE
            self.counts[label] = self.counts.get(label, 0) + 1
            return Resolution(moment, label)

        self.counts[NONE] = self.counts.get(NONE, 0) + 1
        return Resolution(None, NONE)

    def label_for(self, name: str, camera_id: str = "") -> str | None:
        """Annotated species for one image, if any supplied source has one.

        Timestamps are the point of this class; species labels come free with
        the two file-based sources and turn a user's predictions from
        unverified into measurable, so they are surfaced rather than discarded.
        """
        for source in (self.annotations, self.metadata):
            if source is None or not getattr(source, "has_labels", False):
                continue
            label = source.label_for(name, camera_id)
            if label:
                return label
        return None

    @property
    def has_labels(self) -> bool:
        return any(getattr(source, "has_labels", False)
                   for source in (self.annotations, self.metadata)
                   if source is not None)

    # -- reporting ---------------------------------------------------------

    @property
    def chain(self) -> list[str]:
        """The source names in the order they will be tried."""
        return [source.source for source in self.sources]

    def describe_chain(self) -> str:
        """One line naming the order sources will be tried in."""
        if not self.sources:
            return "No timestamp sources are enabled."
        return "Capture times are taken from: " + " → ".join(self.chain) + "."

    def source_reports(self) -> list[SourceReport]:
        """What each supplied file loaded, whether or not it is usable."""
        reports = []
        for source in (self.annotations, self.metadata):
            if source is not None:
                reports.append(source.report)
        return reports

    def breakdown(self) -> list[tuple[str, int]]:
        """Resolved counts per source, in chain order, omitting zeroes."""
        return [(name, self.counts[name]) for name in SOURCE_ORDER
                if self.counts.get(name)]

    def breakdown_line(self) -> str:
        """Plain-English account of where the capture times came from."""
        parts = self.breakdown()
        if not parts:
            return "No images were processed."
        found = [f"{count:,} from {name}" for name, count in parts
                 if name != NONE]
        missing = dict(parts).get(NONE, 0)
        line = "Capture times: " + ", ".join(found) if found else "No capture times found"
        if missing:
            line += f"; {missing:,} with no capture time at all"
        return line + "."

    def warnings(self) -> list[str]:
        """Things the user should know about how the times were obtained."""
        notes: list[str] = []
        parts = dict(self.breakdown())

        if parts.get(FILENAME_DATE):
            notes.append(
                f"{_plural(parts[FILENAME_DATE], 'capture time')} came from a "
                f"date in the file name with no time of day, so "
                f"{'it is' if parts[FILENAME_DATE] == 1 else 'they are'} "
                f"recorded at midnight. Activity timing (day/night) will be "
                f"wrong for those images; seasonal and movement results are "
                f"not affected.")
        if parts.get(NONE):
            note = (f"{_plural(parts[NONE], 'image')} {_has(parts[NONE])} no "
                    f"capture time from any source. Such images are stored and "
                    f"can still be classified, but they contribute nothing to "
                    f"behavioural analysis.")
            if self.metadata is None and self.annotations is None:
                note += (" A metadata CSV of file names and timestamps would "
                         "recover them.")
            else:
                note += (" Check that the file names in your metadata match "
                         "the photographs on disk.")
            notes.append(note)
        for report in self.source_reports():
            for problem in report.problems:
                notes.append(f"{report.source}: {problem}")
        return notes


def build_resolver(annotations: str | Path | BinaryIO | None = None,
                   metadata: str | Path | BinaryIO | None = None,
                   validator: ImageValidator | None = None,
                   accept_date_only: bool = False) -> TimestampResolver:
    """Load whichever files were supplied and assemble the chain.

    A file that fails to load does not stop ingestion: its problems land in
    ``report.problems`` and the chain carries on with the sources that work.
    """
    annotation_file = (AnnotationFile.load(annotations)
                       if annotations is not None else None)
    metadata_file = (MetadataFile.load(metadata)
                     if metadata is not None else None)
    return TimestampResolver(annotations=annotation_file,
                             metadata=metadata_file,
                             validator=validator,
                             accept_date_only=accept_date_only)
