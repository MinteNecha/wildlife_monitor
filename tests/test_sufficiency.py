"""
Data sufficiency assessment, and the tri-state verification it depends on.

The first test is the one that matters. `classify_movement` compares each
camera against the 75th percentile of the others, so on a short deployment
every camera is equally concentrated, the threshold lands on top of all of
them, and every camera is labelled migratory — confidently and wrongly, with
no error raised. These tests pin down that the system now detects that
situation instead of presenting it as a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildlife_monitor.data.sufficiency import (
    assess, guidance_for, MIN_SPAN_MONTHS, MIN_CAMERAS,
)
from wildlife_monitor.pipeline2.labelling import (
    classify_movement, compute_site_fidelity, compute_temporal_concentration,
)


def deployment(cameras: int, span_months: int, per_camera: int,
               seed: int = 0) -> pd.DataFrame:
    """Synthetic detections over a given number of cameras and months."""
    rng = np.random.default_rng(seed)
    rows = []
    for camera in range(cameras):
        for index in range(per_camera):
            month = 1 + int(rng.integers(0, max(span_months, 1)))
            rows.append({
                "camera_id": f"C{camera:03d}",
                "instance_count": 1,
                "timestamp": f"2024-{min(month, 12):02d}-"
                              f"{index % 27 + 1:02d} 09:00:00",
            })
    return pd.DataFrame(rows)


def movement_labels(frame: pd.DataFrame) -> dict[str, int]:
    fidelity = compute_site_fidelity(frame)
    concentration = compute_temporal_concentration(frame)
    counts = frame.groupby("camera_id").size()
    labels: dict[str, int] = {}
    for camera in counts.index:
        label = classify_movement(camera, fidelity, concentration, counts)
        labels[label] = labels.get(label, 0) + 1
    return labels


# ── The failure this module exists to catch ──────────────────────────────────

def test_short_deployment_labels_everything_migratory():
    """Demonstrates the silent failure, so the guard has something to guard."""
    frame = deployment(cameras=30, span_months=1, per_camera=20)
    labels = movement_labels(frame)
    assert labels.get("migratory") == 30
    assert labels.get("territorial", 0) == 0


def test_short_deployment_is_reported_as_unsupported():
    report = assess(deployment(cameras=30, span_months=1, per_camera=20))
    assert report.capability("movement").status == "unsupported"
    assert not report.supports("movement")


def test_more_images_do_not_rescue_a_short_deployment():
    """The constraint is months, not volume — the guard must say so."""
    small = assess(deployment(cameras=30, span_months=1, per_camera=10))
    large = assess(deployment(cameras=30, span_months=1, per_camera=500))
    assert small.capability("movement").status == "unsupported"
    assert large.capability("movement").status == "unsupported"
    assert "months" in guidance_for(large)


def test_more_cameras_do_not_rescue_a_short_deployment():
    report = assess(deployment(cameras=200, span_months=1, per_camera=20))
    assert report.capability("movement").status == "unsupported"


# ── Each axis independently ──────────────────────────────────────────────────

def test_too_few_cameras_blocks_movement_even_over_a_full_year():
    report = assess(deployment(cameras=4, span_months=12, per_camera=30))
    assert report.capability("movement").status == "unsupported"
    assert "cameras" in report.capability("movement").reason


def test_a_full_year_across_many_cameras_is_supported():
    report = assess(deployment(cameras=40, span_months=12, per_camera=25))
    assert report.capability("movement").status == "supported"
    assert report.overall == "supported"
    assert guidance_for(report) == ""


def test_activity_does_not_depend_on_time_span():
    """Day/night needs detections per camera, not months of coverage."""
    report = assess(deployment(cameras=20, span_months=1, per_camera=40))
    assert report.capability("activity").status == "supported"
    assert report.capability("movement").status == "unsupported"


def test_sparse_cameras_block_activity():
    report = assess(deployment(cameras=40, span_months=12, per_camera=2))
    assert report.capability("activity").status == "unsupported"


def test_social_structure_needs_no_time_span():
    report = assess(deployment(cameras=20, span_months=1, per_camera=20))
    assert report.capability("social").status == "supported"


# ── Reporting ────────────────────────────────────────────────────────────────

def test_empty_data_is_unsupported_and_offers_no_advice():
    report = assess(pd.DataFrame())
    assert report.overall == "unsupported"
    assert all(not c.usable for c in report.capabilities)
    assert guidance_for(report) == ""
    assert "No detections" in report.summary_line()


def test_report_records_coverage_facts():
    report = assess(deployment(cameras=12, span_months=6, per_camera=15))
    assert report.cameras == 12
    assert report.detections == 12 * 15
    assert 4 <= report.span_months <= 7
    assert report.first_capture and report.last_capture


def test_undated_detections_are_counted():
    frame = deployment(cameras=5, span_months=12, per_camera=10)
    frame.loc[frame.index[:7], "timestamp"] = "not a date"
    assert assess(frame).undated == 7


def test_summary_line_reads_as_english():
    """Months pluralise, and sub-month coverage reports days not '0 months'."""
    from wildlife_monitor.data.sufficiency import _months
    assert _months(1.0) == "1 month"
    assert _months(6.0) == "6 months"
    assert _months(0.3) == "9 days"
    assert _months(0.0) == "0 days"

    short = assess(deployment(cameras=10, span_months=1, per_camera=10))
    assert "days of coverage" in short.summary_line()
    assert "0 months" not in short.summary_line()


@pytest.mark.parametrize("span,cameras,expected", [
    (1, 40, "unsupported"),
    (4, 40, "limited"),
    (12, 15, "limited"),
    (12, 40, "supported"),
])
def test_movement_grading_across_deployments(span, cameras, expected):
    report = assess(deployment(cameras=cameras, span_months=span,
                                per_camera=20))
    assert report.capability("movement").status == expected


# ── Tri-state verification ───────────────────────────────────────────────────

def test_unverified_detections_are_not_counted_as_wrong():
    """The bug this replaces reported unlabelled data as 0% accurate."""
    from wildlife_monitor.dashboard import data_access as da

    frame = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "confidence": [0.9, 0.8, 0.4, 0.2],
        "correct": ["correct", "incorrect", "unknown", "unknown"],
        "location": ["1,2,3,4"] * 4,
    })
    frame["verification"] = frame["correct"].map(
        lambda v: v if v in ("correct", "incorrect") else "unverified")

    counts = da.verification_counts(frame)
    assert counts == {"correct": 1, "incorrect": 1, "unverified": 2}
    # Accuracy is over labelled detections only: 1 of 2, not 1 of 4.
    assert da.accuracy_of(frame) == pytest.approx(50.0)
    assert da.has_ground_truth(frame)


def test_fully_unlabelled_data_reports_no_accuracy_rather_than_zero():
    from wildlife_monitor.dashboard import data_access as da

    frame = pd.DataFrame({
        "image_id": ["a", "b"],
        "confidence": [0.9, 0.3],
        "verification": ["unverified", "unverified"],
    })
    assert da.accuracy_of(frame) is None
    assert not da.has_ground_truth(frame)


@pytest.mark.parametrize("value,expected", [
    (0.95, "high"), (0.75, "high"), (0.74, "medium"),
    (0.50, "medium"), (0.49, "low"), (0.0, "low"),
])
def test_confidence_bands(value, expected):
    from wildlife_monitor.dashboard import data_access as da
    assert da.confidence_band(value) == expected
