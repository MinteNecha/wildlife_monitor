"""
Counting per-camera classifications up into one statement per species (VL2).

This is the only output in the project that can be checked against published
ecology, so the ways it could mislead matter more than the arithmetic.

Three failures would each look like a result:

* declaring a winner when the vote is nearly even, which turns noise into a
  finding an ecologist would then compare against the literature;
* counting cameras with almost no detections, whose day-or-night label turns
  on one or two photographs;
* letting a stray or empty label win by appearing more often than the real
  classes.

Each has a test. The module must also never claim agreement with literature,
because it holds no literature and deciding that is the ecologist's job.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wildlife_monitor.pipeline2.aggregate import (
    CLEAR_MARGIN, CLEAR_SHARE, Rollup, counts_frame, profile_all,
    profile_species, roll_up, to_frame,
)
from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES,
)


def cameras(species: str, activity: list[str], movement: list[str],
            social: list[str] | None = None,
            detections: list[int] | None = None,
            rule_activity: list[str] | None = None,
            rule_movement: list[str] | None = None) -> pd.DataFrame:
    """A frame of per-camera classifications, shaped like predict_frame."""
    size = len(activity)
    social = social or ["solitary"] * size
    detections = detections or [40] * size
    rule_activity = rule_activity or list(activity)
    rule_movement = rule_movement or list(movement)
    return pd.DataFrame([{
        "camera_id": f"{species[:3].upper()}{index:02d}",
        "species": species,
        "activity_class": activity[index],
        "movement_class": movement[index],
        "social_class": social[index],
        "rule_activity_class": rule_activity[index],
        "rule_movement_class": rule_movement[index],
        "detection_count": detections[index],
    } for index in range(size)])


def movement_vote(split: list[str]) -> Rollup:
    return roll_up(pd.DataFrame({"movement_class": split}),
                   "movement_class", "Movement", MOVEMENT_CLASSES)


# ── counting ─────────────────────────────────────────────────────────────────

def test_the_majority_label_wins_and_the_share_is_the_vote_count():
    frame = cameras("lionfemale",
                    ["nocturnal"] * 31 + ["crepuscular"] * 6 + ["diurnal"] * 3,
                    ["territorial"] * 40)
    profile = profile_species(frame, "lionfemale")

    assert profile.activity.label == "nocturnal"
    assert profile.activity.votes == 40
    assert profile.activity.share == pytest.approx(31 / 40)
    assert profile.activity.percentage == "78%"


def test_every_class_keeps_its_count_even_when_it_did_not_win():
    """The winner alone hides whether a majority was thin."""
    frame = cameras("zebra", ["diurnal"] * 20 + ["nocturnal"] * 15
                    + ["crepuscular"] * 5, ["nomadic"] * 40)
    profile = profile_species(frame, "zebra")

    assert profile.activity.counts == {
        "diurnal": 20, "nocturnal": 15, "crepuscular": 5}
    assert sum(profile.activity.counts.values()) == profile.activity.votes


def test_each_species_is_counted_separately():
    frame = pd.concat([
        cameras("lionfemale", ["nocturnal"] * 10, ["territorial"] * 10),
        cameras("giraffe", ["diurnal"] * 10, ["nomadic"] * 10),
    ], ignore_index=True)

    profiles = {profile.species: profile for profile in profile_all(frame)}
    assert profiles["lionfemale"].activity.label == "nocturnal"
    assert profiles["giraffe"].activity.label == "diurnal"
    assert profiles["lionfemale"].cameras == 10


# ── refusing to declare a winner ─────────────────────────────────────────────

def test_a_near_even_vote_is_reported_as_split():
    """16/14/10 is noise. Calling it a finding would be the real bug."""
    rollup = movement_vote(["migratory"] * 16 + ["territorial"] * 14
                           + ["nomadic"] * 10)

    assert rollup.label == "migratory"          # still names the leader
    assert not rollup.clear                     # but refuses to call it
    assert "split" in rollup.summary_line()


def test_a_dead_heat_is_never_clear():
    rollup = movement_vote(["migratory"] * 20 + ["territorial"] * 20)
    assert rollup.share == pytest.approx(0.5)
    assert not rollup.clear


def test_a_strong_majority_is_clear():
    rollup = movement_vote(["migratory"] * 30 + ["territorial"] * 5
                           + ["nomadic"] * 5)
    assert rollup.clear
    assert "split" not in rollup.summary_line()


def test_clarity_needs_both_a_majority_and_a_margin():
    """A bare majority over a close runner-up is not a pattern."""
    # 51% but the runner-up holds 49%, so the margin fails.
    narrow = movement_vote(["migratory"] * 51 + ["territorial"] * 49)
    assert narrow.share > CLEAR_SHARE
    assert (narrow.share - narrow.runner_up_share) < CLEAR_MARGIN
    assert not narrow.clear


# ── thin cameras ─────────────────────────────────────────────────────────────

def test_cameras_with_too_few_detections_are_not_counted():
    """A day-or-night label from two photographs is not evidence."""
    frame = cameras("cheetah",
                    ["diurnal"] * 4 + ["nocturnal"] * 2,
                    ["nomadic"] * 6,
                    detections=[50, 40, 30, 20, 2, 1])
    profile = profile_species(frame, "cheetah")

    assert profile.cameras == 6
    assert profile.cameras_counted == 4
    assert profile.cameras_excluded == 2
    assert profile.activity.label == "diurnal"
    assert profile.activity.share == 1.0     # the two thin ones did not vote


def test_the_excluded_count_is_stated_not_hidden():
    frame = cameras("cheetah", ["diurnal"] * 3, ["nomadic"] * 3,
                    detections=[50, 2, 1])
    lines = " ".join(profile_species(frame, "cheetah").summary_lines())
    assert "2 of 3 cameras excluded" in lines


def test_the_threshold_can_be_changed():
    frame = cameras("cheetah", ["diurnal"] * 3, ["nomadic"] * 3,
                    detections=[50, 7, 1])
    assert profile_species(frame, "cheetah", min_detections=5).cameras_counted == 2
    assert profile_species(frame, "cheetah", min_detections=10).cameras_counted == 1


def test_a_species_whose_cameras_are_all_thin_reports_nothing():
    frame = cameras("leopard", ["diurnal"] * 3, ["nomadic"] * 3,
                    detections=[1, 2, 1])
    profile = profile_species(frame, "leopard")

    assert profile.cameras_counted == 0
    assert profile.activity.label == ""
    assert not profile.activity.clear
    assert "no cameras with enough detections" in profile.activity.summary_line()


# ── bad input ────────────────────────────────────────────────────────────────

def test_a_stray_label_cannot_win():
    """Only real classes are counted, whatever else appears in the column."""
    rollup = roll_up(pd.DataFrame({"movement_class":
                                   ["rubbish"] * 50 + ["migratory"] * 3}),
                     "movement_class", "Movement", MOVEMENT_CLASSES)
    assert rollup.label == "migratory"
    assert rollup.votes == 3


def test_empty_and_missing_input_produce_no_claim():
    assert roll_up(pd.DataFrame(), "movement_class", "Movement",
                   MOVEMENT_CLASSES).label == ""
    assert roll_up(pd.DataFrame({"other": [1]}), "movement_class", "Movement",
                   MOVEMENT_CLASSES).label == ""
    assert profile_all(pd.DataFrame()) == []


def test_ties_resolve_the_same_way_every_run():
    """A tie must not depend on row order, or the table changes each run."""
    forward = movement_vote(["migratory"] * 10 + ["territorial"] * 10)
    backward = movement_vote(["territorial"] * 10 + ["migratory"] * 10)
    assert forward.label == backward.label


# ── model against rule ───────────────────────────────────────────────────────

def test_agreement_with_the_rule_is_reported():
    """Which of the two to blame for a mismatch with the literature."""
    frame = cameras("zebra", ["diurnal"] * 30, ["migratory"] * 30,
                    rule_activity=["diurnal"] * 30,
                    rule_movement=["nomadic"] * 30)
    profile = profile_species(frame, "zebra")

    assert profile.model_and_rule_agree == {"activity": True, "movement": False}
    assert profile.movement.label == "migratory"
    assert profile.rule_movement.label == "nomadic"


def test_missing_rule_columns_are_tolerated():
    frame = cameras("zebra", ["diurnal"] * 5, ["nomadic"] * 5).drop(
        columns=["rule_activity_class", "rule_movement_class"])
    profile = profile_species(frame, "zebra")

    assert profile.rule_activity is None
    assert profile.model_and_rule_agree == {}


# ── the exported table ───────────────────────────────────────────────────────

def test_the_table_has_one_row_per_species_with_label_and_share():
    frame = pd.concat([
        cameras("lionfemale", ["nocturnal"] * 31 + ["diurnal"] * 9,
                ["territorial"] * 40),
        cameras("giraffe", ["diurnal"] * 29 + ["nocturnal"] * 6,
                ["nomadic"] * 35),
    ], ignore_index=True)

    table = to_frame(profile_all(frame)).set_index("species")
    assert list(table.index) == ["giraffe", "lionfemale"]
    assert table.loc["lionfemale", "activity"] == "nocturnal"
    assert table.loc["lionfemale", "activity_agreement"] == "78%"
    assert table.loc["giraffe", "activity"] == "diurnal"


def test_the_table_marks_which_results_are_clear():
    frame = pd.concat([
        cameras("a", ["diurnal"] * 30 + ["nocturnal"] * 2, ["nomadic"] * 32),
        cameras("b", ["diurnal"] * 11 + ["nocturnal"] * 10
                + ["crepuscular"] * 9, ["nomadic"] * 30),
    ], ignore_index=True)

    table = to_frame(profile_all(frame)).set_index("species")
    assert bool(table.loc["a", "activity_clear"]) is True
    assert bool(table.loc["b", "activity_clear"]) is False


def test_the_full_vote_export_covers_every_class():
    frame = cameras("zebra", ["diurnal"] * 20 + ["nocturnal"] * 10,
                    ["nomadic"] * 30)
    detail = counts_frame(profile_all(frame))

    assert set(detail["dimension"]) == {"Activity", "Movement", "Group size"}
    activity = detail[detail["dimension"] == "Activity"].set_index("class")
    assert list(activity.index) == ACTIVITY_CLASSES
    assert activity.loc["crepuscular", "cameras"] == 0


def test_the_module_never_claims_agreement_with_literature():
    """It holds no literature, so it must not appear to validate itself."""
    import wildlife_monitor.pipeline2.aggregate as module

    exported = {name for name in dir(module) if not name.startswith("_")}
    for forbidden in ("EXPECTED", "KNOWN_ECOLOGY", "LITERATURE",
                      "validate_against_literature", "expected_activity"):
        assert forbidden not in exported


# ── shapes the caller may pass ───────────────────────────────────────────────

def test_a_list_of_dicts_works_as_well_as_a_frame():
    rows = cameras("zebra", ["diurnal"] * 10, ["nomadic"] * 10).to_dict(
        "records")
    assert profile_species(rows, "zebra").activity.label == "diurnal"


def test_alternative_column_names_are_accepted():
    frame = pd.DataFrame({
        "species": ["zebra"] * 10,
        "activity": ["diurnal"] * 10,
        "movement": ["nomadic"] * 10,
        "social": ["small group"] * 10,
        "detections": [40] * 10,
    })
    profile = profile_species(frame, "zebra")
    assert profile.activity.label == "diurnal"
    assert profile.social.label == "small group"
    assert profile.cameras_counted == 10
