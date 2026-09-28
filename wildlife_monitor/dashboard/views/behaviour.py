"""
Behavioural Analysis page (UC4, UC5, UC6 · FR4, FR5) — Pipeline 2.

This is the surface for the temporal behavioural model. It loads a trained
checkpoint, classifies every camera for the selected species, and presents the
results for an ecologist rather than for a terminal: what each camera's
behaviour looks like, how confident the model is, where it disagrees with the
rule-derived label it was trained against, and how it scored on cameras it
never saw during training.

Two honesty requirements shape this page. Per-class recall is shown alongside
overall accuracy, because a rare class such as migratory disappears inside a
mean. And the social-structure class is marked as rule-derived, because it is
computed from instance counts rather than predicted by the network.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import (
    header, rule, note, sufficiency_panel,
)
from wildlife_monitor.dashboard.theme import (
    PLOT, GRID, CHART_SEQ, WHITE, BLACK, GREY, POS, NEG, ACCENT,
)
from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, SOCIAL_CLASSES, CLASS_DESCRIPTIONS,
)
from wildlife_monitor.pipeline2.query import QueryEngine, EXAMPLE_QUERIES
from wildlife_monitor.pipeline2.validation import PatternValidator, VERDICTS

CLASS_COLOURS = dict(zip(MOVEMENT_CLASSES, CHART_SEQ))


@st.cache_resource(show_spinner=False)
def _service(species: str, architecture: str):
    """Cached model load — a checkpoint should not be re-read on every rerun."""
    return da.load_behaviour_service(species, architecture)


@st.cache_data(show_spinner=False)
def _predictions(species: str, architecture: str) -> pd.DataFrame:
    service = _service(species, architecture)
    if service is None:
        return pd.DataFrame()
    return service.predict_frame(da.behaviour_detections(species), species)


def render(species: str) -> None:
    header("Behavioural Analysis",
           "Pipeline 2 · temporal behaviour model · UC4, UC5, UC6")

    architectures = da.behaviour_architectures(species)
    if not architectures:
        _no_model_guidance(species)
        return

    architecture = _architecture_selector(architectures)
    service = _service(species, architecture)
    if service is None:
        st.error(f"The checkpoint for {da.pretty(species)} ({architecture}) "
                 f"could not be loaded. Retrain it to continue.")
        return

    metrics = da.load_behaviour_metrics(species, architecture)
    predictions = _predictions(species, architecture)
    if predictions.empty:
        st.warning("The model loaded but produced no predictions — there are "
                   "no detections for this species to classify.")
        return

    _model_in_use(service, species)
    _metric_row(predictions, metrics, service)
    rule()
    _data_sufficiency(species)
    rule()
    _class_distributions(predictions)
    rule()
    _held_out_performance(metrics)
    rule()
    _generalisation(architecture)
    rule()
    _camera_table(predictions)
    rule()
    _seasonal_profile(species, predictions)
    rule()
    _species_summary(species, predictions)
    rule()
    _query_section(species)
    rule()
    _validation_section(species)
    rule()
    _provenance(service, metrics)


def _data_sufficiency(species: str) -> None:
    """What this deployment's data can honestly support."""
    st.subheader("What This Data Supports")
    report = da.sufficiency(species)
    sufficiency_panel(report)
    if not report.supports("movement"):
        note("Movement classifications are shown below because the model "
             "produces them, but on this data they are not dependable. Read "
             "the activity results instead.", ok=False)


def _no_model_guidance(species: str) -> None:
    """Shown when no behaviour model has been trained for this species yet."""
    st.info(f"No behavioural model has been trained for "
            f"**{da.pretty(species)}** yet.")
    detections = da.behaviour_detections(species)
    if detections.empty:
        st.markdown(
            "There are also no Pipeline 1 detections for this species. "
            "Run the detection pipeline first:")
        st.code(f"python scripts/run_pipeline.py --pipeline bioclip_megadetector "
                f"--species {species} --top_n 1500", language="bash")
        return

    cameras = detections["camera_id"].nunique()
    st.markdown(f"There are **{len(detections):,} detections** across "
                f"**{cameras} cameras** ready to train on:")
    st.code(f"python scripts/train_behaviour.py --species {species} --model lstm",
            language="bash")
    trained = da.behaviour_checkpoints()
    if trained:
        st.caption("Trained models available for: "
                   + ", ".join(sorted({da.pretty(entry['species'])
                                       for entry in trained})))


def _architecture_selector(architectures: list[str]) -> str:
    """Pick the architecture, defaulting to whatever Settings selected."""
    preferred = str(st.session_state.get("arch", "lstm")).lower()
    index = architectures.index(preferred) if preferred in architectures else 0
    labels = {"lstm": "LSTM", "transformer": "Transformer"}
    return st.radio(
        "Model architecture", architectures, index=index, horizontal=True,
        format_func=lambda name: labels.get(name, name.title()))


def _metric_row(predictions: pd.DataFrame, metrics: dict, service) -> None:
    held_out = metrics.get("metrics", {})
    movement_agreement = predictions["movement_agrees"].mean() * 100
    truncated = int(predictions["sequence_truncated"].sum())

    columns = st.columns(5)
    for column, (label, value, help_text) in zip(columns, [
        ("Cameras Classified", f"{len(predictions):,}",
         "Every camera with detections for this species."),
        ("Activity Accuracy", _percent(held_out.get("activity_accuracy")),
         "On held-out cameras never seen during training."),
        ("Movement Accuracy", _percent(held_out.get("movement_accuracy")),
         "On held-out cameras never seen during training."),
        ("Agrees With Rule", f"{movement_agreement:.0f}%",
         "How often the model's movement class matches the rule-derived "
         "label it was trained against, across all cameras."),
        ("Sequences Truncated", f"{truncated}",
         "Cameras with more detections than the model's input window. "
         "Labels still use their full history."),
    ]):
        column.metric(label, value, help=help_text)


def _percent(value) -> str:
    return f"{value:.1f}%" if isinstance(value, (int, float)) else "—"


def _class_distributions(predictions: pd.DataFrame) -> None:
    st.subheader("Behavioural Classifications")
    left, middle, right = st.columns(3)
    for column, (title, field, classes, caption) in zip(
        [left, middle, right],
        [("Activity Timing", "activity_class", ACTIVITY_CLASSES, "Predicted"),
         ("Movement Strategy", "movement_class", MOVEMENT_CLASSES, "Predicted"),
         ("Social Structure", "social_class", SOCIAL_CLASSES, "Rule-derived")],
    ):
        with column:
            st.markdown(f"**{title}**")
            st.caption(caption)
            counts = (predictions[field].value_counts()
                      .reindex(classes, fill_value=0))
            figure = go.Figure(go.Bar(
                x=counts.values, y=counts.index, orientation="h",
                marker=dict(color=CHART_SEQ[:len(classes)], line=dict(width=0)),
                text=counts.values, textposition="outside",
                textfont=dict(color=BLACK, size=11)))
            figure.update_layout(
                **PLOT, height=200, bargap=0.45, showlegend=False,
                xaxis=dict(range=[0, max(counts.max() * 1.25, 1)], **GRID),
                yaxis=dict(showgrid=False, tickfont=dict(color=BLACK, size=11)))
            st.plotly_chart(figure, width="stretch")

    with st.expander("What these classes mean"):
        for name, description in CLASS_DESCRIPTIONS.items():
            st.markdown(f"**{name}** — {description}")


def _held_out_performance(metrics: dict) -> None:
    """Per-class recall on held-out cameras — where a rare class shows up."""
    st.subheader("Held-Out Performance by Class")
    held_out = metrics.get("metrics", {})
    if not held_out:
        st.info("No saved metrics for this model. Retrain it to record them.")
        return

    rows = []
    for task, key in [("Activity", "activity_per_class"),
                      ("Movement", "movement_per_class")]:
        for class_name, entry in (held_out.get(key) or {}).items():
            rows.append({
                "Task": task,
                "Class": class_name,
                "Test cameras": entry.get("support", 0),
                "Correct": entry.get("correct", 0),
                "Recall": ("—" if entry.get("recall") is None
                           else f"{entry['recall'] * 100:.0f}%"),
            })
    if not rows:
        st.info("No per-class breakdown was recorded for this model.")
        return

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    supports = [row["Test cameras"] for row in rows if row["Task"] == "Movement"]
    smallest = min(supports) if supports else 0
    if 0 < smallest <= 7:
        note(f"The rarest movement class has only {smallest} held-out "
             f"cameras. At that size a single prediction moves recall by "
             f"more than ten percentage points, so treat these figures as "
             f"indicative rather than precise.", ok=False)


def _camera_table(predictions: pd.DataFrame) -> None:
    st.subheader("Per-Camera Classifications")

    left, right = st.columns([2, 3])
    with left:
        movement_filter = st.multiselect(
            "Movement strategy", MOVEMENT_CLASSES, default=MOVEMENT_CLASSES)
    with right:
        minimum_confidence = st.slider(
            "Minimum movement confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = predictions[
        predictions["movement_class"].isin(movement_filter)
        & (predictions["movement_confidence"] >= minimum_confidence)
    ]
    if filtered.empty:
        st.info("No cameras match these filters.")
        return

    display = filtered[[
        "camera_id", "activity_class", "activity_confidence",
        "movement_class", "movement_confidence", "social_class",
        "detection_count", "sequence_truncated", "movement_agrees",
    ]].rename(columns={
        "camera_id": "Camera",
        "activity_class": "Activity",
        "activity_confidence": "Act. conf.",
        "movement_class": "Movement",
        "movement_confidence": "Mov. conf.",
        "social_class": "Social (rule)",
        "detection_count": "Detections",
        "sequence_truncated": "Truncated",
        "movement_agrees": "Matches rule",
    })
    st.dataframe(display.sort_values("Detections", ascending=False),
                 width="stretch", hide_index=True)

    disagreements = int((~filtered["movement_agrees"]).sum())
    st.caption(f"{disagreements} of {len(filtered)} shown cameras disagree with "
               f"the rule-derived movement label. Disagreement is not "
               f"automatically an error — the rule is a heuristic, and these "
               f"cameras are the ones worth reviewing by eye.")


def _seasonal_profile(species: str, predictions: pd.DataFrame) -> None:
    """Monthly detection shape for one camera — the migratory signal itself."""
    st.subheader("Seasonal Profile")
    st.caption("Share of a camera's detections by month, computed from its "
               "complete history. This is the signal the movement head reads.")

    camera = st.selectbox(
        "Camera", predictions.sort_values("detection_count", ascending=False)
        ["camera_id"].tolist())
    detections = da.behaviour_detections(species)
    profile = da.monthly_profile(detections, camera)
    if profile.empty:
        st.info("No monthly profile available for this camera.")
        return

    row = predictions.loc[predictions["camera_id"] == camera].iloc[0]
    colour = CLASS_COLOURS.get(row["movement_class"], ACCENT)
    figure = go.Figure(go.Bar(
        x=profile["month"], y=profile["share"] * 100,
        marker=dict(color=colour, line=dict(width=0)),
        hovertemplate="%{x}: %{y:.1f}% of detections<extra></extra>"))
    figure.update_layout(
        **PLOT, height=240, bargap=0.35,
        xaxis=dict(showgrid=False, tickfont=dict(color=BLACK, size=11)),
        yaxis=dict(title="Share of detections (%)", **GRID))
    st.plotly_chart(figure, width="stretch")

    peak = profile.loc[profile["share"].idxmax()]
    st.markdown(
        f"<div style='font-size:12px;color:{GREY}'>"
        f"<b style='color:{BLACK}'>{camera}</b> · "
        f"{int(row['detection_count'])} detections · busiest month "
        f"<b style='color:{BLACK}'>{peak['month']}</b> at "
        f"{peak['share'] * 100:.0f}% of the camera's total · classified "
        f"<b style='color:{colour}'>{row['movement_class']}</b> "
        f"({row['movement_confidence'] * 100:.0f}% confidence)</div>",
        unsafe_allow_html=True)


def _query_section(species: str) -> None:
    """Natural-language querying of stored patterns (UC4, FR6)."""
    st.subheader("Query Patterns")
    st.caption("Ask in plain English. The parser matches against known "
               "behaviour classes, species, habitats, months and camera "
               "codes, and shows you exactly what it understood.")

    patterns = da.stored_patterns(species)
    if patterns.empty:
        st.info("No stored patterns to query yet. Patterns are saved when a "
                "model is trained.")
        st.code(f"python scripts/train_behaviour.py --species {species}",
                language="bash")
        return

    query = st.text_input(
        "Query", placeholder=EXAMPLE_QUERIES[0], key="behaviour_query")
    st.caption("Try: " + " · ".join(f"“{example}”"
                                     for example in EXAMPLE_QUERIES[:3]))
    if not query:
        return

    results, parsed = QueryEngine().run(query, patterns)

    if parsed.is_empty:
        note("Nothing in that query matched a known term, so every pattern is "
             "shown. Try naming a behaviour class, species, habitat or month.",
             ok=False)
    else:
        note(f"Understood: {parsed.describe()}", ok=True)
    if parsed.unrecognised:
        st.caption("Ignored words: " + ", ".join(parsed.unrecognised[:8]))

    st.markdown(f"**{len(results)} matching pattern"
                f"{'' if len(results) == 1 else 's'}**")
    if results.empty:
        st.info("No patterns match. The filter was applied correctly — there "
                "simply are none with that combination.")
        return

    st.dataframe(
        results[["camera_id", "activity_class", "movement_class",
                 "social_class", "movement_confidence", "detection_count",
                 "habitat_type"]].rename(columns={
                     "camera_id": "Camera", "activity_class": "Activity",
                     "movement_class": "Movement", "social_class": "Social",
                     "movement_confidence": "Confidence",
                     "detection_count": "Detections",
                     "habitat_type": "Habitat"}),
        width="stretch", hide_index=True)


def _validation_section(species: str) -> None:
    """Ecologist review of a pattern (UC6, FR7)."""
    st.subheader("Validate Patterns")
    st.caption("Record whether a classification matches what is known about "
               "this species and site. Verdicts are appended, never "
               "overwritten, and never alter the model's own confidence.")

    patterns = da.stored_patterns(species)
    if patterns.empty:
        st.info("No stored patterns to review yet.")
        return

    validator = PatternValidator()
    counts = validator.counts(species)
    columns = st.columns(4)
    for column, (label, value) in zip(columns, [
        ("Patterns", len(patterns)),
        ("Validated", counts.get("validated", 0)),
        ("Novel", counts.get("novel", 0)),
        ("Spurious", counts.get("spurious", 0)),
    ]):
        column.metric(label, value)

    left, right = st.columns([2, 3])
    with left:
        camera = st.selectbox("Camera to review",
                              patterns["camera_id"].tolist(),
                              key="validate_camera")
        verdict = st.radio("Verdict", VERDICTS, key="validate_verdict",
                           format_func=str.title)
    with right:
        row = patterns.loc[patterns["camera_id"] == camera].iloc[0]
        st.markdown(
            f"<div style='font-size:12px;color:{GREY};line-height:1.9'>"
            f"<b style='color:{BLACK}'>{camera}</b> · "
            f"{int(row['detection_count'])} detections<br>"
            f"Activity <b style='color:{BLACK}'>{row['activity_class']}</b> · "
            f"Movement <b style='color:{BLACK}'>{row['movement_class']}</b> "
            f"({row['movement_confidence'] * 100:.0f}%) · "
            f"Social <b style='color:{BLACK}'>{row['social_class']}</b></div>",
            unsafe_allow_html=True)
        notes = st.text_area("Notes", key="validate_notes", height=90,
                             placeholder="Why does this verdict apply?")

    pattern_id = int(row["pattern_id"])
    if st.button("Record verdict"):
        try:
            validator.record_verdict(pattern_id, verdict, notes)
            st.success(f"Recorded '{verdict}' for {camera}.")
        except (ValueError, KeyError) as error:
            st.error(str(error))

    try:
        summary = validator.summary(pattern_id)
    except KeyError:
        return
    if summary.reviewed:
        st.markdown(
            f"<div style='font-size:12px;color:{GREY}'>"
            f"Reviewed {summary.review_count} time"
            f"{'' if summary.review_count == 1 else 's'} · latest "
            f"<b style='color:{BLACK}'>{summary.latest_verdict}</b> · "
            f"model confidence {summary.model_confidence * 100:.0f}% → "
            f"review-adjusted "
            f"<b style='color:{POS if summary.adjusted_confidence >= summary.model_confidence else NEG}'>"
            f"{summary.adjusted_confidence * 100:.0f}%</b></div>",
            unsafe_allow_html=True)

    history = validator.review_state(species)
    if not history.empty:
        with st.expander(f"Review history ({len(history)} patterns reviewed)"):
            st.dataframe(history[["camera_id", "verdict", "notes",
                                  "validated_at"]].rename(columns={
                                      "camera_id": "Camera",
                                      "verdict": "Verdict", "notes": "Notes",
                                      "validated_at": "Recorded"}),
                         width="stretch", hide_index=True)


def _species_summary(species: str, predictions) -> None:
    """One statement for the species, counted from its cameras (VL2).

    Every other figure on this page measures agreement with the project's own
    quartile rule. This is the only output that can be checked against
    published ecology, because ecology is written about species rather than
    about camera sites.
    """
    from wildlife_monitor.pipeline2.aggregate import MIN_DETECTIONS, profile_species

    profile = profile_species(predictions, species)
    if not profile.cameras_counted:
        return

    st.subheader("Species-Level Summary")
    st.caption(f"Each camera votes once. Cameras with fewer than "
               f"{MIN_DETECTIONS} detections are not counted, because a "
               f"day-or-night label from two photographs is not evidence.")

    columns = st.columns(3)
    for column, rollup in zip(columns, (profile.activity, profile.movement,
                                         profile.social)):
        column.metric(rollup.dimension,
                      rollup.label if rollup.clear else "split",
                      f"{rollup.percentage} of cameras",
                      delta_color="off")

    unclear = [rollup.dimension for rollup in
               (profile.activity, profile.movement)
               if not rollup.clear]
    if unclear:
        note(f"No single label is clearly ahead for: "
             f"{', '.join(unclear).lower()}. The cameras disagree too much "
             f"for a species-level statement, so do not read one from the "
             f"leading label alone.", ok=False)

    disagreement = [name for name, agrees
                    in profile.model_and_rule_agree.items() if not agrees]
    if disagreement:
        note(f"The model and the labelling rule give different species-level "
             f"answers for {', '.join(disagreement)}. If this species "
             f"disagrees with published ecology, that points at the model. "
             f"Where they agree, it points at the rule.", ok=False)

    if profile.cameras_excluded:
        st.caption(f"{profile.cameras_excluded} of {profile.cameras} cameras "
                   f"were excluded as too thin to vote.")

    st.caption("Compare these against published ecology for this species. "
               "The system does not make that comparison itself, because "
               "which source is authoritative is a judgement for the "
               "ecologist rather than something to hard-code.")


def _model_in_use(service, species: str) -> None:
    """Say plainly which model answered, before any numbers are shown.

    A prediction from a model that has never seen this animal is a weaker
    claim than one from a model trained on it. A reader who cannot tell the
    two apart will over-trust the first.
    """
    if not getattr(service, "is_cross_species", False):
        return

    trained = ", ".join(da.pretty(name)
                        for name in service.trained_species) or "several species"
    if service.saw_species(species):
        note(f"Classified by the cross-species model, trained on {trained}. "
             f"{da.pretty(species)} was part of its training data.", ok=True)
        return

    note(f"<b>{da.pretty(species)} has no model of its own.</b> These "
         f"classifications come from the cross-species model, trained on "
         f"{trained}. It has never seen this species, so treat the results as "
         f"indicative and expect lower accuracy than the figures below, which "
         f"were measured on species it had also never seen.", ok=False)


def _generalisation(architecture: str) -> None:
    """How well the pooled model transfers to species it never trained on.

    This is the evidence behind any prediction made for an untrained species,
    so it belongs on the page rather than only in a metrics file.
    """
    report = da.load_cross_species_metrics(architecture)
    folds = report.get("folds") or []
    if not folds:
        return

    st.subheader("Generalisation to Unseen Species")
    st.caption("Each row trains the model without that species, then tests it "
               "on that species alone. Every test camera is therefore an "
               "animal the model has never encountered.")

    chance = float(report.get("chance_baseline", 33.33))
    frame = pd.DataFrame([{
        "Held-out species": da.pretty(fold["held_out_species"]),
        "Trained on": len(fold.get("train_species", [])),
        "Test cameras": fold.get("test_cameras", 0),
        "Activity": f"{fold.get('activity_accuracy', 0):.1f}%",
        "Movement": f"{fold.get('movement_accuracy', 0):.1f}%",
    } for fold in folds])
    st.dataframe(frame, width="stretch", hide_index=True)

    mean = float(report.get("mean_movement_accuracy", 0.0))
    scores = [float(fold.get("movement_accuracy", 0.0)) for fold in folds]
    columns = st.columns(3)
    columns[0].metric("Mean movement accuracy", f"{mean:.1f}%")
    columns[1].metric("Chance baseline", f"{chance:.1f}%")
    columns[2].metric("Spread across species",
                      f"{max(scores) - min(scores):.1f} pts" if scores else "—")

    if mean >= 70:
        note(f"Movement classification transfers to unseen species at "
             f"{mean:.1f}%. The model learned the shape of a detection "
             f"history rather than the identity of one animal.", ok=True)
    elif mean >= chance + 10:
        note(f"Movement classification partly transfers at {mean:.1f}%, "
             f"against a {chance:.1f}% chance baseline. There is "
             f"species-independent signal, but well short of the accuracy on "
             f"a species the model was trained on.", ok=False)
    else:
        note(f"Movement classification does not transfer. At {mean:.1f}% "
             f"against a {chance:.1f}% chance baseline, predictions for an "
             f"untrained species should not be relied on.", ok=False)

    st.caption("These labels come from a quantile rule computed within each "
               "species, so this measures whether the rule's shape transfers, "
               "not whether real animal behaviour does.")


def _provenance(service, metrics: dict) -> None:
    """Where this prediction came from — model, data, and known limitations."""
    st.subheader("Model Provenance")
    left, right = st.columns(2)

    with left:
        st.markdown(
            f"<div style='font-size:12px;color:{GREY};line-height:1.9'>"
            f"<b style='color:{BLACK}'>Model</b> {service.model_version}<br>"
            f"<b style='color:{BLACK}'>Trained</b> "
            f"{metrics.get('trained_at', 'unknown')}<br>"
            f"<b style='color:{BLACK}'>Training cameras</b> "
            f"{metrics.get('train_cameras', '—')}<br>"
            f"<b style='color:{BLACK}'>Held-out cameras</b> "
            f"{metrics.get('test_cameras', '—')}<br>"
            f"<b style='color:{BLACK}'>Epochs</b> "
            f"{metrics.get('epochs', '—')} · "
            f"<b style='color:{BLACK}'>Input window</b> "
            f"{metrics.get('max_length', '—')} detections</div>",
            unsafe_allow_html=True)

    with right:
        counts = metrics.get("movement_label_counts") or {}
        if counts:
            total = sum(counts.values())
            lines = " · ".join(
                f"{name} {count} ({count / total * 100:.0f}%)"
                for name, count in sorted(counts.items(),
                                           key=lambda item: -item[1]))
            st.markdown(
                f"<div style='font-size:12px;color:{GREY};line-height:1.9'>"
                f"<b style='color:{BLACK}'>Movement label balance</b><br>"
                f"{lines}</div>", unsafe_allow_html=True)

    if hasattr(service, "provenance"):
        st.caption(service.provenance())

    st.caption(
        "Activity and movement are model predictions. Social structure is "
        "derived by rule from the largest single-frame instance count, not "
        "predicted by the network. Movement labels themselves come from a "
        "heuristic over site fidelity and seasonal concentration, so the "
        "accuracy figures measure agreement with that heuristic on unseen "
        "cameras — not against independently observed ground truth.")
