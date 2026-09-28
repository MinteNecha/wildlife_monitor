# Experiments and design decisions

A record of changes that were tested against real data and then **rejected**,
with the measurements behind each decision, plus the design choices that
depart from the proposal and why.

Negative results are recorded here deliberately. An ablation that did not work
is evidence about the problem, and it is the difference between "we did not
need that" and "we tested it and here is what happened".

---

## 1. Class-weighted loss — tested, rejected

**The problem.** Movement labels are imbalanced: nomadic dominates, migratory
is rare. The standard remedy is to weight the cross-entropy by inverse class
frequency so a mistake on the rare class costs more.

**What was tested.** Inverse-frequency weighting applied to the movement loss
only, across 5 species × 2 architectures, weighted versus unweighted, all
other settings identical.

### Movement accuracy, baseline → weighted

| Species          | Model       | Baseline | Weighted | Outcome |
|------------------|-------------|----------|----------|---------|
| buffalo          | LSTM        | 75.0%    | 55.0%    | worse |
| lionfemale       | LSTM        | 77.3%    | 77.3%    | no change (identical predictions) |
| gazellethomsons  | LSTM        | 88.9%    | 88.9%    | no change overall |
| buffalo          | Transformer | 70.0%    | 60.0%    | worse |
| lionfemale       | Transformer | 68.2%    | 63.6%    | worse |
| gazellethomsons  | Transformer | 70.4%    | 51.9%    | much worse |

### Migratory recall, baseline → weighted

| Species          | Model       | Baseline   | Weighted   |
|------------------|-------------|------------|------------|
| buffalo          | LSTM        | 2/5 (40%)  | 2/5 (40%), **3 new false positives** |
| lionfemale       | LSTM        | 4/7 (57%)  | 4/7 (57%)  |
| gazellethomsons  | LSTM        | 1/3 (33%)  | 2/3 (67%)  |
| buffalo          | Transformer | 4/5 (80%)  | 3/5 (60%)  |
| lionfemale       | Transformer | 4/7 (57%)  | 3/7 (43%)  |
| gazellethomsons  | Transformer | 0/3 (0%)   | 0/3 (0%)   |

**Finding.** Weighting is consistently harmful for the Transformer (4 of 5
species worse) and inconsistent for the LSTM (2 improved, 1 unchanged, 2
worse). Where the LSTM degraded, the failure mode was specific and repeatable:
it did not detect *more* migratory cameras, it began misclassifying *nomadic*
cameras as migratory. The same signature appeared on two unrelated species.

**Decision.** Not adopted. The weighting code was removed rather than left as
a disabled option, so there is one code path and one set of results.

---

## 2. Even-spaced sequence sampling — tested, rejected

**The problem.** `build_sequence` truncates a camera's history to the first
`max_length` detections chronologically. For a camera with hundreds of
detections, everything after the cutoff is discarded — including, potentially,
the months that define whether it looks migratory.

**What was tested.** Replacing "first N" with N evenly-spaced samples across
the camera's full season, so a long sequence still shows the whole year.

| Species          | Model       | Before | After | Outcome |
|------------------|-------------|--------|-------|---------|
| buffalo          | LSTM        | 75.0%  | 65.0% | worse, plus new false positives |
| lionfemale       | LSTM        | 77.3%  | 77.3% | **identical** — never triggered |
| gazellethomsons  | LSTM        | 88.9%  | 88.9% | **identical** — never triggered |
| buffalo          | Transformer | 70.0%  | 55.0% | worse |
| lionfemale       | Transformer | 68.2%  | 72.7% | improved |
| gazellethomsons  | Transformer | 70.4%  | 63.0% | worse |

**Why it could not have helped.** A direct measurement of which cameras the
cap actually touches:

| Species          | Cameras over the 40 cap | territorial | migratory | nomadic |
|------------------|-------------------------|-------------|-----------|---------|
| buffalo          | 21 (20.8%)              | 20          | 1         | 0 |
| lionfemale       | 22 (20.0%)              | 20          | 2         | 0 |
| gazellethomsons  | 22 (16.2%)              | 21          | 1         | 0 |
| zebra            | 20 (13.5%)              | 20          | 0         | 0 |
| wildebeest       | 13 (11.0%)              | 12          | 1         | 0 |

**No nomadic camera in any species exceeds the cap**, and 94–100% of the
cameras that do are territorial. Across all five species only about 5 of
roughly 123 migratory cameras are affected at all. This follows directly from
how the labels are defined: territorial means the animal favours that site,
which mechanically produces a high detection count; migratory means brief
seasonal passage, which does not.

**Finding.** The change was a "how territorial cameras are represented" change
wearing a migratory-detection justification. It perturbed the input for the
class that was already working and could not reach the class it was meant to
help.

**Decision.** Reverted to first-N truncation. The measurement is the more
useful result: it relocates the migratory problem away from data truncation
entirely. Over 94% of migratory cameras are *already fully visible* to the
model, so the bottleneck is representation, not data loss.

---

## 3. Monthly seasonal profile as a feature — adopted

Following from experiment 2: if migratory cameras are already fully visible
and recall is still poor, the model is not extracting seasonal concentration
from raw per-detection timestamps on its own.

`compute_monthly_distribution` gives each camera a 12-value vector of the
share of its detections falling in each calendar month, computed from its
**full, uncapped** history, and concatenates it to the pooled representation
at the movement head only.

This is deliberately the *monthly counts*, not the `temporal_concentration`
scalar that the labelling rule thresholds on. Feeding the label's own input
would be close to leakage: the model would only have to learn a threshold, and
any accuracy gain would show that the statistic determines the label — which
is true by construction — rather than that the architecture generalises.

---

## 4. Feature set — deviation from the proposal

The proposal specifies hour of day, day of year, time since previous
detection, **camera identity**, and habitat type. Four of the five are
implemented. Camera identity is deliberately excluded.

Every model here is evaluated on held-out cameras — sites never seen during
training. A raw camera identifier cannot generalise to a camera absent from
the training set, so it would be either inert at evaluation time or actively
harmful, letting the model memorise training sites and score better than its
real generalisation warrants. Camera *context* that does generalise — habitat
type — is included instead.

Current vector, 11 values: `sin_hour, cos_hour, sin_day, cos_day, log_count,
log_interval_hours,` and a five-way one-hot habitat encoding.

---

## 5. Social structure — deviation from the proposal

FR5 specifies three model outputs. Activity and movement are predicted by the
network; social structure is derived by rule from the largest single-frame
instance count. This is labelled as rule-derived everywhere it is displayed
and exported, rather than being presented as a model prediction.

Adding a third head is a small change to `models.py` and `train.py`. It was
not made here because it would invalidate the comparison runs above without a
measured reason to expect it to help.

---

## 6. Schema deviations

| Table | Change | Reason |
|-------|--------|--------|
| `Detection` | added `instance_count` | Multi-instance counting post-dates the proposal's schema. The count is one pipeline's reading of one image, so Detection is the correct home. |
| `Sequence`  | added `peak_month` | Lets a query filter on the seasonal peak. Without it, a month filter parses successfully and then silently does nothing. |
| `BehaviourPattern` | split `confidence` into `activity_confidence` and `movement_confidence` | The two heads have separate confidences; one column would have had to discard one of them. |

---

## 7. Movement labels are relative, not absolute — and degrade silently

`classify_movement` compares each camera against `quantile(0.75)` of the other
cameras. It never asks whether a camera is seasonally concentrated in absolute
terms, only whether it ranks in the top quartile **of this deployment**.

Two consequences follow, and both were measured rather than assumed.

**A short deployment labels everything migratory.** With one month of data
every camera has 100% of its detections in one month, so the 75th-percentile
threshold lands on top of all of them and the `>=` comparison catches every
camera:

| Deployment | Cameras | Result |
|---|---|---|
| 1 month | 6 | 100% migratory |
| 1 month | 30 | 100% migratory |
| 1 month | 120 | 100% migratory |
| 12 months | 4 | 100% migratory (too few cameras for a quartile) |
| 12 months | 30 | 27% migratory / 33% territorial / 40% nomadic |

Note that **more cameras does not fix a short deployment, and more images
never does**. The binding constraint is months of coverage.

**The class proportions are partly imposed by the rule.** Because the
thresholds are quartiles, roughly 25% of cameras are territorial and up to 25%
migratory almost regardless of the animals. The real data does vary (zebra
12%, buffalo 22%, wildebeest 26%), but that variation comes mostly from how
many cameras fall below the three-detection floor rather than from ecology.

**Mitigation.** `wildlife_monitor/data/sufficiency.py` assesses coverage before
any behavioural claim is displayed and states, in plain language, which
outputs the data supports. The Behavioural Analysis page refuses to present
movement results as dependable when the assessment says they are not, and the
Upload page warns about a short span from EXIF timestamps alone — before any
detection has been run. This does not fix the rule; it stops the rule being
believed on data that cannot support it.

---

## 8. Correctness display replaced by confidence

The original Image Review page split detections into "correct" and "incorrect"
grids against the citizen-science labels. That only works on Snapshot
Serengeti, which ships a label for every image. A user with their own camera
trap photographs has no labels — that is why they are running a classifier —
so the split would be empty.

Worse, the dashboard mapped unknown ground truth to `False`, so unlabelled
data displayed as **0% accurate** rather than unverified.

Verification is now tri-state (`correct` / `incorrect` / `unverified`) and
accuracy is computed over labelled detections only. On a mixed test set of
1,462 detections of which 932 carry labels, the reported accuracy moved from
45.4% (664 of all 1,462, counting unlabelled as wrong) to 71.2% (664 of the
932 labelled) — the second figure being the only defensible one.

Review is now organised by model confidence, defaulting to the lowest band.
Without ground truth you cannot know which predictions are wrong, but you
always know which the model was unsure about, and those are where a
reviewer's attention belongs. The ecologist's review is then what creates
ground truth, through the `PatternValidator` verdict loop.

---

## 9. Two decisions in the external-user path

**Enlarging images is offered, and labelled as cosmetic.** A user whose photos
fall below the resolution threshold can replace them, lower the threshold, or
have the system enlarge them. Enlargement is the one option that cannot
improve a result: interpolating a 320x240 frame to 640x480 invents pixels and
recovers no detail, and §6.5 of the design document already establishes that
"stripes and rosettes are exactly what gets lost at low resolution". The image
passes the check and performs exactly as badly as before.

It is offered anyway, because a flagged result may be more useful to a user
than no result. What makes that defensible is that the flag is durable: the
`Image` table carries an `upscaled` column, it is surfaced in the flat
detection view, and the ingestion report states how many images were enlarged
and that detection on them will be no better. Format conversion, EXIF rotation
and downscaling are offered separately and without caveat, because they lose
nothing.

**Classification is gated by an animal detector.** BioCLIP scores an image
against a closed candidate list and returns the nearest species. Shown a bird,
a vehicle, or waving grass — the last being most of what a camera trap
actually records — it returns a mammal, confidently. §6.5 already found that
"a confidence threshold alone cannot be used to filter out bad predictions
because the bad ones are not necessarily low-confidence", so post-hoc
filtering cannot fix this.

MegaDetector answers an independent question: is there an animal in this frame
at all. Running it first means empty frames never reach the recogniser. The
test suite pins both halves of this — that the recogniser is never called on
an empty frame with the gate on, and that the same frames become confident
species records with it off — so the guard cannot be removed without a test
failing.

---

## 10. Two timestamp sources considered and refused

Section 2.3.2 of the design document specifies "an optional metadata CSV ...
pulls timestamps from EXIF data where the metadata does not provide them". Only
the EXIF half was implemented, which left a gap: the Serengeti runs never used
EXIF at all. They read `ann["datetime"]` out of `SnapshotSerengetiS01.json`
through `scripts/extract_ground_truth.py`, so the pipeline was being fed
curated timestamps while an external user got only whatever EXIF survived.

`data/timestamps.py` closes that gap with four sources in priority order:
annotation JSON, metadata CSV, EXIF, file name. Two further candidates were
considered and deliberately left out.

**File modification time — refused.** It is the most tempting source available,
because unlike every other one it is *always* present. That is exactly the
problem. Copying a folder, extracting a zip, or syncing to cloud storage resets
it to the moment of the copy, so a user who moved their photographs off the SD
card — which is everyone — would get every image stamped within minutes of each
other. The system would then report a deployment spanning one day, and by
section 7 above a short deployment labels **100% of cameras migratory**. A
missing timestamp is reported, counted, and excluded from behavioural analysis;
a wrong one is silently believed. `test_file_modification_time_is_never_used`
sets a plausible mtime on an otherwise anonymous photograph and asserts that
resolution still comes back empty, so the source cannot be reintroduced without
a test failing.

**A date-only file name — off by default.** `20240315.jpg` carries a real
calendar month, which is what movement classification needs, but no hour. The
only way to store it is at midnight, which makes every such image read as
nocturnal and corrupts activity classification — the one task that works on
thin data. It is available behind `--dates-from-filenames` for a user who wants
seasonal coverage and accepts losing day/night timing, and resolutions taken
that way are reported under their own source name so the count is visible rather
than folded into the file-name total.

The symmetry is deliberate: both refusals trade a *present but wrong* value for
an *absent and reported* one, which is the same choice made for the `upscaled`
flag in section 9 and for tri-state verification in section 8.

**A side benefit worth stating.** Both file-based sources may carry species
labels as well as times — an annotation JSON always does, a metadata CSV often
does. Where present these are stored in `Image.ground_truth_id`, which the
existing `DetectionFlat` view already reads, so a user who has such a file gets
measured accuracy on the Image Review page instead of the unverified display
described in section 8. No schema change was needed for this; the column was
there from the start for Serengeti's sake.

---

## 11. Held-out cameras versus held-out species

Until now every behavioural model was trained on one species and tested on
held-out *cameras* of that same species. That answers a narrow question: can
the model apply the rule to a new site for an animal it already knows.

FR5 asks for something harder, namely "a minimum classification accuracy of
75% **on held-out species**", and VL3 specifies the protocol: "train the model
excluding 15% of species entirely". `scripts/train_cross_species.py`
implements it.

**Why the distinction matters.** A per-species model only ever sees one
animal, so a high score cannot separate two very different things: learning
what a migratory detection history looks like, or memorising the quirks of one
species' cameras. Only a species the model has never seen can tell them apart.

**Leave-one-species-out rather than a single split.** With five species a
single held-out split rests on whichever species is chosen. Every species
takes a turn instead, giving one generalisation result per species. This also
makes each test set far larger than the per-species runs: a whole species of
roughly 100 to 150 cameras, rather than 20% of one species' 20 to 27.

| | Per-species model | Cross-species model |
|---|---|---|
| Models stored | one per species | one, total |
| Training cameras | about 98 | about 500 |
| Test cameras | about 25 | 101 to 148 |
| Test set is | new sites, known animal | an animal never seen |
| Serves a new species | no, must train first | yes |

**Labels are computed per species, then pooled.** This is the subtle part.
`classify_movement` compares a camera against `quantile(0.75)` of the other
cameras. Pooling the frames before computing that quantile would span a
mixture of animals and silently relabel every camera, making the cross-species
results incomparable with everything above. Computing per species and pooling
afterwards keeps every label identical.
`test_pooling_does_not_change_a_single_label` pins this, because nothing would
crash if it broke.

**A negative control guards the measurement.** A model that scores well
against randomly permuted held-out labels is measuring an artefact rather than
signal. `test_shuffled_labels_collapse_to_chance` asserts that accuracy falls
back toward the 33.3% chance baseline when the labels are shuffled. On
synthetic data the real labels score 100% and the shuffled ones 30%.

**What the result will mean.** Around 70% or better means the timing pattern
transfers and the model learned behaviour rather than an animal. Around 33% is
chance across three classes, meaning the per-species models were learning
species-specific patterns. Either is a result worth reporting, and the honest
one cannot be known until the run happens.

**One caveat that belongs with any figure produced.** Because the labels come
from a quantile rule computed within each species, this measures whether the
rule's *shape* transfers across species, not whether real animal behaviour
does. That is a narrower claim than the accuracy figure suggests.

**Memory.** Training was full-batch: every camera sequence processed
together, with the autodiff graph holding each intermediate tensor until the
backward pass finished. At roughly a hundred cameras per species that peaked
near one gigabyte and went unnoticed. Pooling five species raises it to 512
cameras, and the peak measured at **5.0 GB**, which exhausts a laptop.

`train_model` now takes an optional `batch_size`. The default stays `None`,
which is full-batch and bit-identical to every published single-species run;
`test_mini_batching_leaves_full_batch_training_unchanged` pins that. The
cross-species script passes 32.

| Batch size | Peak memory, 512 cameras |
|---|---|
| full batch (old) | 4,976 MB |
| 128 | 2,663 MB |
| 64 | 1,445 MB |
| 32 | 743 MB |

Evaluation had the same problem for a different reason: a forward pass builds
the same graph whether or not a backward pass follows, so scoring a held-out
species of 148 cameras cost as much as training on it. `predict_in_chunks`
bounds that too, and is asserted to return identical numbers to a single pass.

**Serving consequence.** `BehaviourService.load_for` prefers a species' own
model and falls back to the cross-species model. Without that fallback an
ecologist uploading a species the project never trained on gets nothing at all
from the behavioural page. The dashboard states which model answered, and
says plainly when a prediction comes from a model that has never seen that
animal.

---

## 12. Counting cameras up into species statements

Every accuracy figure in this document measures agreement with the project's
own quartile rule. None of them check the system against real animals. VL2
asks for that check: "diurnal/nocturnal classifications match known species
ecology (e.g. lions are nocturnal; giraffes are diurnal)".

The obstacle is a mismatch of units. The model predicts per camera, because
that is where the data is and because "is this site a corridor or a home
range" is a question no textbook answers. Published ecology is written about
species. `wildlife_monitor/pipeline2/aggregate.py` bridges the two by counting
each species' camera classifications into one label.

**What it deliberately does not do.** It holds no literature and makes no
comparison. Which source is authoritative for a species is a judgement for the
ecologist, and a system that graded itself against a hard-coded table would
look like validation without being it.
`test_the_module_never_claims_agreement_with_literature` asserts that no such
table creeps in later.

**Three ways a count could mislead, and what stops each.**

| Risk | Guard |
|---|---|
| A 40% majority read as a finding | A label must hold 50% of the vote and lead by 15 points before it is called clear; otherwise the result reads `split` |
| A camera with two detections voting on day-or-night | Cameras below five detections are excluded and the exclusion is stated |
| A stray or empty label winning by frequency | Only the defined classes are counted |

**Model against rule.** The rule labels are summarised beside the predictions.
Where the two agree, a mismatch with published ecology points at the labelling
rule. Where they disagree, it points at the model. Without that column a
mismatch is uninterpretable.

**An early observation.** On a trial run, activity recovered the expected
pattern cleanly, while movement came out `split` at roughly 35% for every
species. That is the quartile rule doing what section 7 describes: forcing
about a third of cameras into each class regardless of the animal. The
aggregation surfaces that as a refusal to make a claim rather than as a
confident wrong answer, which is the behaviour intended.

---

## 13. Interpreting the accuracy figures

Two caveats apply to every number in this document.

**The labels are heuristic.** Movement classes come from a rule over site
fidelity and seasonal concentration, not from observed animal behaviour. The
accuracy figures measure agreement with that rule on unseen cameras. A model
that scored 100% would have learned the rule perfectly, which is not the same
as being correct about the animals.

**The test sets are small.** Each species holds out 20–30 cameras, of which
3–7 are migratory. One flipped prediction moves migratory recall by 14–33
percentage points. Differences of a few points between configurations are
within noise at this sample size, and none of the conclusions above rest on a
difference that small — the rejected changes failed by 10–20 points, or failed
on the mechanism rather than the number.
