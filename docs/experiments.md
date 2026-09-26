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

## 7. Interpreting the accuracy figures

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
