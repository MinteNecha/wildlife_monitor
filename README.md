# Wildlife Monitor

Multimodal wildlife monitoring for camera trap imagery. The system identifies
and localises animals in Snapshot Serengeti camera trap images (Pipeline 1),
then learns behavioural patterns from the resulting detection sequences
(Pipeline 2), and presents both through a dashboard aimed at conservation
ecologists rather than at a terminal.

## The two pipelines

**Pipeline 1 — detection and localisation (P2).** BioCLIP supplies zero-shot
species recognition from a text prompt; three interchangeable localisation
backends turn that into a spatial result.

| Pipeline               | Recognition | Localisation           | Output                      |
|------------------------|-------------|------------------------|-----------------------------|
| `bioclip_sam`          | BioCLIP     | SAM 1 mask             | Multi-instance pixel masks  |
| `bioclip_yolo`         | BioCLIP     | YOLOv11 bounding box   | Single best box; fastest    |
| `bioclip_megadetector` | BioCLIP     | MegaDetector boxes     | All animals per frame + count |

**Pipeline 2 — temporal behavioural analysis (P3).** One camera's detection
history becomes one sequence. An LSTM and a Transformer encoder — both built
on this project's own NumPy autodiff engine, no deep-learning framework —
classify each sequence into activity timing and movement strategy, with social
structure derived by rule from instance counts.

| Task             | Classes                                   | Source           |
|------------------|-------------------------------------------|------------------|
| Activity timing  | diurnal · nocturnal · crepuscular         | model prediction |
| Movement strategy| migratory · territorial · nomadic         | model prediction |
| Social structure | solitary · small group · large herd       | rule (instance count) |

## Project layout

```
wildlife_monitor/
  wildlife_monitor/
    config/      settings.py    paths, model IDs, SystemConfig singleton (P4)
                 validator.py   ConfigValidator — FR9 range checking
    data/        loader.py      dataset acquisition + per-species loading (P1)
                 validator.py   ImageValidator — FR1 ingestion checks
                 export.py      ExportService — FR8 CSV / JSON / PDF
    db/          schema.sql     third-normal-form schema (section 5)
                 connection.py  connections, schema application
                 repository.py  one repository per table; DetectionFlat view
    models/      bioclip.py sam1.py sam3.py yolo.py megadetector.py   (P2)
    pipelines/   base.py compare.py + the three concrete pipelines    (P2)
    pipeline2/   autodiff.py         Tensor, autograd, Adam, clipping
                 feature_extractor.py TemporalFeatureExtractor — FR4
                 sequence_builder.py  SequenceBuilder, DetectionSequence
                 labelling.py         the behavioural label rules
                 models.py            BehaviourModel + LSTM + Transformer — FR5
                 train.py             training loop and metrics
                 inference.py         BehaviourService — serving predictions
                 query.py             QueryEngine — FR6 rule-based NL parsing
                 validation.py        PatternValidator — FR7 ecologist verdicts
    dashboard/   app.py, theme.py, components.py, data_access.py
                 views/       nine pages, one render() each
    utils/       records.py validation.py visualisation.py
  scripts/       setup_data.py init_db.py import_results.py
                 download_subset_images.py extract_ground_truth.py
                 run_pipeline.py run_compare.py train_behaviour.py
                 evaluate.py evaluate_counts.py compare_models.py
  tests/         pytest suite (85 tests)
  data/          wildlife.db, subset metadata, images
  models/        detector checkpoints + behaviour/*.npz
  results/       overlays, masks, comparison reports, behaviour metrics
```

## Install

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell
source venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
pip install -e .                  # makes 'wildlife_monitor' importable
```

## Setup

```bash
python scripts/setup_data.py     # directories, SAM checkpoint, subset check
python scripts/init_db.py        # create data/wildlife.db from schema.sql
```

If you have detection CSVs from before the database existed, import them
rather than recomputing:

```bash
python scripts/import_results.py --subset
```

## Run

```bash
# Pipeline 1 — detection
python scripts/run_pipeline.py --pipeline bioclip_megadetector --species zebra --top_n 1500
python scripts/run_compare.py --species zebra --top_n 15

# Pipeline 2 — behaviour
python scripts/train_behaviour.py --species zebra --model lstm
python scripts/train_behaviour.py --all --model transformer

# Dashboard
streamlit run wildlife_monitor/dashboard/app.py
```

`--top_n` defaults to 15, which is a sampling run, not a full one. Pass it
explicitly for real runs.

## Storage

SQLite (`data/wildlife.db`) is the system of record, in third normal form.
Pipelines write detections there; Pipeline 2 reads its sequences from there;
the dashboard reads everything it displays from there. Large binaries — JPEGs,
masks, overlays — stay on disk, with their paths stored in the tables.

CSV is an export format, not a storage format. `ExportService` produces CSV,
JSON or a PDF report on demand (FR8). Archived CSVs still work as a fallback
everywhere, so results predating the migration remain usable.

The `DetectionFlat` view reassembles the normalised tables into the wide row
that Pipeline 2 and the dashboard expect, so normalisation is invisible to
consumers.

## Dashboard

Nine pages, each mapped to the use cases it serves:

| Page                  | Use cases | Requirements |
|-----------------------|-----------|--------------|
| Overview              | UC2, UC3  | FR2, FR3     |
| Upload                | UC1       | FR1          |
| Species Detection     | UC2       | FR2          |
| Detection Map         | UC5       | FR6          |
| Behavioural Analysis  | UC4, UC5, UC6 | FR4, FR5, FR6, FR7 |
| Pipeline Comparison   | UC3       | FR3          |
| Image Review          | UC2, UC6  | FR2, FR7     |
| Settings              | UC8       | FR9          |
| Export                | UC7       | FR8          |

## Tests

```bash
python -m pytest tests/ -q
```

The autodiff tests compare every backward pass against a central finite
difference. A wrong gradient does not crash — it quietly trains the wrong
thing — so these are the suite's most important tests.

## What the data must look like

The behavioural rules compare each camera against the others, so they degrade
quietly rather than failing loudly on thin data. `data/sufficiency.py` checks
three things before any result is presented:

| Requirement | Minimum | Dependable | Why |
|---|---|---|---|
| **Time span** | 3 months | 6+ months | Migratory and territorial differ by seasonal spread. Under 3 months every camera looks equally concentrated and *every* camera is labelled migratory. |
| **Cameras** | 10 | 30+ | Movement classes are assigned by percentile across cameras. With 4 cameras, "top 25%" is one camera. |
| **Detections per camera** | 3 | 10+ | Three is the coded floor for migratory eligibility; below ten the day/night vote flips on one or two photographs. |

The headline for a new user: **this needs months, not more photographs.** A
thousand images from a two-week deployment cannot support movement
classification. Thirty cameras over six months can.

Activity timing (day/night) and social structure (group size) need no
particular time span and work on much less data.

### What an external user supplies

1. Images in one folder per camera site
2. A four-column `cameras.csv` — `camera_id, latitude, longitude, habitat_type`
3. Optionally a species shortlist; otherwise the default 13 are used

Capture times are read from EXIF. No species labels are required — and where
none exist, detections are shown as **unverified** rather than counted wrong,
with model confidence as the available signal.

## Known limitations

These are stated plainly because they affect how the results should be read.
`docs/experiments.md` records the changes that were tested and rejected, with
the measurements behind each decision.

- **Movement labels are heuristic, not observed.** They come from a rule over
  site fidelity and seasonal concentration. Accuracy figures therefore measure
  agreement with that heuristic on unseen cameras, not against independently
  observed animal behaviour.
- **Migratory recall is modest** (roughly 20–60% depending on species and
  architecture), and the migratory class has only 3–7 held-out cameras per
  species, so a single prediction moves recall by more than ten points.
- **Social structure is rule-derived**, not a model output, and is labelled as
  such everywhere it appears.
- **Camera identity is deliberately excluded** from the feature vector. Models
  are evaluated on held-out cameras, so a raw camera identifier cannot
  generalise and would only let the model memorise training sites.
- **Movement classes are relative to one deployment.** They say which cameras
  stand out among the cameras supplied, not whether an animal migrates in an
  absolute sense. Roughly a quarter of cameras come out territorial by
  construction.

## Optional: enable SAM 3

The `sam3` path falls back to SAM 1 until the gated weights are present:

1. `pip install -U ultralytics` (>= 8.3.237)
2. Request access and download `sam3.pt` from <https://huggingface.co/facebook/sam3>
3. Place `sam3.pt` in `models/`

## Target species

`buffalo, cheetah, elephant, giraffe, leopard, wildebeest, zebra, lionmale,
lionfemale, lioncub, hyenaspotted, hyenabrown, gazellethomsons`
