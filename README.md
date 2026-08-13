# Wildlife Monitor

Multimodal wildlife monitoring for camera trap imagery. The system
identifies and localises animals in Snapshot Serengeti camera trap images
using three configurable pipelines, then compares them side by side.

## Pipelines

| Pipeline       | Recognition | Localisation            | Notes                          |
|----------------|-------------|-------------------------|--------------------------------|
| `bioclip_sam`  | BioCLIP     | SAM 1 mask              | Best spatial precision         |
| `bioclip_yolo` | BioCLIP     | YOLOv11 bounding box    | Fastest; large-scale runs      |
| `sam3`         | SAM 3 text  | SAM 3 concept mask      | Single model; SAM 1 fallback   |

BioCLIP supplies zero-shot species recognition from a text prompt. The
`bioclip_sam` pipeline deliberately stays on SAM 1 — the known-good
combination — while the standalone `sam3` pipeline uses Meta's latest
SAM 3 (Nov 2025) for true text-prompted concept segmentation.

## Project layout

```
wildlife_monitor/
  wildlife_monitor/          Python package
    config/    settings.py   central config + SystemConfig singleton (P4)
    data/      loader.py      dataset acquisition + per-species loading (P1)
    models/    bioclip.py     shared BioCLIP retrieval (P2)
               sam1.py        SAM 1 segmenter (P2)
               sam3.py        SAM 3 concept segmenter + SAM 1 fallback (P2)
               yolo.py        YOLOv11 detector (P2)
    pipelines/ base.py        abstract DetectionPipeline (P2)
               bioclip_sam.py, bioclip_yolo.py, sam3.py
               compare.py     three-way PipelineComparator
    utils/     records.py     DetectionRecord contract + CSV repository
               visualisation.py  drawing + overlay helpers
  scripts/
    setup_data.py            one-time environment + checkpoint setup
    run_pipeline.py          run one pipeline
    run_compare.py           run and compare all three
  data/                      subset_metadata.csv + images/ go here
  models/                    checkpoints (auto-downloaded where possible)
  results/                   pipeline output (created at runtime)
  requirements.txt
```

## Install

```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .          # makes 'wildlife_monitor' importable
```

## Setup

```bash
python scripts/setup_data.py
```

This creates the directories, downloads the SAM 1 checkpoint, checks for
your `data/subset_metadata.csv`, and reports whether the optional SAM 3
weights are present.

## Run

```bash
# One pipeline
python scripts/run_pipeline.py --pipeline bioclip_sam --species zebra
python scripts/run_pipeline.py --pipeline sam3 --species elephant --top_n 20

# All three + comparison report
python scripts/run_compare.py --species zebra --top_n 15
```

Output is written under `results/<pipeline>/` (detections CSV, overlays,
masks) and `results/comparison/` (report + three-way visuals).

## Optional: enable SAM 3

The `sam3` pipeline falls back to SAM 1 until the gated SAM 3 weights are
present:

1. `pip install -U ultralytics` (>= 8.3.237)
2. Request access and download `sam3.pt` from
   https://huggingface.co/facebook/sam3
3. Place `sam3.pt` in `models/`

## Target species

`buffalo, cheetah, elephant, giraffe, leopard, wildebeest, zebra,
lionmale, lionfemale, lioncub, hyenaspotted, hyenabrown, gazellethomsons`
