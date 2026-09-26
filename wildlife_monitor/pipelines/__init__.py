"""
Pipeline registry — maps pipeline names to concrete pipeline classes.

Three detection pipelines — each uses BioCLIP for species identification
and a different localisation backend:

    bioclip_sam          Pipeline 1a — BioCLIP + SAM 3
                         Text-prompted concept segmentation.
                         Multi-instance masks + count.
                         Most precise. Slowest on CPU.

    bioclip_yolo         Pipeline 1b — BioCLIP + YOLOv11
                         Bounding box detection.
                         Single best box. Fastest.

    bioclip_megadetector Pipeline 1c — BioCLIP + MegaDetector
                         Multi-instance bounding boxes.
                         All animals per frame + count.
                         Moderate speed. Count-capable.

Imports here are deliberately lazy. Each pipeline pulls in a heavy computer
vision stack (torch, ultralytics, segment-anything, megadetector), and
importing this package eagerly would force all of that on every consumer —
including the dashboard, which only reads result files, and Pipeline 2, which
runs on the project's own NumPy autodiff engine. Naming a pipeline is now free;
only constructing one loads its models.
"""

from __future__ import annotations

from typing import Any

# Pipeline name -> "module path:class name". Resolved on demand.
_PIPELINE_PATHS: dict[str, str] = {
    "bioclip_sam": "wildlife_monitor.pipelines.bioclip_sam:BioCLIPSAMPipeline",
    "bioclip_yolo": "wildlife_monitor.pipelines.bioclip_yolo:BioCLIPYOLOPipeline",
    "bioclip_megadetector": "wildlife_monitor.pipelines.megadetector:MegaDetectorPipeline",
}

_LAZY_ATTRS: dict[str, str] = {
    **{path.rsplit(":", 1)[1]: path for path in _PIPELINE_PATHS.values()},
    "PipelineComparator": "wildlife_monitor.pipelines.compare:PipelineComparator",
}

#: Pipeline names, available without importing any model code.
PIPELINE_NAMES: tuple[str, ...] = tuple(_PIPELINE_PATHS)


def _resolve(target: str) -> Any:
    from importlib import import_module
    module_path, attribute = target.split(":")
    return getattr(import_module(module_path), attribute)


def get_pipeline(name: str) -> type:
    """Return a pipeline class by name, importing it on first use."""
    if name not in _PIPELINE_PATHS:
        raise KeyError(f"Unknown pipeline '{name}'. "
                        f"Expected one of: {', '.join(PIPELINE_NAMES)}")
    return _resolve(_PIPELINE_PATHS[name])


def __getattr__(name: str) -> Any:
    """Lazily expose pipeline classes and the eager-style registry (PEP 562)."""
    if name == "PIPELINE_REGISTRY":
        return {key: get_pipeline(key) for key in _PIPELINE_PATHS}
    if name in _LAZY_ATTRS:
        return _resolve(_LAZY_ATTRS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BioCLIPSAMPipeline",
    "BioCLIPYOLOPipeline",
    "MegaDetectorPipeline",
    "PipelineComparator",
    "PIPELINE_NAMES",
    "PIPELINE_REGISTRY",
    "get_pipeline",
]
