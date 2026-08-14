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
"""

from wildlife_monitor.pipelines.bioclip_sam import BioCLIPSAMPipeline
from wildlife_monitor.pipelines.bioclip_yolo import BioCLIPYOLOPipeline
from wildlife_monitor.pipelines.megadetector import MegaDetectorPipeline
from wildlife_monitor.pipelines.compare import PipelineComparator

PIPELINE_REGISTRY: dict[str, type] = {
    "bioclip_sam":          BioCLIPSAMPipeline,
    "bioclip_yolo":         BioCLIPYOLOPipeline,
    "bioclip_megadetector": MegaDetectorPipeline,
}

__all__ = [
    "BioCLIPSAMPipeline",
    "BioCLIPYOLOPipeline",
    "MegaDetectorPipeline",
    "PipelineComparator",
    "PIPELINE_REGISTRY",
]
