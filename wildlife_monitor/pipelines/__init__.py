"""Pipeline package — the abstract base, three pipelines, and comparator."""

from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.pipelines.bioclip_sam import BioCLIPSAMPipeline
from wildlife_monitor.pipelines.bioclip_yolo import BioCLIPYOLOPipeline
from wildlife_monitor.pipelines.sam3 import SAM3Pipeline
from wildlife_monitor.pipelines.compare import PipelineComparator

#: Registry mapping pipeline names to their classes, for the CLI.
PIPELINE_REGISTRY = {
    BioCLIPSAMPipeline.name: BioCLIPSAMPipeline,
    BioCLIPYOLOPipeline.name: BioCLIPYOLOPipeline,
    SAM3Pipeline.name: SAM3Pipeline,
}

__all__ = [
    "DetectionPipeline", "LocalisationResult",
    "BioCLIPSAMPipeline", "BioCLIPYOLOPipeline", "SAM3Pipeline",
    "PipelineComparator", "PIPELINE_REGISTRY",
]
