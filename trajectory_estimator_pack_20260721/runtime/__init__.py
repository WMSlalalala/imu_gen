"""Runtime interfaces for shared-plan trajectory/IMU generation."""

from .trajectory_layer import TrajectoryDiffusionLayer
from .paired_layer import (
    PairedGenerationService,
    audit_paired_generation,
    cross_modal_consistency_features,
    cross_modal_consistency_from_record,
)

__all__ = [
    "TrajectoryDiffusionLayer", "PairedGenerationService",
    "audit_paired_generation", "cross_modal_consistency_features",
    "cross_modal_consistency_from_record",
]
