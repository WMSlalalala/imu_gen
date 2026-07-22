"""Runtime wrappers for trajectory PAD / human-likeness estimators."""

from .feature_estimator import FeatureEstimatorArtifact, feature_vector_from_record
from .deep_estimator import DeepEstimatorArtifact
from .service import TrajectoryEstimatorService
from .total_detector import TotalDetectorArtifact
from .paired_dataset import PAIRED_DATASET_SCHEMA, PairedDetectorTable
from .paired_dataset_builder import (
    COMPONENT_TABLE_SCHEMA,
    DetectorComponentTable,
    build_paired_detector_table,
)
from .duration_metrics import fit_duration_bins, duration_stratified_metrics
from .real_pair_index import REAL_PAIR_INDEX_SCHEMA, build_real_pair_index
from .consistency_component import build_real_consistency_component
from .fake_event_plan_archive import load_event_plans_from_archive
from .fake_imu_pairs import build_fake_imu_unit, validate_fake_imu_unit
from .fake_consistency_component import build_fake_consistency_component
from .component_merge import merge_component_tables
from .meta_pool import remap_component_to_meta_pools
from .trajectory_score_component import build_trajectory_score_component
from .trajectory_duration_report import (
    build_trajectory_duration_report,
    validate_trajectory_duration_report,
)
from .total_detector_audit import validate_total_detector_outputs
from .trajectory_release import (
    build_trajectory_estimator_release,
    validate_trajectory_estimator_release,
)
from .runtime_benchmark import (
    benchmark_trajectory_estimator_latency,
    benchmark_total_detector_latency,
)
from .paired_imu_scorer import (
    build_paired_imu_feature_table,
    load_paired_imu_feature_table,
    train_paired_imu_scorers,
)
from .trajectory_kinematics_audit import audit_trajectory_kinematics

__all__ = [
    "FeatureEstimatorArtifact",
    "DeepEstimatorArtifact",
    "TrajectoryEstimatorService",
    "TotalDetectorArtifact",
    "PAIRED_DATASET_SCHEMA",
    "PairedDetectorTable",
    "COMPONENT_TABLE_SCHEMA",
    "DetectorComponentTable",
    "build_paired_detector_table",
    "fit_duration_bins",
    "duration_stratified_metrics",
    "REAL_PAIR_INDEX_SCHEMA",
    "build_real_pair_index",
    "build_real_consistency_component",
    "load_event_plans_from_archive",
    "build_fake_imu_unit",
    "validate_fake_imu_unit",
    "build_fake_consistency_component",
    "merge_component_tables",
    "remap_component_to_meta_pools",
    "build_trajectory_score_component",
    "build_trajectory_duration_report",
    "validate_trajectory_duration_report",
    "validate_total_detector_outputs",
    "build_trajectory_estimator_release",
    "validate_trajectory_estimator_release",
    "benchmark_trajectory_estimator_latency",
    "benchmark_total_detector_latency",
    "build_paired_imu_feature_table",
    "load_paired_imu_feature_table",
    "train_paired_imu_scorers",
    "audit_trajectory_kinematics",
    "feature_vector_from_record",
]
