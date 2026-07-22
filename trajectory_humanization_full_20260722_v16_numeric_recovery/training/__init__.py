"""Strict, leakage-auditable trajectory diffusion training pipeline."""

from .corpus import (
    ACTIONS,
    FORMAL_SPLIT_SHA256,
    NumericTrajectoryCorpus,
    SplitDefinition,
    audit_corpus_directory,
    canonical_sample_sha256,
)
from .fewshot_dataset import (
    ReferenceRegistry,
    StrictFiveReferenceDataset,
    StrictVariableLengthCollator,
)

__all__ = [
    "ACTIONS",
    "FORMAL_SPLIT_SHA256",
    "NumericTrajectoryCorpus",
    "SplitDefinition",
    "ReferenceRegistry",
    "StrictFiveReferenceDataset",
    "StrictVariableLengthCollator",
    "audit_corpus_directory",
    "canonical_sample_sha256",
]
