"""Evaluation and Forensics package for Robust AIGC Detector."""

from .bias_audit import run_bias_audit, extract_metadata_features
from .robustness import evaluate_robustness_suite
from .ood import evaluate_wildfake, evaluate_genimage_generalization
from .bootstrap_ci import compute_bootstrap_auc_ci
from .error_analysis import perform_error_analysis

__all__ = [
    "run_bias_audit",
    "extract_metadata_features",
    "evaluate_robustness_suite",
    "evaluate_wildfake",
    "evaluate_genimage_generalization",
    "compute_bootstrap_auc_ci",
    "perform_error_analysis",
]
