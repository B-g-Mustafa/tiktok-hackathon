"""
Non-Parametric Bootstrap Confidence Intervals for Metric Reliability.
Ensures evaluation results are statistically rigorous (e.g., AUC = 0.942 +/- 0.006).
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score


def compute_bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute non-parametric bootstrap confidence interval for ROC-AUC.
    Returns:
        mean, std, ci_lower, ci_upper, formatted_string
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    bootstrapped_aucs: List[float] = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(score)

    if not bootstrapped_aucs:
        return {
            "mean": 0.5,
            "std": 0.0,
            "ci_lower": 0.5,
            "ci_upper": 0.5,
            "formatted": "0.5000 +/- 0.0000",
        }

    bootstrapped_aucs = np.array(bootstrapped_aucs)
    mean_auc = float(np.mean(bootstrapped_aucs))
    std_auc = float(np.std(bootstrapped_aucs))
    
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(bootstrapped_aucs, alpha * 100))
    ci_upper = float(np.percentile(bootstrapped_aucs, (1.0 - alpha) * 100))

    formatted = f"{mean_auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (+/- {std_auc:.4f})"

    return {
        "mean": mean_auc,
        "std": std_auc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "formatted": formatted,
    }
