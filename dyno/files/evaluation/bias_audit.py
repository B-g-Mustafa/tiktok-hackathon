"""
Dataset Shortcut and Bias Audit (Grommelt et al., ECCV 2024).
Verifies whether a benchmark allows trivial classification shortcuts based on:
- Resolution (Width, Height, Total Pixels)
- Aspect Ratio (Width / Height)
- File Size in bytes
- Bytes per pixel (Compression ratio proxy)
- File format (PNG vs JPEG vs WebP)
- JPEG Quantization table variance
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from datasets.sid import scan_directory_for_samples


def extract_metadata_features(image_path: str) -> Dict[str, Any]:
    """
    Extract non-pixel metadata features from an image file.
    Does not inspect pixel semantic content.
    """
    path = Path(image_path)
    file_size = path.stat().st_size if path.exists() else 0
    ext = path.suffix.lower()

    # Image dimension and format inspection
    width, height = 0, 0
    aspect_ratio = 1.0
    is_jpeg = 1 if ext in [".jpg", ".jpeg"] else 0
    is_png = 1 if ext == ".png" else 0
    is_webp = 1 if ext == ".webp" else 0
    q_table_mean = 0.0
    q_table_std = 0.0

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / max(1, height)
            
            # Extract quantization tables if present in JPEG
            if hasattr(img, "quantization") and img.quantization:
                q_tables = [np.array(tbl) for tbl in img.quantization.values()]
                if q_tables:
                    concat_q = np.concatenate(q_tables)
                    q_table_mean = float(np.mean(concat_q))
                    q_table_std = float(np.std(concat_q))
    except Exception:
        pass

    total_pixels = max(1, width * height)
    bytes_per_pixel = file_size / total_pixels

    return {
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "aspect_ratio": aspect_ratio,
        "file_size_bytes": file_size,
        "bytes_per_pixel": bytes_per_pixel,
        "is_jpeg": is_jpeg,
        "is_png": is_png,
        "is_webp": is_webp,
        "q_table_mean": q_table_mean,
        "q_table_std": q_table_std,
    }


def run_bias_audit(
    data_dir: str,
    output_report_path: Optional[str] = None,
    max_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs Grommelt et al. metadata-only bias audit on the specified dataset.
    """
    print(f"=== Running Metadata Bias Audit on {data_dir} ===")
    samples = scan_directory_for_samples(data_dir)

    if not samples:
        print(f"No samples found in {data_dir}")
        return {"metadata_auc": 0.5, "status": "no_data"}

    if len(samples) > max_samples:
        np.random.seed(seed)
        indices = np.random.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in indices]

    print(f"Extracting metadata features from {len(samples)} samples...")
    feature_list = []
    labels = []

    for path, label in samples:
        feats = extract_metadata_features(path)
        feature_list.append(feats)
        labels.append(label)

    df = pd.DataFrame(feature_list)
    y = np.array(labels)

    if len(np.unique(y)) < 2:
        print("Warning: Only one class found. Cannot compute AUC.")
        return {"metadata_auc": 0.5, "status": "single_class"}

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Train Random Forest on metadata only
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)
    rf.fit(X_train, y_train)

    preds_prob = rf.predict_proba(X_test)[:, 1]
    preds_binary = rf.predict(X_test)

    auc = float(roc_auc_score(y_test, preds_prob))
    acc = float(accuracy_score(y_test, preds_binary))

    # Feature importances
    feature_importances = dict(zip(df.columns, rf.feature_importances_))
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Metadata Audit Results ---")
    print(f"Metadata Baseline ROC-AUC: {auc:.4f}")
    print(f"Metadata Baseline Accuracy: {acc:.4f}")
    print("\nTop Feature Importances:")
    for feat_name, imp in sorted_importances[:5]:
        print(f"  - {feat_name}: {imp:.4f}")

    has_severe_bias = auc > 0.85
    has_moderate_bias = 0.70 < auc <= 0.85

    if has_severe_bias:
        print("\n[CRITICAL WARNING] Severe dataset shortcut detected (Metadata AUC > 0.85)!")
        print("Models trained naively on this dataset will exploit resolution / compression shortcuts.")
    elif has_moderate_bias:
        print("\n[WARNING] Moderate metadata correlation detected (0.70 < Metadata AUC <= 0.85).")
    else:
        print("\n[PASSED] No significant metadata shortcut detected (Metadata AUC <= 0.70). Dataset is clean.")

    results = {
        "metadata_auc": auc,
        "metadata_acc": acc,
        "feature_importances": dict(sorted_importances),
        "severe_bias": has_severe_bias,
        "moderate_bias": has_moderate_bias,
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            f.write("# Grommelt et al. Dataset Bias Audit Report\n\n")
            f.write(f"- **Dataset Path**: `{data_dir}`\n")
            f.write(f"- **Samples Audited**: {len(samples)}\n")
            f.write(f"- **Metadata-Only ROC-AUC**: **{auc:.4f}**\n")
            f.write(f"- **Metadata-Only Accuracy**: **{acc:.4f}**\n")
            f.write(f"- **Shortcut Diagnosis**: {'Severe Leakage' if has_severe_bias else ('Moderate Leakage' if has_moderate_bias else 'Clean Benchmark')}\n\n")
            f.write("### Feature Importance Breakdown\n\n")
            f.write("| Feature | Importance |\n|---|---|\n")
            for feat, imp in sorted_importances:
                f.write(f"| `{feat}` | {imp:.4f} |\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grommelt et al. Dataset Bias Audit")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing real/fake images")
    parser.add_argument("--output_report", type=str, default="./outputs/bias_audit_report.md", help="Output report markdown path")
    args = parser.parse_args()

    run_bias_audit(args.data_dir, args.output_report)
