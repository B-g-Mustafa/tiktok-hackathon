"""
Systematic Robustness Evaluation Suite for AIGC Detection.
Evaluates detector across all individual and compound transformations:
- Clean
- JPEG Compression (Q=90, 70, 50, 30)
- Gaussian Blur (sigma=0.5, 1.0, 2.0)
- Resize (0.5x, 0.25x)
- Gaussian Noise (sigma=0.02, 0.05, 0.10)
- Color Jitter (+/-20%)
- Center Crop (80%)
- Compound Degradation Chains
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from PIL import Image

from datasets.base import load_image_rgb
from datasets.sid import scan_directory_for_samples, DirectParquetDataset, decode_image_bytes
from transforms import (
    BENCHMARK_DISTORTIONS,
    COMPOUND_BENCHMARK,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_clean_transform,
)
from models import build_detector, ConsistencyDetectorWrapper, DualStreamDetector
from evaluation.bootstrap_ci import compute_bootstrap_auc_ci


def transform_and_normalize(
    image: np.ndarray,
    distortion_fn: Any,
    image_size: int = 384,
) -> torch.Tensor:
    """Apply distortion and return normalized tensor (3, H, W)."""
    distorted = distortion_fn(image)
    pipeline = get_clean_transform(image_size=image_size)
    res = pipeline(image=distorted)
    return res["image"] if isinstance(res, dict) and "image" in res else res


def evaluate_robustness_suite(
    model: nn.Module,
    images_raw: List[np.ndarray],
    labels: List[int],
    device: torch.device,
    image_size: int = 384,
    batch_size: int = 64,
    include_compound: bool = True,
    n_bootstraps: int = 500,
) -> Dict[str, Any]:
    """
    Run full robustness sweep across all transformations.
    """
    model.eval()

    all_distortions = dict(BENCHMARK_DISTORTIONS)
    if include_compound:
        all_distortions.update(COMPOUND_BENCHMARK)

    y_true = np.array(labels)
    n_total = len(images_raw)

    print(f"\nRunning Robustness Benchmark across {len(all_distortions)} distortions on {n_total} samples...")

    results_table = []
    clean_auc = 0.5

    for dist_name, dist_fn in all_distortions.items():
        preds = []
        for i in range(0, n_total, batch_size):
            batch_imgs = images_raw[i:i + batch_size]
            tensors = [transform_and_normalize(img, dist_fn, image_size) for img in batch_imgs]
            batch_tensor = torch.stack(tensors).to(device)

            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs.tolist())

        y_pred = np.array(preds)
        try:
            auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
        ci_res = compute_bootstrap_auc_ci(y_true, y_pred, n_bootstraps=n_bootstraps)

        if dist_name == "clean":
            clean_auc = auc
            delta_auc = 0.0
        else:
            delta_auc = clean_auc - auc

        results_table.append({
            "transformation": dist_name,
            "auc": auc,
            "acc": acc,
            "delta_auc": delta_auc,
            "ci_lower": ci_res["ci_lower"],
            "ci_upper": ci_res["ci_upper"],
            "ci_formatted": ci_res["formatted"],
        })

    df_results = pd.DataFrame(results_table)
    non_clean_df = df_results[df_results["transformation"] != "clean"]
    mean_robust_auc = float(non_clean_df["auc"].mean())
    max_degradation = float(non_clean_df["delta_auc"].max())

    print("\n" + "=" * 65)
    print("           ROBUSTNESS EVALUATION BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Clean AUROC:        {clean_auc:.4f}")
    print(f"Mean Robust AUROC:  {mean_robust_auc:.4f}")
    print(f"Max Delta-AUC:      {max_degradation:.4f}")
    print("-" * 65)
    for _, row in df_results.iterrows():
        print(f"{row['transformation']:<28} | AUC: {row['auc']:.4f} | Acc: {row['acc']:.4f} | Delta-AUC: {row['delta_auc']:+.4f}")
    print("=" * 65 + "\n")

    return {
        "clean_auc": clean_auc,
        "mean_robust_auc": mean_robust_auc,
        "max_delta_auc": max_degradation,
        "detailed_table": df_results.to_dict(orient="records"),
        "dataframe": df_results,
    }


def main():
    parser = argparse.ArgumentParser(description="AIGC Detector Robustness Evaluation Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default=None, help="Evaluation dataset directory")
    parser.add_argument("--output_report", type=str, default="outputs/robustness_report.md", help="Output Markdown report path")
    parser.add_argument("--max_samples", type=int, default=1000, help="Number of evaluation samples to benchmark")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and load checkpoint
    from training.utils import load_checkpoint
    detector = build_detector(config)
    if Path(args.checkpoint).exists():
        print(f"Loading checkpoint from: {args.checkpoint}")
        load_checkpoint(args.checkpoint, detector)
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Running with current weights.")

    detector.to(device)
    detector.eval()

    # Determine evaluation samples
    data_dir = Path(args.data_dir or config.get("data", {}).get("val_data_dir", "./data/sid_set/data"))
    images_raw: List[np.ndarray] = []
    labels: List[int] = []

    parquet_files = list(data_dir.glob("*.parquet")) + list(data_dir.rglob("*.parquet"))
    if parquet_files:
        val_parquets = [p for p in parquet_files if any(k in p.name.lower() for k in ["validation", "val", "test", "eval"])]
        if not val_parquets:
            val_parquets = parquet_files[:5]
        
        print(f"Loading evaluation images from {len(val_parquets)} Parquet files in {data_dir}...")
        ds = DirectParquetDataset(parquet_files=val_parquets, max_shards=5)
        n_samples = min(len(ds), args.max_samples)
        indices = np.linspace(0, len(ds) - 1, n_samples, dtype=int)
        for idx in indices:
            raw_val = ds.img_column[int(idx)].as_py()
            lbl_val = ds.lbl_column[int(idx)].as_py()
            try:
                img_np = decode_image_bytes(raw_val)
                label = 1 if lbl_val >= 0.5 else 0
                images_raw.append(img_np)
                labels.append(label)
            except Exception:
                continue
    else:
        print(f"Scanning directory {data_dir} for image files...")
        samples = scan_directory_for_samples(data_dir)
        if not samples:
            raise FileNotFoundError(f"No image or parquet samples found in {data_dir}")
        np.random.seed(42)
        np.random.shuffle(samples)
        samples = samples[:args.max_samples]
        for p, y in samples:
            try:
                img = load_image_rgb(p)
                images_raw.append(img)
                labels.append(y)
            except Exception:
                continue

    print(f"Successfully loaded {len(images_raw)} test images ({labels.count(0)} authentic, {labels.count(1)} synthetic).")

    # Run Benchmark
    image_size = config.get("data", {}).get("image_size", 384)
    results = evaluate_robustness_suite(
        model=detector,
        images_raw=images_raw,
        labels=labels,
        device=device,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    # Save Markdown Report
    out_report_path = Path(args.output_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = results["dataframe"]
    report_md = f"""# Robust AIGC Detection: Benchmark Robustness Report

**Model**: `{config.get('model', {}).get('backbone', 'DINOv2-Large')}`  
**Checkpoint**: `{args.checkpoint}`  
**Evaluated Samples**: {len(images_raw)} (Real: {labels.count(0)}, Synthetic: {labels.count(1)})  
**Clean AUROC**: `{results['clean_auc']:.4f}`  
**Mean Robust AUROC**: `{results['mean_robust_auc']:.4f}`  
**Max Degradation (\\Delta AUC)**: `{results['max_delta_auc']:.4f}`  

---

## Transformation Benchmark Results

| Transformation | AUROC | 95% Confidence Interval | Accuracy | Degradation (\\Delta AUC) |
|:---|:---:|:---:|:---:|:---:|
"""
    for _, row in df.iterrows():
        report_md += f"| **{row['transformation']}** | `{row['auc']:.4f}` | {row['ci_formatted']} | `{row['acc']*100:.2f}%` | `{row['delta_auc']:+.4f}` |\n"

    report_md += """
---
*Generated automatically by Systematic Robustness Evaluation Suite.*
"""
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"--> Saved detailed robustness report to: {out_report_path}")


if __name__ == "__main__":
    main()
