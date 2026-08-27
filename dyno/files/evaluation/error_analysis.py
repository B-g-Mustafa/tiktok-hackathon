"""
Systematic Error Analysis and Failure Mode Forensics.
Identifies:
1. Confusion Matrix & False Positive / False Negative Rates
2. High-confidence False Positives (Real images predicted as AI)
3. High-confidence False Negatives (AI images predicted as Real)
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from PIL import Image

from datasets.base import load_image_rgb
from datasets.sid import scan_directory_for_samples, DirectParquetDataset, decode_image_bytes
from transforms import get_clean_transform
from models import build_detector


def perform_error_analysis(
    model: nn.Module,
    images: List[np.ndarray],
    labels: List[int],
    identifiers: List[str],
    device: torch.device,
    output_report_path: str = "./outputs/error_analysis_report.md",
    image_size: int = 384,
    top_k: int = 15,
) -> Dict[str, Any]:
    """
    Perform deep failure mode and error analysis on evaluation samples.
    """
    model.eval()
    clean_tf = get_clean_transform(image_size=image_size)

    records = []
    print(f"Running Error Analysis on {len(images)} samples...")

    for img, label, ident in zip(images, labels, identifiers):
        try:
            tensor = clean_tf(image=img)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                prob_clean = torch.sigmoid(model(tensor)).item()

            records.append({
                "identifier": ident,
                "label": int(label),
                "prob_clean": prob_clean,
                "pred_clean": 1 if prob_clean >= 0.5 else 0,
            })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if df.empty:
        return {}

    # False Positives: Label == 0 (Real), Pred == 1 (Fake)
    fp_df = df[(df["label"] == 0) & (df["pred_clean"] == 1)].sort_values(by="prob_clean", ascending=False)
    # False Negatives: Label == 1 (Fake), Pred == 0 (Real)
    fn_df = df[(df["label"] == 1) & (df["pred_clean"] == 0)].sort_values(by="prob_clean", ascending=True)

    cm = confusion_matrix(df["label"], df["pred_clean"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print("\n" + "=" * 50)
    print("           CONFUSION MATRIX SUMMARY")
    print("=" * 50)
    print(f"True Real (TN):  {tn:<6} | False AI (FP):  {fp:<6}")
    print(f"False Real (FN): {fn:<6} | True AI (TP):   {tp:<6}")
    print("-" * 50)
    print(f"Total False Positives (FPR): {fp} ({fp / max(1, tn + fp) * 100:.2f}%)")
    print(f"Total False Negatives (FNR): {fn} ({fn / max(1, tp + fn) * 100:.2f}%)")
    print("=" * 50 + "\n")

    # Write detailed markdown report
    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w") as f:
        f.write("# Robust AIGC Detector Error Analysis Report\n\n")
        f.write("## 1. Overall Confusion Matrix\n")
        f.write(f"- **Total Samples Evaluated**: {len(df)}\n")
        f.write(f"- **True Negatives (Correct Real)**: {tn}\n")
        f.write(f"- **True Positives (Correct AI)**: {tp}\n")
        f.write(f"- **False Positives (Real misclassified as AI)**: {fp} ({fp / max(1, tn + fp) * 100:.2f}% FPR)\n")
        f.write(f"- **False Negatives (AI misclassified as Real)**: {fn} ({fn / max(1, tp + fn) * 100:.2f}% FNR)\n\n")

        f.write("## 2. Top Most Confident False Positives (Real -> Mistaken for AI)\n")
        f.write("| Sample Identifier | True Label | Model P(AI) |\n|---|---|---|\n")
        for _, row in fp_df.head(top_k).iterrows():
            f.write(f"| `{row['identifier']}` | Real (0) | **{row['prob_clean']:.4f}** |\n")

        f.write("\n## 3. Top Most Confident False Negatives (AI -> Mistaken for Real)\n")
        f.write("| Sample Identifier | True Label | Model P(AI) |\n|---|---|---|\n")
        for _, row in fn_df.head(top_k).iterrows():
            f.write(f"| `{row['identifier']}` | AI (1) | **{row['prob_clean']:.4f}** |\n")

    print(f"Error analysis report saved to {out_p}")

    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "top_false_positives": fp_df.head(top_k).to_dict(orient="records"),
        "top_false_negatives": fn_df.head(top_k).to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser(description="AIGC Detector Error Analysis")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/sid_set/data")
    parser.add_argument("--output_report", type=str, default="./outputs/exp_c_consistency/error_analysis_report.md")
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = build_detector(config)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        clean_state = {k.replace("module.", "").replace("detector.", ""): v for k, v in state_dict.items()}
        detector.load_state_dict(clean_state, strict=False)
    detector.to(device)

    data_dir = Path(args.data_dir)
    images_raw: List[np.ndarray] = []
    labels: List[int] = []
    identifiers: List[str] = []

    parquet_files = list(data_dir.glob("*.parquet")) + list(data_dir.rglob("*.parquet"))
    if parquet_files:
        val_parquets = [p for p in parquet_files if any(k in p.name.lower() for k in ["validation", "val", "test", "eval"])]
        if not val_parquets:
            val_parquets = parquet_files[:5]
        
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
                identifiers.append(f"parquet_val_sample_{idx}")
            except Exception:
                continue
    else:
        samples = scan_directory_for_samples(data_dir)[:args.max_samples]
        for p, y in samples:
            try:
                img = load_image_rgb(p)
                images_raw.append(img)
                labels.append(y)
                identifiers.append(str(p))
            except Exception:
                continue

    image_size = config.get("data", {}).get("image_size", 384)
    perform_error_analysis(
        model=detector,
        images=images_raw,
        labels=labels,
        identifiers=identifiers,
        device=device,
        output_report_path=args.output_report,
        image_size=image_size,
    )


if __name__ == "__main__":
    main()
