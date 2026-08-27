"""
Out-of-Distribution (OOD) and Cross-Generator Generalization Evaluator.
Evaluates model on:
1. WildFake Benchmark (COCO val2017 real vs DALL-E advanced synthetic)
2. GenImage Held-Out Generator evaluation
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from datasets.wildfake import get_wildfake_dataloader
from datasets.genimage import get_genimage_eval_loader
from models import build_detector
from evaluation.bootstrap_ci import compute_bootstrap_auc_ci


def evaluate_wildfake(
    model: nn.Module,
    wildfake_dir: str,
    device: torch.device,
    image_size: int = 384,
    batch_size: int = 64,
    clean_dev_auc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate detector on untouched WildFake organizer test set.
    """
    loader = get_wildfake_dataloader(
        root_dir=wildfake_dir,
        image_size=image_size,
        batch_size=batch_size,
    )

    if len(loader.dataset) == 0:
        print(f"Warning: No samples found in WildFake directory: {wildfake_dir}")
        return {"wildfake_auc": 0.5, "status": "no_data"}

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].cpu().numpy()
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    ap = float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    acc = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    ci_res = compute_bootstrap_auc_ci(y_true, y_pred)

    gen_gap = (clean_dev_auc - auc) if clean_dev_auc is not None else 0.0

    print("\n" + "=" * 50)
    print("WILDFEKE OUT-OF-DISTRIBUTION (OOD) RESULTS")
    print("=" * 50)
    print(f"WildFake AUROC:         {auc:.4f} ({ci_res['formatted']})")
    print(f"WildFake AP:            {ap:.4f}")
    print(f"WildFake Accuracy:      {acc:.4f}")
    if clean_dev_auc is not None:
        print(f"Generalization Gap:     {gen_gap:+.4f} (Dev AUC: {clean_dev_auc:.4f} -> WildFake: {auc:.4f})")
    print("=" * 50 + "\n")

    return {
        "wildfake_auc": auc,
        "wildfake_ap": ap,
        "wildfake_acc": acc,
        "generalization_gap": gen_gap,
        "ci": ci_res,
    }


from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, roc_curve


def evaluate_genimage_generalization(
    model: nn.Module,
    genimage_dir: str,
    held_out_generator: str,
    device: torch.device,
    image_size: int = 384,
    batch_size: int = 64,
    use_tta: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate detector on unseen held-out GenImage generator with optional TTA.
    """
    test_loader = get_genimage_eval_loader(
        root_dir=genimage_dir,
        generator_name=held_out_generator,
        image_size=image_size,
        batch_size=batch_size,
    )

    if len(test_loader.dataset) == 0:
        return {"held_out_generator": held_out_generator, "auc": 0.5, "status": "no_data"}

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].cpu().numpy()
            
            if use_tta:
                logits_1 = model(images)
                logits_2 = model(torch.flip(images, dims=[-1]))
                probs = ((torch.sigmoid(logits_1) + torch.sigmoid(logits_2)) / 2.0).cpu().numpy()
            else:
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    acc_default = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    
    # Calibrated Accuracy via optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    opt_idx = int(np.argmax(tpr - fpr))
    opt_thresh = float(thresholds[opt_idx]) if opt_idx < len(thresholds) else 0.5
    acc_calibrated = float(accuracy_score(y_true, (y_pred >= opt_thresh).astype(int)))
    
    ci_res = compute_bootstrap_auc_ci(y_true, y_pred)

    print(f"\nGenImage Held-Out [{held_out_generator}] {'(TTA ON)' if use_tta else ''}")
    print(f"--> AUROC: {auc:.4f} ({ci_res['formatted']})")
    print(f"--> Accuracy (Threshold 0.50): {acc_default*100:.2f}%")
    print(f"--> Calibrated Accuracy (Threshold {opt_thresh:.3f}): {acc_calibrated*100:.2f}%")

    return {
        "held_out_generator": held_out_generator,
        "auc": auc,
        "acc": acc_default,
        "calibrated_acc": acc_calibrated,
        "optimal_threshold": opt_thresh,
        "ci": ci_res,
    }


def evaluate_community_forensics(
    model: nn.Module,
    community_dir: str,
    device: torch.device,
    image_size: int = 384,
    batch_size: int = 64,
    use_tta: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate detector on OwensLab/CommunityForensics-Eval (CVPR 2025).
    """
    from datasets.genimage import scan_genimage_generator_dir
    from datasets.base import ImageDataset
    from transforms import get_clean_transform
    from torch.utils.data import DataLoader

    comm_path = Path(community_dir)
    samples = scan_genimage_generator_dir(comm_path)
    if len(samples) == 0:
        print(f"Warning: No samples found in {community_dir}")
        return {"benchmark": "CommunityForensics-Eval", "auc": 0.5, "status": "no_data"}

    tf = get_clean_transform(image_size=image_size)
    dataset = ImageDataset(samples=samples, transform=tf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].cpu().numpy()

            if use_tta:
                logits_1 = model(images)
                logits_2 = model(torch.flip(images, dims=[-1]))
                probs = ((torch.sigmoid(logits_1) + torch.sigmoid(logits_2)) / 2.0).cpu().numpy()
            else:
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    auc = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    ap = float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    acc_default = float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    opt_idx = int(np.argmax(tpr - fpr))
    opt_thresh = float(thresholds[opt_idx]) if opt_idx < len(thresholds) else 0.5
    acc_calibrated = float(accuracy_score(y_true, (y_pred >= opt_thresh).astype(int)))
    
    ci_res = compute_bootstrap_auc_ci(y_true, y_pred)

    print("\n" + "=" * 65)
    print("      CommunityForensics-Eval (CVPR 2025) Benchmark")
    print("=" * 65)
    print(f"--> AUROC: {auc:.4f} ({ci_res['formatted']})")
    print(f"--> Average Precision: {ap:.4f}")
    print(f"--> Accuracy (Threshold 0.50): {acc_default*100:.2f}%")
    print(f"--> Calibrated Accuracy (Threshold {opt_thresh:.3f}): {acc_calibrated*100:.2f}%")
    print("=" * 65 + "\n")

    return {
        "benchmark": "CommunityForensics-Eval",
        "auc": auc,
        "ap": ap,
        "acc": acc_default,
        "calibrated_acc": acc_calibrated,
        "optimal_threshold": opt_thresh,
        "ci": ci_res,
    }


def main():
    parser = argparse.ArgumentParser(description="AIGC Detector OOD & Generalization Suite")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--wildfake_dir", type=str, default=None, help="WildFake dataset directory")
    parser.add_argument("--genimage_dir", type=str, default=None, help="GenImage dataset directory")
    parser.add_argument("--community_dir", type=str, default="./data/community_forensics", help="CommunityForensics-Eval dataset directory")
    parser.add_argument("--output_report", type=str, default="outputs/ood_report.md", help="Output Markdown report")
    parser.add_argument("--benchmark", choices=["wildfake", "genimage", "community_forensics", "all"], default="all", help="Which OOD benchmark to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--use_tta", action="store_true", help="Enable Multi-View Test-Time Augmentation")
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
        print(f"Warning: Checkpoint '{args.checkpoint}' not found.")

    detector.to(device)
    detector.eval()

    image_size = config.get("data", {}).get("image_size", 384)
    wildfake_dir = args.wildfake_dir or config.get("data", {}).get("wildfake_data_dir", "./data/wildfake")
    genimage_dir = args.genimage_dir or config.get("data", {}).get("genimage_data_dir", "./data/genimage")
    community_dir = args.community_dir

    wf_results = {}
    gi_results = []
    cf_results = {}

    # 1. Evaluate WildFake if requested
    if args.benchmark in ["wildfake", "all"] and Path(wildfake_dir).exists():
        wf_results = evaluate_wildfake(
            model=detector,
            wildfake_dir=wildfake_dir,
            device=device,
            image_size=image_size,
            batch_size=args.batch_size,
        )

    # 2. Evaluate GenImage if requested
    if args.benchmark in ["genimage", "all"] and Path(genimage_dir).exists():
        gen_subdirs = [d.name for d in Path(genimage_dir).iterdir() if d.is_dir()]
        for gen_name in gen_subdirs:
            res = evaluate_genimage_generalization(
                model=detector,
                genimage_dir=genimage_dir,
                held_out_generator=gen_name,
                device=device,
                image_size=image_size,
                batch_size=args.batch_size,
                use_tta=args.use_tta,
            )
            gi_results.append(res)

    # 3. Evaluate CommunityForensics-Eval if requested
    if args.benchmark in ["community_forensics", "all"] and Path(community_dir).exists():
        cf_results = evaluate_community_forensics(
            model=detector,
            community_dir=community_dir,
            device=device,
            image_size=image_size,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
        )

    # Save Report
    out_report_path = Path(args.output_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    report_md = f"""# Out-of-Distribution (OOD) & Cross-Generator Generalization Report

**Model**: `{config.get('model', {}).get('backbone', 'DINOv2-Large')}`  
**Checkpoint**: `{args.checkpoint}`  

---

## 1. WildFake Benchmark (Real COCO vs DALL-E)
* **WildFake AUROC**: `{wf_results.get('wildfake_auc', 0.5):.4f}`
* **WildFake AP**: `{wf_results.get('wildfake_ap', 0.5):.4f}`
* **WildFake Accuracy**: `{wf_results.get('wildfake_acc', 0.5)*100:.2f}%`

---

## 2. CommunityForensics-Eval (CVPR 2025 Benchmark)
* **CommunityForensics AUROC**: `{cf_results.get('auc', 0.5):.4f}`
* **CommunityForensics AP**: `{cf_results.get('ap', 0.5):.4f}`
* **CommunityForensics Accuracy**: `{cf_results.get('acc', 0.5)*100:.2f}%`
* **Calibrated Accuracy**: `{cf_results.get('calibrated_acc', 0.5)*100:.2f}% (Threshold: {cf_results.get('optimal_threshold', 0.5):.3f})`

---

## 3. GenImage Benchmark (Unseen Held-Out Generators)
| Generator Subset | AUROC | Accuracy | Status |
|:---|:---:|:---:|:---:|
"""
    for r in gi_results:
        report_md += f"| **{r.get('held_out_generator', 'unknown')}** | `{r.get('auc', 0.5):.4f}` | `{r.get('acc', 0.0)*100:.2f}%` | {'Evaluated' if 'auc' in r else 'No Data'} |\n"

    report_md += """
---
*Generated automatically by OOD Generalization Suite.*
"""
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"--> Saved OOD report to: {out_report_path}")


if __name__ == "__main__":
    main()
