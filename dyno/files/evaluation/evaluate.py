"""
Master Evaluation CLI for Robust AIGC Detector.
Executes:
1. Clean AUROC & AP Evaluation
2. Full Robustness Sweep & Delta-AUC breakdown
3. Grommelt et al. Bias Audit
4. WildFake OOD Benchmark
5. GenImage Generator-Held-Out Evaluation
6. Error Analysis & Bootstrap CIs
"""

import os
import sys
import argparse
from pathlib import Path
import json
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from models import build_detector, DualStreamDetector
from training.utils import load_checkpoint
from datasets.sid import scan_directory_for_samples
from evaluation.robustness import evaluate_robustness_suite
from evaluation.bias_audit import run_bias_audit
from evaluation.ood import evaluate_wildfake, evaluate_genimage_generalization
from evaluation.error_analysis import perform_error_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Master Evaluation Suite for Robust AIGC Detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--config", type=str, default=None, help="Path to model config YAML")
    parser.add_argument("--val_dir", type=str, default="./data/sid_set/val", help="Validation dataset directory")
    parser.add_argument("--wildfake_dir", type=str, default="./data/wildfake", help="WildFake benchmark directory")
    parser.add_argument("--genimage_dir", type=str, default="./data/genimage", help="GenImage benchmark directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_results", help="Directory to save eval reports")
    parser.add_argument("--image_size", type=int, default=384, help="Input image dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--run_bias_audit", action="store_true", help="Run Grommelt et al. bias audit")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Config & Model
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"model": {"backbone": "facebook/dinov2-large", "freeze_backbone": True}}

    print("Building model...")
    model = build_detector(config)
    if config.get("model", {}).get("use_frequency_branch", False):
        model = DualStreamDetector(model)

    print(f"Loading weights from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    eval_summary = {}

    # 2. Bias Audit
    if args.run_bias_audit and Path(args.val_dir).exists():
        bias_res = run_bias_audit(
            data_dir=args.val_dir,
            output_report_path=str(out_dir / "bias_audit_report.md"),
        )
        eval_summary["bias_audit"] = bias_res

    # 3. Robustness Sweep on Validation Set
    if Path(args.val_dir).exists():
        samples = scan_directory_for_samples(args.val_dir)
        if samples:
            print(f"\n--- Running Robustness Suite on {len(samples)} Validation Samples ---")
            rob_res = evaluate_robustness_suite(
                model=model,
                samples=samples,
                device=device,
                image_size=args.image_size,
                batch_size=args.batch_size,
            )
            eval_summary["robustness"] = {
                "clean_auc": rob_res["clean_auc"],
                "mean_robust_auc": rob_res["mean_robust_auc"],
                "max_delta_auc": rob_res["max_delta_auc"],
                "details": rob_res["detailed_table"],
            }
            # Save table to CSV
            rob_res["dataframe"].to_csv(out_dir / "robustness_table.csv", index=False)

            # Error analysis
            err_res = perform_error_analysis(
                model=model,
                samples=samples,
                device=device,
                output_report_path=str(out_dir / "error_analysis_report.md"),
                image_size=args.image_size,
            )
            eval_summary["error_analysis"] = err_res

    # 4. WildFake OOD Evaluation
    if Path(args.wildfake_dir).exists():
        clean_auc = eval_summary.get("robustness", {}).get("clean_auc", None)
        wf_res = evaluate_wildfake(
            model=model,
            wildfake_dir=args.wildfake_dir,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            clean_dev_auc=clean_auc,
        )
        eval_summary["wildfake_ood"] = wf_res

    # 5. GenImage Generalization
    if Path(args.genimage_dir).exists():
        for held_out in ["wukong", "vqdm", "adm"]:
            if (Path(args.genimage_dir) / held_out).exists():
                gi_res = evaluate_genimage_generalization(
                    model=model,
                    genimage_dir=args.genimage_dir,
                    held_out_generator=held_out,
                    device=device,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                )
                eval_summary[f"genimage_{held_out}"] = gi_res

    # Save master summary JSON
    summary_file = out_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        # Convert non-serializable objects
        clean_summary = {}
        for k, v in eval_summary.items():
            try:
                json.dumps(v)
                clean_summary[k] = v
            except Exception:
                clean_summary[k] = str(v)
        json.dump(clean_summary, f, indent=2)

    print(f"\nAll evaluation reports successfully exported to {out_dir}")


if __name__ == "__main__":
    main()
