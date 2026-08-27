#!/usr/bin/env python3
"""Train the head on cached features and produce the robustness matrix.

Runs in seconds once features are cached, which is the point of caching: the
whole ablation sweep is affordable because this step is cheap.

Report ordering is deliberate. Worst-case AUROC is printed first and clean
AUROC last, because the failure this project exists to avoid is exactly the one
that looks like success on the clean column. In the NTIRE 2026 robust-detection
challenge, one team scored 0.9954 clean -- statistically tied with the winner --
and finished 9th on a robust AUROC of 0.8302.

Usage:
    python scripts/train_and_evaluate.py \
        --train-cache artifacts/features/train__....npz \
        --eval-cache  artifacts/features/cross_generator__....npz \
        --control-cache artifacts/features/content_matched_control__....npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.evaluation.metrics import RobustnessMatrix, compute_metrics  # noqa: E402
from src.models.budget import ParameterBudget  # noqa: E402
from src.models.encoders import ENCODER_CATALOG  # noqa: E402
from src.training.head import LinearHead, load_cache  # noqa: E402
from src.transforms.robustness import eval_grid  # noqa: E402

# Order for the matrix rows: clean first as the baseline, then families.
MATRIX_ORDER = [t.name for t in eval_grid()]


def evaluate_matrix(head: LinearHead, cache, name: str) -> RobustnessMatrix:
    """Score every transform view in a cache."""
    matrix = RobustnessMatrix(name)

    for view in cache.unique_views():
        subset = cache.view(view)
        if len(subset) == 0:
            continue
        if len(np.unique(subset.labels)) < 2:
            continue
        scores = head.predict_proba(subset.features)
        matrix.add(view, compute_metrics(subset.labels, scores))

    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, action="append", default=[],
                        help="Evaluation cache (repeatable).")
    parser.add_argument("--control-cache", type=Path, default=None,
                        help="Content-matched control (Control D).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports"))
    parser.add_argument(
        "--save-checkpoint", type=Path, default=None,
        help="Directory to save head.npz + meta.json, loadable by "
             "predict.py --model siglip2 --checkpoint <dir>.",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_cache = load_cache(args.train_cache)
    config_hash = train_cache.meta.get("config_hash")

    print("=" * 78)
    print("TRAIN + EVALUATE")
    print("=" * 78)
    print(f"train cache : {args.train_cache.name}")
    print(f"  rows      : {len(train_cache):,}  dim={train_cache.dim}")
    print(f"  encoder   : {train_cache.meta.get('encoder')} "
          f"(hash {config_hash})")
    print(f"  views     : {train_cache.unique_views()}")

    head = LinearHead(C=args.C, seed=args.seed).fit(
        train_cache.features, train_cache.labels
    )
    print(f"  head params: {head.n_parameters:,}")

    # -- parameter budget ---------------------------------------------------
    encoder_key = train_cache.meta.get("encoder")
    budget = ParameterBudget()
    if encoder_key in ENCODER_CATALOG:
        budget.add(
            f"{encoder_key} (frozen)", ENCODER_CATALOG[encoder_key]["params"]
        )
    budget.add("linear head", head.n_parameters, trainable=True)
    budget.check()
    print(f"  budget     : {budget.total:,} / 2,000,000,000 "
          f"({budget.utilization:.1%})")

    if args.save_checkpoint is not None:
        args.save_checkpoint.mkdir(parents=True, exist_ok=True)
        head.save(args.save_checkpoint / "head.npz")
        (args.save_checkpoint / "meta.json").write_text(json.dumps(
            {
                "encoder": encoder_key,
                "n_layers": len(train_cache.meta.get("layers", [])) or 3,
                "config_hash": config_hash,
                "total_parameters": budget.total,
            },
            indent=2,
        ))
        print(f"  checkpoint : {args.save_checkpoint} "
              f"(load with predict.py --model siglip2 --checkpoint {args.save_checkpoint})")

    # -- evaluation ---------------------------------------------------------
    results: dict[str, dict] = {}
    matrices: list[RobustnessMatrix] = []

    for cache_path in args.eval_cache:
        cache = load_cache(cache_path, expect_hash=config_hash)
        matrix = evaluate_matrix(head, cache, cache_path.stem.split("__")[0])
        matrices.append(matrix)

        print("\n" + "=" * 78)
        print(f"ROBUSTNESS MATRIX -- {matrix.model_name}")
        print("=" * 78)
        print(matrix.to_markdown(order=MATRIX_ORDER))
        results[matrix.model_name] = matrix.summary()

    # -- Control D: the falsification test ----------------------------------
    if args.control_cache is not None:
        control = load_cache(args.control_cache, expect_hash=config_hash)
        control_matrix = evaluate_matrix(head, control, "content_matched_control")
        matrices.append(control_matrix)

        print("\n" + "=" * 78)
        print("CONTROL D -- content-matched (faces vs generated faces)")
        print("=" * 78)
        print(
            "If the detector is reading synthesis artifacts, it should still\n"
            "separate these. If it was reading subject matter, it collapses here."
        )
        print()
        print(control_matrix.to_markdown(order=MATRIX_ORDER))
        results["content_matched_control"] = control_matrix.summary()

        clean = control_matrix.clean_auroc
        verdict = (
            "PASSES -- signal survives content matching (forensic, not semantic)"
            if clean >= 0.70
            else "FAILS -- the detector was largely reading content, not artifacts"
        )
        print(f"\nControl D verdict: clean AUROC {clean:.4f} -> {verdict}")
        results["control_d_verdict"] = verdict

    # -- headline -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("HEADLINE (worst case first -- clean accuracy is a diagnostic, not a goal)")
    print("=" * 78)
    for matrix in matrices:
        summary = matrix.summary()
        print(
            f"  {matrix.model_name:<28} "
            f"worst={summary['worst_auroc']:.4f} "
            f"({summary['worst_transform']})  "
            f"mean={summary['mean_auroc']:.4f}  "
            f"clean={summary['clean_auroc']:.4f}  "
            f"drop={summary['relative_degradation']:.1%}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "robustness_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))

    markdown = ["# Robustness Report", ""]
    markdown.append(budget.to_markdown())
    for matrix in matrices:
        markdown += ["", f"## {matrix.model_name}", "", matrix.to_markdown(MATRIX_ORDER)]
    (args.output / "robustness_report.md").write_text("\n".join(markdown))

    print(f"\nreports -> {report_path} and robustness_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
