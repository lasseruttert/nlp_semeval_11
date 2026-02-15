"""
Formal Validity Derivation Experiment for SemEval 2026 Task 11 – Subtask 1.

This experiment checks the logical validity of canonical (variable-replaced)
syllogisms using a symbolic evaluator (SyllogismLogic).

It compares the derived validity (True/False) of each syllogism
with the human-annotated ground truth provided in train_data.json.

JSON output schema (UNCHANGED):
{
  "accuracy": float,
  "results": [
    {
      "id": "...",
      "pred_valid": bool,
      "true_valid": bool,
      "match": bool
    }
  ]
}

Additionally:
- Runs the official SemEval evaluation metrics
- Appends all metrics to a CSV file if --metrics-csv is provided

Usage:
    poetry run python experiments/run_experiment2.2.py --input "data/polished/polished_syllogisms_variables.json" --ground-truth "train_data/subtask 1/train_data.json" --output "predictions/derived_validity.json" --metrics-csv "results/metrics.csv"

Optional:
    --limit 100
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# Add src + evaluation kit to path
# ---------------------------------------------------------
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "evaluation_kit" / "task 1 & 3"))

from src.syllogism_logic import SyllogismLogic

from evaluation_script import (
    calculate_accuracy,
    calculate_subgroup_accuracy,
    calculate_content_effect_bias,
    calculate_smooth_combined_metric,
)

# ---------------------------------------------------------
# CSV helper
# ---------------------------------------------------------
def append_metrics_to_csv(results: dict, csv_path: str) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "accuracy": results["accuracy"],
        "content_effect": results["content_effect"],
        "combined_score": results["combined_score"],
        "correct_predictions": results["correct_predictions"],
        "total_predictions": results["total_predictions"],
        "acc_plausible_valid": results["subgroup_accuracies"]["plausible_valid"],
        "acc_implausible_valid": results["subgroup_accuracies"]["implausible_valid"],
        "acc_plausible_invalid": results["subgroup_accuracies"]["plausible_invalid"],
        "acc_implausible_invalid": results["subgroup_accuracies"]["implausible_invalid"],
        "content_effect_intra_validity": results["content_effect_breakdown"]["intra_validity"],
        "content_effect_inter_validity": results["content_effect_breakdown"]["inter_validity"],
    }

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Derive formal logical validity for canonical syllogisms"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    print("=" * 70)
    print("SemEval 2026 – Formal Validity Derivation (Symbolic Logic Engine)")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------
    with open(args.input, "r", encoding="utf-8") as f:
        polished = json.load(f)
    if args.limit:
        polished = polished[:args.limit]

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_map = {ex["id"]: ex for ex in gt_data}

    # ---------------------------------------------------------
    # Run symbolic evaluation
    # ---------------------------------------------------------
    results = []
    wrong_predictions = []

    correct = 0
    total = 0

    for ex in polished:
        sid = ex["id"]
        normalized_syll = ex["syllogism"].strip()

        parts = [p.strip() for p in normalized_syll.split(".") if p.strip()]
        if len(parts) != 3:
            continue

        major, minor, conclusion = [p + "." for p in parts]

        try:
            pred_valid = SyllogismLogic.is_valid(major, minor, conclusion)
        except Exception:
            pred_valid = None

        true_valid = gt_map[sid]["validity"]
        match = pred_valid == true_valid

        total += 1
        if match:
            correct += 1
        else:
            wrong_predictions.append(
                {
                    "id": sid,
                    "original_syllogism": gt_map[sid].get("syllogism"),
                    "normalized_syllogism": normalized_syll,
                    "true_valid": true_valid,
                    "pred_valid": pred_valid,
                }
            )

        results.append(
            {
                "id": sid,
                "validity": pred_valid,
            }
        )

    accuracy = correct / total if total else 0.0
    wrong_count = total - correct

    # ---------------------------------------------------------
    # Save prediction JSON (UNCHANGED)
    # ---------------------------------------------------------
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "results": results}, f, indent=2)

    # ---------------------------------------------------------
    # Save WRONG predictions JSON (WITH SUMMARY AT TOP)
    # ---------------------------------------------------------
    wrong_out = Path(args.output).with_name("wrong_predictions.json")

    wrong_payload = {
        "summary": {
            "total_predictions": total,
            "wrong_predictions": wrong_count,
            "accuracy": round(accuracy, 4),
        },
        "errors": wrong_predictions,
    }

    with wrong_out.open("w", encoding="utf-8") as f:
        json.dump(wrong_payload, f, indent=2)

    print(
        f"Wrong predictions written to {wrong_out} "
        f"({wrong_count} / {total} wrong)"
    )

    # ---------------------------------------------------------
    # Official SemEval evaluation
    # ---------------------------------------------------------
    overall_acc, _, _ = calculate_accuracy(
        ground_truth_list=gt_data,
        predictions_list=results,
        metric_name="validity",
        prediction_key="validity",
        plausibility_filter=None,
    )

    acc_pv, _, _ = calculate_subgroup_accuracy(gt_map, results, True, True)
    acc_iv, _, _ = calculate_subgroup_accuracy(gt_map, results, True, False)
    acc_pi, _, _ = calculate_subgroup_accuracy(gt_map, results, False, True)
    acc_ii, _, _ = calculate_subgroup_accuracy(gt_map, results, False, False)

    conditional = {
        "acc_plausible_valid": acc_pv,
        "acc_implausible_valid": acc_iv,
        "acc_plausible_invalid": acc_pi,
        "acc_implausible_invalid": acc_ii,
    }

    bias = calculate_content_effect_bias(conditional)
    content_effect = bias["tot_content_effect"]
    combined = calculate_smooth_combined_metric(overall_acc, content_effect)

    eval_results = {
        "accuracy": round(overall_acc, 4),
        "content_effect": round(content_effect, 4),
        "combined_score": round(combined, 4),
        "correct_predictions": correct,
        "total_predictions": total,
        "subgroup_accuracies": {
            "plausible_valid": round(acc_pv, 2),
            "implausible_valid": round(acc_iv, 2),
            "plausible_invalid": round(acc_pi, 2),
            "implausible_invalid": round(acc_ii, 2),
        },
        "content_effect_breakdown": {
            "intra_validity": round(bias["content_effect_intra_validity_label"], 4),
            "inter_validity": round(bias["content_effect_inter_validity_label"], 4),
        },
    }

    if args.metrics_csv:
        append_metrics_to_csv(eval_results, args.metrics_csv)

    print("\n✅ Evaluation completed successfully\n")


if __name__ == "__main__":
    main()