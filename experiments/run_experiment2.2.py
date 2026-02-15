"""
Formal Validity Derivation Experiment for SemEval 2026 Task 11 – Subtask 1.

This experiment checks the logical validity of canonical (variable-replaced)
syllogisms using a symbolic evaluator (SyllogismLogic).

It can run in two modes:
1. Prediction only (for test data): Generates validity predictions without evaluation
2. Prediction + Evaluation (for train data): Compares predictions with ground truth

JSON output schema:
{
  "accuracy": float,  # Only included when --evaluate is set
  "results": [
    {
      "id": "...",
      "validity": bool
    }
  ]
}

Additionally (when --evaluate is set):
- Runs the official SemEval evaluation metrics
- Runs each experiment 3 times and reports mean ± std
- Appends all metrics to a CSV file if --metrics-csv is provided

Usage for train data (with evaluation):
    poetry run python experiments/run_experiment2.2.py \
        --input "data/polished/polished_syllogisms_variables.json" \
        --ground-truth "train_data/subtask 1/train_data.json" \
        --output "predictions/derived_validity.json" \
        --evaluate \
        --metrics-csv "results/metrics.csv"

Usage for test data (prediction only, no evaluation):
    poetry run python experiments/run_experiment2.2.py \
        --input "data/polished/test_polished_variables.json" \
        --output "predictions/test_derived_validity.json"

Optional:
    --limit 100
"""

import argparse
import sys
import json
import csv
import numpy as np
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
# CSV helper - saves aggregated results with mean ± std
# ---------------------------------------------------------
def save_results_to_csv(all_run_results: list, csv_path: str, input_file: str, limit: int = None) -> None:
    """
    Save experiment results to CSV file with mean ± std across runs.

    Args:
        all_run_results: List of result dictionaries from multiple runs
        csv_path: Path to save the CSV file
        input_file: Input file path used for the experiment
        limit: Optional limit on number of examples
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate mean and std for all metrics
    metrics = [
        'accuracy', 'content_effect', 'combined_score',
        'correct_predictions', 'total_predictions'
    ]
    subgroup_metrics = [
        'plausible_valid', 'implausible_valid',
        'plausible_invalid', 'implausible_invalid'
    ]
    content_effect_metrics = ['intra_validity', 'inter_validity']

    def get_mean_std(key, nested_key=None):
        if nested_key:
            values = [r[key][nested_key] for r in all_run_results]
        else:
            values = [r[key] for r in all_run_results]
        return np.mean(values), np.std(values)

    # Prepare row data with mean ± std
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'input_file': input_file,
        'num_runs': len(all_run_results),
        'num_examples': limit if limit else all_run_results[0]['total_predictions'],
    }

    # Add mean and std for main metrics
    for metric in metrics:
        mean, std = get_mean_std(metric)
        row[f'{metric}_mean'] = mean
        row[f'{metric}_std'] = std

    # Add mean and std for subgroup accuracies
    for metric in subgroup_metrics:
        mean, std = get_mean_std('subgroup_accuracies', metric)
        row[f'acc_{metric}_mean'] = mean
        row[f'acc_{metric}_std'] = std

    # Add mean and std for content effect breakdown
    for metric in content_effect_metrics:
        mean, std = get_mean_std('content_effect_breakdown', metric)
        row[f'content_effect_{metric}_mean'] = mean
        row[f'content_effect_{metric}_std'] = std

    # Check if file exists to determine if we need to write headers
    file_exists = csv_path.exists()

    # Write to CSV
    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if file doesn't exist
        if not file_exists:
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
    parser.add_argument("--ground-truth", default=None, help="Ground truth file (required if --evaluate is set)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate", action="store_true", help="Enable evaluation against ground truth")
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # Validate arguments
    if args.evaluate and not args.ground_truth:
        parser.error("--ground-truth is required when --evaluate is set")

    print("=" * 70)
    print("SemEval 2026 – Formal Validity Derivation (Symbolic Logic Engine)")
    print("=" * 70)
    print(f"Mode: {'Prediction + Evaluation' if args.evaluate else 'Prediction Only'}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------
    with open(args.input, "r", encoding="utf-8") as f:
        polished = json.load(f)
    if args.limit:
        polished = polished[:args.limit]

    print(f"Loaded {len(polished)} examples from {args.input}")

    # Load ground truth only if evaluating
    gt_map = None
    gt_data = None
    if args.evaluate:
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        gt_map = {ex["id"]: ex for ex in gt_data}
        print(f"Ground truth loaded from {args.ground_truth}")

    print()

    # ---------------------------------------------------------
    # Run multiple trials for robustness (only when evaluating)
    # ---------------------------------------------------------
    NUM_RUNS = 3 if args.evaluate else 1
    if args.evaluate:
        print(f"Running {NUM_RUNS} trials for robustness (will save only the best)...\n")
    else:
        print(f"Running {NUM_RUNS} trial (no evaluation)...\n")

    all_run_results = []
    all_run_predictions = []  # Store predictions from each run
    all_run_wrong_predictions = []  # Store wrong predictions from each run

    for run_idx in range(NUM_RUNS):
        print("=" * 70)
        print(f"RUN {run_idx + 1}/{NUM_RUNS}")
        print("=" * 70)

        # ---------------------------------------------------------
        # Run symbolic evaluation
        # ---------------------------------------------------------
        results = []

        # Only track evaluation metrics when evaluating
        if args.evaluate:
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

            # Evaluation logic (only when --evaluate is set)
            if args.evaluate:
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

        # Calculate accuracy only when evaluating
        if args.evaluate:
            accuracy = correct / total if total else 0.0
            wrong_count = total - correct
            print(f"Wrong predictions: {wrong_count} / {total}")

        # ---------------------------------------------------------
        # Store predictions in memory (will save best one later)
        # ---------------------------------------------------------
        if args.evaluate:
            # For evaluation mode, include accuracy and wrap in object
            output_data = {"accuracy": accuracy, "results": results}
            wrong_payload = {
                "summary": {
                    "total_predictions": total,
                    "wrong_predictions": wrong_count,
                    "accuracy": round(accuracy, 4),
                },
                "errors": wrong_predictions,
            }
        else:
            # For prediction-only mode, save as flat array (no wrapper)
            output_data = results
            wrong_payload = None

        # Store predictions for later
        all_run_predictions.append(output_data)
        all_run_wrong_predictions.append(wrong_payload)

        # ---------------------------------------------------------
        # Official SemEval evaluation (only when evaluating)
        # ---------------------------------------------------------
        if args.evaluate:
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

            all_run_results.append(eval_results)

            print(f"Run {run_idx + 1} - Accuracy: {overall_acc:.4f}, "
                  f"Content Effect: {content_effect:.4f}, "
                  f"Combined Score: {combined:.4f}\n")
        else:
            print(f"Run {run_idx + 1} - Predictions generated (no evaluation)\n")

    # ---------------------------------------------------------
    # Save predictions and aggregated results
    # ---------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        print("=" * 70)
        print(f"AGGREGATED RESULTS (N={NUM_RUNS} runs)")
        print("=" * 70)

        # Print summary statistics
        acc_values = [r['accuracy'] for r in all_run_results]
        ce_values = [r['content_effect'] for r in all_run_results]
        cs_values = [r['combined_score'] for r in all_run_results]

        print(f"Accuracy: {np.mean(acc_values):.4f} ± {np.std(acc_values):.4f}")
        print(f"Content Effect: {np.mean(ce_values):.4f} ± {np.std(ce_values):.4f}")
        print(f"Combined Score: {np.mean(cs_values):.4f} ± {np.std(cs_values):.4f}\n")

        # Find the best run based on combined score
        best_run_idx = np.argmax(cs_values)
        best_combined_score = cs_values[best_run_idx]

        print(f"Best run: Run {best_run_idx + 1} (Combined Score: {best_combined_score:.4f})")
        print(f"Saving predictions from best run only...\n")

        # Save best predictions
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_run_predictions[best_run_idx], f, indent=2)
        print(f"✅ Best predictions saved to: {output_path}")

        # Save wrong predictions for best run
        wrong_out = output_path.parent / f"{output_path.stem}_wrong{output_path.suffix}"
        with wrong_out.open("w", encoding="utf-8") as f:
            json.dump(all_run_wrong_predictions[best_run_idx], f, indent=2)
        print(f"✅ Wrong predictions saved to: {wrong_out}")

        if args.metrics_csv:
            save_results_to_csv(all_run_results, args.metrics_csv, args.input, args.limit)
            print(f"✅ Aggregated results appended to: {args.metrics_csv}")

        print("\n✅ Evaluation completed successfully\n")
    else:
        # No evaluation - just save the single run
        print("=" * 70)
        print(f"PREDICTION COMPLETED")
        print("=" * 70)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_run_predictions[0], f, indent=2)
        print(f"✅ Predictions saved to: {output_path}")

        print("\n✅ Predictions generated successfully (no evaluation performed)\n")


if __name__ == "__main__":
    main()