"""
Formal Validity Derivation Experiment for SemEval 2026 Task 11 – Subtask 1.

This experiment checks the logical validity of canonical (variable-replaced)
syllogisms using a symbolic evaluator (SyllogismLogic).

It compares the derived validity (True/False) of each syllogism
with the human-annotated ground truth provided in train_data.json.

The JSON schema of the output is:
{
  "accuracy": float,
  "results": [
    {
      "id": "...",
      "pred_valid": bool,
      "true_valid": bool,
      "match": bool
    },
    ...
  ]
}

Usage:
    poetry run python experiments/run_experiment2.2.py --input "data/polished/polished_syllogisms_variables.json" --ground-truth "train_data/subtask_1/train_data.json" --output "predictions/derived_validity.json"

Optional:
    --limit 100
"""

import argparse
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.syllogism_logic import SyllogismLogic


def main():
    parser = argparse.ArgumentParser(
        description="Derive formal logical validity for canonical syllogisms"
    )

    # Input arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to canonical syllogisms JSON (variable-replaced forms)"
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to training JSON with ground-truth validity labels"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save derived validity results"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of syllogisms to process"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SemEval 2026 – Formal Validity Derivation (Symbolic Logic Engine)")
    print("=" * 70)
    print(f"Input syllogisms : {args.input}")
    print(f"Ground truth file: {args.ground_truth}")
    print(f"Output results   : {args.output}")
    if args.limit:
        print(f"Limit            : {args.limit}")
    print("=" * 70 + "\n")

    # ---------------------------------------------------------
    # Load canonical syllogisms (variable-replaced)
    # ---------------------------------------------------------
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            polished = json.load(f)
        if args.limit:
            polished = polished[:args.limit]
        print(f"Loaded {len(polished)} canonical syllogisms.\n")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Load ground truth (train data)
    # ---------------------------------------------------------
    try:
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            ground_truth = {ex["id"]: ex["validity"] for ex in gt_data}
        print(f"Loaded ground-truth labels: {len(ground_truth)} entries.\n")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # Run formal logic evaluation
    # ---------------------------------------------------------
    print("Deriving formal validity for each syllogism...")
    results = []
    correct = 0
    total = 0

    for ex in polished:
        sid = ex["id"]
        text = ex["syllogism"].strip()
        if not text:
            continue

        parts = [p.strip() for p in text.split(".") if p.strip()]
        if len(parts) != 3:
            print(f"[Warning] Skipped malformed syllogism: {sid}")
            continue

        major, minor, conclusion = [p + "." for p in parts]

        try:
            pred_valid = SyllogismLogic.is_valid(major, minor, conclusion)
        except Exception as e:
            print(f"[Error] Failed to evaluate {sid}: {e}")
            pred_valid = None

        true_valid = ground_truth.get(sid)
        match = (pred_valid == true_valid)
        total += 1
        correct += 1 if match else 0

        results.append({
            "id": sid,
            "pred_valid": pred_valid,
            "true_valid": true_valid,
            "match": match
        })

    accuracy = correct / total if total else 0.0
    print(f"\n✅ Formal Validity Accuracy: {accuracy:.3f} ({correct}/{total})")

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, indent=2)
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)

    print("\nFormal validity derivation experiment completed successfully!\n")


if __name__ == "__main__":
    main()
