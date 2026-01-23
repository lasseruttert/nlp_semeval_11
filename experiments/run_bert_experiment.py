"""
BERT/RoBERTa experiment runner for SemEval 2026 Task 11 - Subtask 1.

Usage:
    python experiments/run_bert_experiment.py \
        --checkpoint "experiments/data_source_search/bert_base_uncased_original/checkpoint-235" \
        --input "train_data/subtask 1/train_data.json" \
        --output "predictions/bert_base_original.json" \
        --evaluate \
        --reference "train_data/subtask 1/train_data.json"

    Results are automatically saved to CSV with filename:
        experiments/bert_results_{model_name}_{data_source}.csv

    Or specify custom path:
        --results-csv "experiments/my_results.csv"
"""
import argparse
import sys
import csv
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_test_data, save_predictions
from src.evaluator import evaluate_predictions


def extract_model_info(checkpoint_path: str) -> Dict[str, str]:
    """
    Extract model name, data source, and checkpoint step from checkpoint path.

    Expected path format:
        .../data_source_search/{model_name}_{data_source}/checkpoint-{step}
        e.g., experiments/data_source_search/bert_base_uncased_original/checkpoint-235

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dictionary with:
        - model_name: e.g., "bert_base_uncased"
        - data_source: e.g., "original", "gemini", "opus", "qwen"
        - checkpoint_step: e.g., "235"
        - model_type: e.g., "bert" or "roberta"
        - base_model: HuggingFace model identifier for tokenizer
    """
    path = Path(checkpoint_path)

    # Extract checkpoint step from folder name (e.g., "checkpoint-235")
    checkpoint_step = "unknown"
    if path.name.startswith("checkpoint-"):
        checkpoint_step = path.name.replace("checkpoint-", "")

    # Parent directory contains model_name_data_source
    parent_name = path.parent.name

    # Known data sources
    data_sources = ["original", "gemini", "opus", "qwen"]

    # Try to extract data source and model name
    model_name = parent_name
    data_source = "unknown"

    for ds in data_sources:
        if parent_name.endswith(f"_{ds}"):
            data_source = ds
            model_name = parent_name[: -(len(ds) + 1)]  # Remove _datasource suffix
            break

    # Determine model type and base model for tokenizer
    model_type = "unknown"
    base_model = "bert-base-uncased"  # default fallback

    if "roberta" in model_name.lower():
        model_type = "roberta"
        if "large" in model_name.lower():
            base_model = "roberta-large"
        else:
            base_model = "roberta-base"
    elif "bert" in model_name.lower():
        model_type = "bert"
        if "large" in model_name.lower():
            if "uncased" in model_name.lower():
                base_model = "bert-large-uncased"
            else:
                base_model = "bert-large-cased"
        else:
            if "uncased" in model_name.lower():
                base_model = "bert-base-uncased"
            else:
                base_model = "bert-base-cased"

    return {
        "model_name": model_name,
        "data_source": data_source,
        "checkpoint_step": checkpoint_step,
        "model_type": model_type,
        "base_model": base_model,
    }


def load_model_and_tokenizer(
    checkpoint_path: str, device: str
) -> Tuple[Any, Any, str]:
    """
    Load fine-tuned model and tokenizer.

    Args:
        checkpoint_path: Path to checkpoint directory
        device: Device to load model on ("cuda", "cpu", or "auto")

    Returns:
        Tuple of (model, tokenizer, resolved_device)
    """
    # Resolve device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {resolved_device}")

    # Extract model info for tokenizer
    model_info = extract_model_info(checkpoint_path)

    # Load config to get model type
    config = AutoConfig.from_pretrained(checkpoint_path)

    # Load tokenizer from base model (checkpoints don't include tokenizer)
    print(f"Loading tokenizer from: {model_info['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(model_info["base_model"])

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        config=config,
    )
    model.to(resolved_device)
    model.eval()

    print(f"Model loaded: {config.architectures[0] if config.architectures else 'unknown'}")
    print(f"Number of labels: {config.num_labels if hasattr(config, 'num_labels') else 2}")

    return model, tokenizer, resolved_device


def run_batch_inference(
    model: Any,
    tokenizer: Any,
    examples: List[Any],
    device: str,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Run batch inference on examples.

    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        examples: List of TestExample objects
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        List of prediction dictionaries with 'id' and 'validity' keys
    """
    predictions = []

    # Process in batches
    for i in tqdm(range(0, len(examples), batch_size), desc="Running inference"):
        batch = examples[i : i + batch_size]

        # Prepare texts
        texts = [ex.syllogism for ex in batch]

        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Get predictions (assuming label 1 = valid, label 0 = invalid)
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        # Create prediction dictionaries
        for ex, pred in zip(batch, preds):
            predictions.append({
                "id": ex.id,
                "validity": bool(pred),
            })

    return predictions


def save_results_to_csv(
    args: argparse.Namespace,
    results: Dict[str, Any],
    model_info: Dict[str, str],
    csv_path: str,
):
    """
    Save experiment results to CSV file with BERT-specific schema.

    Args:
        args: Parsed command-line arguments
        results: Dictionary of evaluation results
        model_info: Dictionary with model information
        csv_path: Path to save the CSV file
    """
    # Prepare row data with BERT-specific columns
    row = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_info["model_name"],
        "model_type": model_info["model_type"],
        "data_source": model_info["data_source"],
        "checkpoint_step": model_info["checkpoint_step"],
        "batch_size": args.batch_size,
        "input_file": args.input,
        "output_file": args.output,
        "reference_file": args.reference,
        "num_examples": args.limit if args.limit else results["total_predictions"],
        # Main metrics
        "accuracy": results["accuracy"],
        "content_effect": results["content_effect"],
        "combined_score": results["combined_score"],
        "correct_predictions": results["correct_predictions"],
        "total_predictions": results["total_predictions"],
        # Subgroup accuracies
        "acc_plausible_valid": results["subgroup_accuracies"]["plausible_valid"],
        "acc_implausible_valid": results["subgroup_accuracies"]["implausible_valid"],
        "acc_plausible_invalid": results["subgroup_accuracies"]["plausible_invalid"],
        "acc_implausible_invalid": results["subgroup_accuracies"]["implausible_invalid"],
        # Content effect breakdown
        "content_effect_intra_validity": results["content_effect_breakdown"]["intra_validity"],
        "content_effect_inter_validity": results["content_effect_breakdown"]["inter_validity"],
    }

    # Check if file exists to determine if we need to write headers
    file_exists = Path(csv_path).exists()

    # Write to CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on fine-tuned BERT/RoBERTa checkpoints"
    )

    # Checkpoint path (required)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )

    # Input/Output paths
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (test data)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save predictions JSON file",
    )

    # Evaluation options
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions against reference data",
    )

    parser.add_argument(
        "--reference",
        type=str,
        help="Path to reference/ground truth JSON file (required if --evaluate is set)",
    )

    # Optional parameters
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to run inference on (default: auto)",
    )

    # Results CSV (optional)
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Path to save experiment results CSV (default: auto-generated)",
    )

    args = parser.parse_args()

    # Validate evaluation arguments
    if args.evaluate and not args.reference:
        parser.error("--reference is required when --evaluate is set")

    # Extract model info from checkpoint path
    model_info = extract_model_info(args.checkpoint)

    print("=" * 60)
    print("SemEval 2026 Task 11 - BERT/RoBERTa Inference")
    print("=" * 60)
    print(f"Model: {model_info['model_name']}")
    print(f"Model Type: {model_info['model_type']}")
    print(f"Data Source: {model_info['data_source']}")
    print(f"Checkpoint Step: {model_info['checkpoint_step']}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch Size: {args.batch_size}")
    if args.limit:
        print(f"Limit: {args.limit} examples")
    print("=" * 60 + "\n")

    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.device)
        print("Model and tokenizer loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load test data
    print(f"Loading test data from: {args.input}...")
    try:
        test_examples = load_test_data(args.input)
        if args.limit:
            test_examples = test_examples[: args.limit]
        print(f"Loaded {len(test_examples)} test examples.\n")
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading test data: {e}")
        sys.exit(1)

    # Run inference
    print("Running inference...")
    try:
        predictions = run_batch_inference(
            model=model,
            tokenizer=tokenizer,
            examples=test_examples,
            device=device,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"\nError during inference: {e}")
        sys.exit(1)

    # Save predictions
    print(f"\nSaving predictions to: {args.output}...")
    try:
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_predictions(predictions, args.output)
        print("Predictions saved successfully.\n")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)

    # Evaluate if requested
    if args.evaluate:
        print("Evaluating predictions...")
        try:
            results = evaluate_predictions(
                ground_truth_path=args.reference,
                predictions_path=args.output,
                output_path=None,  # Don't save evaluation results to separate file
                verbose=True,
            )

            # Save results to CSV
            if args.results_csv:
                csv_path = args.results_csv
            else:
                # Auto-generate CSV filename from model and data source
                csv_path = f"experiments/bert_results_{model_info['model_name']}_{model_info['data_source']}.csv"

            # Create experiments directory if it doesn't exist
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

            save_results_to_csv(args, results, model_info, csv_path)
            print(f"Results appended to: {csv_path}")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            sys.exit(1)

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
