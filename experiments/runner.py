"""
Runner script to execute all BERT/RoBERTa checkpoint experiments.

For checkpoints trained on 'original' data: uses train_data.json
For checkpoints trained on other sources (gemini, opus, qwen): uses polished_syllogisms_variables_opus.json
"""
import subprocess
import sys
from pathlib import Path

# All checkpoints to run
CHECKPOINTS = [
    "experiments/data_source_search/bert_base_uncased_original/checkpoint-235",
    "experiments/data_source_search/bert_base_uncased_gemini/checkpoint-235",
    "experiments/data_source_search/bert_base_uncased_opus/checkpoint-329",
    "experiments/data_source_search/bert_base_uncased_qwen/checkpoint-423",
    "experiments/data_source_search/bert_large_uncased_original/checkpoint-282",
    "experiments/data_source_search/bert_large_uncased_gemini/checkpoint-470",
    "experiments/data_source_search/bert_large_uncased_opus/checkpoint-470",
    "experiments/data_source_search/bert_large_uncased_qwen/checkpoint-470",
    "experiments/data_source_search/roberta_base_original/checkpoint-188",
    "experiments/data_source_search/roberta_base_gemini/checkpoint-376",
    "experiments/data_source_search/roberta_base_opus/checkpoint-423",
    "experiments/data_source_search/roberta_base_qwen/checkpoint-94",
]

# Input files based on data source
INPUT_ORIGINAL = "train_data/subtask 1/train_data.json"
INPUT_POLISHED = "train_data/subtask 1/polished_syllogisms_variables_opus.json"
# Reference file (ground truth) - always use train_data.json since it has validity labels
REFERENCE_FILE = "train_data/subtask 1/train_data.json"


def get_data_source(checkpoint_path: str) -> str:
    """Extract data source from checkpoint path."""
    parent_name = Path(checkpoint_path).parent.name
    for source in ["original", "gemini", "opus", "qwen"]:
        if parent_name.endswith(f"_{source}"):
            return source
    return "unknown"


def get_model_name(checkpoint_path: str) -> str:
    """Extract model name from checkpoint path."""
    return Path(checkpoint_path).parent.name


def main():
    results = []

    for checkpoint in CHECKPOINTS:
        model_name = get_model_name(checkpoint)
        data_source = get_data_source(checkpoint)

        # Select input file based on data source
        if data_source == "original":
            input_file = INPUT_ORIGINAL
        else:
            input_file = INPUT_POLISHED

        output_file = f"predictions/bert_{model_name}.json"

        print()
        print("=" * 60)
        print(f"Running: {model_name}")
        print(f"Data source: {data_source}")
        print(f"Input: {input_file}")
        print("=" * 60)

        cmd = [
            sys.executable,
            "experiments/run_bert_experiment.py",
            "--checkpoint", checkpoint,
            "--input", input_file,
            "--output", output_file,
            "--evaluate",
            "--reference", REFERENCE_FILE,
        ]

        try:
            result = subprocess.run(cmd, check=True)
            results.append((model_name, "SUCCESS"))
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {model_name} failed with exit code {e.returncode}")
            results.append((model_name, "FAILED"))
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, status in results:
        print(f"  {model_name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
