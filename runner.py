#!/usr/bin/env python3
# runner.py
import os
import sys
import shlex
import subprocess
from typing import List, Tuple, Union

# ------------------------------------------------------------
# Pflege deine Tasks hier (Strings oder Listen):
#   Beispiele:
# TASKS = [
#     "prep.py --data data.csv",
#     ["train.py", "--epochs", "10", "--lr", "1e-3"],
#     r"eval.py --ckpt C:\tmp\model.pt --verbose", 
# ]
# ------------------------------------------------------------
TASKS: List[Union[str, List[str]]] = [
    # ===== NORMALIZATION (Test Data) =====
    # Normalize test data with different models to prepare for formal validity checks

    # # Qwen - Test Data Normalization
    # [
    #     "experiments/run_experiment2.1.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--input", "test_data/subtask 1/test_data_subtask_1.json",
    #     "--output", "data/polished/test_polished_variables_qwen.json"
    # ],

    # # Gemini - Test Data Normalization
    # [
    #     "experiments/run_experiment2.1.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--input", "test_data/subtask 1/test_data_subtask_1.json",
    #     "--output", "data/polished/test_polished_variables_gemini.json"
    # ],

    # # Claude Opus 4.5 - Test Data Normalization
    # [
    #     "experiments/run_experiment2.1.py",
    #     "--model", "anthropic/claude-opus-4.5",
    #     "--input", "test_data/subtask 1/test_data_subtask_1.json",
    #     "--output", "data/polished/test_polished_variables_opus.json"
    # ],

    # Claude Opus 4.6 - Test Data Normalization
    [
        "experiments/run_experiment2.1.py",
        "--model", "anthropic/claude-opus-4.6",
        "--input", "test_data/subtask 1/test_data_subtask_1.json",
        "--output", "data/polished/test_polished_variables_opus46.json"
    ],

    # ===== FORMAL VALIDITY (Test Data - No Evaluation) =====
    # Run formal logic check on normalized test data (prediction only, no ground truth)

    # # Formal validity on Qwen-normalized test data
    # [
    #     "experiments/run_experiment2.2.py",
    #     "--input", "data/polished/test_polished_variables_qwen.json",
    #     "--output", "predictions/test_formal_validity_qwen.json"
    # ],

    # # Formal validity on Gemini-normalized test data
    # [
    #     "experiments/run_experiment2.2.py",
    #     "--input", "data/polished/test_polished_variables_gemini.json",
    #     "--output", "predictions/test_formal_validity_gemini.json"
    # ],

    # # Formal validity on Opus-normalized test data
    # [
    #     "experiments/run_experiment2.2.py",
    #     "--input", "data/polished/test_polished_variables_opus.json",
    #     "--output", "predictions/test_formal_validity_opus.json"
    # ],

    # ===== LLM EXPERIMENTS (Test Data - No Evaluation) =====
    # For each LLM: raw+direct, normalized+direct, normalized+formal
    # Output: predictions/<run_name>/predictions.json

    # # QWEN - Raw test data + direct prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "direct",
    #     "--input", "test_data/subtask 1/test_data_subtask_1.json",
    #     "--output", "predictions/qwen_raw_direct/predictions.json"
    # ],

    # # QWEN - Normalized test data + direct prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "direct",
    #     "--input", "data/polished/test_polished_variables_qwen.json",
    #     "--output", "predictions/qwen_normalized_direct/predictions.json"
    # ],

    # # QWEN - Normalized test data + formal prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "formal",
    #     "--input", "data/polished/test_polished_variables_qwen.json",
    #     "--output", "predictions/qwen_normalized_formal/predictions.json"
    # ],

    # # GEMINI - Raw test data + direct prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--prompt", "direct",
    #     "--input", "test_data/subtask 1/test_data_subtask_1.json",
    #     "--output", "predictions/gemini_raw_direct/predictions.json"
    # ],

    # # GEMINI - Normalized test data + direct prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--prompt", "direct",
    #     "--input", "data/polished/test_polished_variables_gemini.json",
    #     "--output", "predictions/gemini_normalized_direct/predictions.json"
    # ],

    # # GEMINI - Normalized test data + formal prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--prompt", "formal",
    #     "--input", "data/polished/test_polished_variables_gemini.json",
    #     "--output", "predictions/gemini_normalized_formal/predictions.json"
    # ],

    # # OPUS - Raw test data + direct prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "anthropic/claude-opus-4.5",
    #     "--prompt", "direct",
    #     "--input", "test_data/subtask 1/test_data_subtask_1.json",
    #     "--output", "predictions/opus_raw_direct/predictions.json"
    # ],

    # # OPUS - Normalized test data + direct prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "anthropic/claude-opus-4.5",
    #     "--prompt", "direct",
    #     "--input", "data/polished/test_polished_variables_opus.json",
    #     "--output", "predictions/opus_normalized_direct/predictions.json"
    # ],

    # # OPUS - Normalized test data + formal prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "anthropic/claude-opus-4.5",
    #     "--prompt", "formal",
    #     "--input", "data/polished/test_polished_variables_opus.json",
    #     "--output", "predictions/opus_normalized_formal/predictions.json"
    # ],

    # ===== OPUS 4.6 EXPERIMENTS =====

    # OPUS 4.6 - Raw test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.6",
        "--prompt", "direct",
        "--input", "test_data/subtask 1/test_data_subtask_1.json",
        "--output", "predictions/opus46_raw_direct/predictions.json"
    ],

    # OPUS 4.6 - Normalized test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.6",
        "--prompt", "direct",
        "--input", "data/polished/test_polished_variables_opus46.json",
        "--output", "predictions/opus46_normalized_direct/predictions.json"
    ],

    # OPUS 4.6 - Normalized test data + formal prompt
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.6",
        "--prompt", "formal",
        "--input", "data/polished/test_polished_variables_opus46.json",
        "--output", "predictions/opus46_normalized_formal/predictions.json"
    ],

    # # 1. Qwen Instruct - Direct Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "direct",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_direct.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 2. Qwen Instruct - Chain of Thought Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "cot",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_cot.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 3. Qwen Instruct - Counterexample Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "counterexample",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_counterexample.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 4. Qwen Instruct - Venn Diagram Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "venn",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_venn.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 5. Qwen Instruct - Syllogistic Rules Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "rules",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_rules.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 6. Qwen Instruct - Minimal Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "minimal",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_minimal.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 7. Qwen Instruct - Symbolic Logic Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "symbolic",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_symbolic.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 8. Qwen Instruct - Comprehensive Prompt (with examples, rules, strategies)
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "comprehensive",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_comprehensive.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # # 9. Qwen Instruct - Fallacy Focus Prompt
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "qwen/qwen3-vl-235b-a22b-instruct",
    #     "--prompt", "fallacy",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/qwen_instruct_fallacy.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json"
    # ],

    # Example: LLM prediction on train data (with evaluation)
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--prompt", "direct",
    #     "--input", "data/polished/polished_syllogisms_variables_opus.json",
    #     "--output", "predictions/google_gemini3_direct.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json",
    # ],
    
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--prompt", "reason",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/google_gemini3_reason.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json",
    # ],
    
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "anthropic/claude-opus-4.5",
    #     "--prompt", "direct",
    #     "--input", "train_data/subtask 1/polished_syllogisms_variables_opus.json",
    #     "--output", "predictions/anthropic_claude_opus_direct.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json",
    # ],
    
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "anthropic/claude-opus-4.5",
    #     "--prompt", "reason",
    #     "--input", "train_data/subtask 1/train_data.json",
    #     "--output", "predictions/anthropic_claude_opus_reason.json",
    #     "--evaluate",
    #     "--reference", "train_data/subtask 1/train_data.json",
    #     "--limit", "300"
    # ],
]


SHLEX_POSIX = os.name != "nt"

def normalize_tasks(raw: List[Union[str, List[str]]]) -> List[List[str]]:
    out: List[List[str]] = []
    for t in raw:
        if isinstance(t, str):
            parts = shlex.split(t, posix=SHLEX_POSIX)
        elif isinstance(t, list):
            parts = [str(x) for x in t]
        else:
            raise TypeError("Jede Task muss String oder Liste von Strings sein.")
        if parts:
            out.append(parts)
    return out

def run_command(cmd_parts: List[str]) -> Tuple[int, str]:
    # Use -u flag for unbuffered output so we see prints in real-time
    full_cmd = [sys.executable, "-u", *cmd_parts]
    try:
        # Explicitly pass through stdout/stderr and flush output
        rc = subprocess.run(
            full_cmd,
            check=False,
            env=os.environ.copy(),
            stdout=None,  # Pass through to parent stdout
            stderr=None,  # Pass through to parent stderr
        ).returncode
        return rc, " ".join(full_cmd)
    except Exception as e:
        print(f"[Runner] Unerwarteter Fehler: {e}", file=sys.stderr)
        return 255, " ".join(full_cmd)

def main():
    if not TASKS:
        print(
            "Keine Tasks definiert."
        )
        sys.exit(2)

    tasks = normalize_tasks(TASKS)

    results: List[Tuple[int, str]] = []
    for i, parts in enumerate(tasks, 1):
        print(f"\n[Runner] ({i}/{len(tasks)}) Starte: {parts[0]} {' '.join(parts[1:])}")
        rc, cmd_str = run_command(parts)
        status = "OK" if rc == 0 else f"FEHLER (rc={rc})"
        print(f"[Runner] Beendet: {status}")
        results.append((rc, cmd_str))

    print("\n===== Zusammenfassung =====")
    failed = 0
    for rc, cmd_str in results:
        mark = "OK" if rc == 0 else f"FEHLER (rc={rc})"
        print(f"{mark}  |  {cmd_str}")
        if rc != 0:
            failed += 1

    sys.exit(0 if failed == 0 else min(failed, 255))

if __name__ == "__main__":
    main()
