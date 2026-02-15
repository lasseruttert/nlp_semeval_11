#!/usr/bin/env python3
# runner.py
import os
import sys
import shlex
import subprocess
from typing import List, Tuple, Union

# ------------------------------------------------------------
# LLM Experiment Tasks - Test Data Predictions
#
# For each LLM (Qwen, Gemini, Opus), run 3 configurations:
# 1. Raw test data + direct prompt
# 2. Normalized test data + direct prompt
# 3. Normalized test data + formal prompt (new)
# ------------------------------------------------------------
TASKS: List[Union[str, List[str]]] = [
    # ===== QWEN EXPERIMENTS =====

    # Qwen - Raw test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "qwen/qwen3-vl-235b-a22b-instruct",
        "--prompt", "direct",
        "--input", "test_data/subtask 1/test_data_subtask_1.json",
        "--output", "predictions/qwen_raw_direct.json"
    ],

    # Qwen - Normalized test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "qwen/qwen3-vl-235b-a22b-instruct",
        "--prompt", "direct",
        "--input", "data/polished/test_polished_variables_qwen.json",
        "--output", "predictions/qwen_normalized_direct.json"
    ],

    # Qwen - Normalized test data + formal prompt
    [
        "experiments/run_experiment.py",
        "--model", "qwen/qwen3-vl-235b-a22b-instruct",
        "--prompt", "formal",
        "--input", "data/polished/test_polished_variables_qwen.json",
        "--output", "predictions/qwen_normalized_formal.json"
    ],

    # ===== GEMINI EXPERIMENTS =====

    # Gemini - Raw test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "google/gemini-3-flash-preview",
        "--prompt", "direct",
        "--input", "test_data/subtask 1/test_data_subtask_1.json",
        "--output", "predictions/gemini_raw_direct.json"
    ],

    # Gemini - Normalized test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "google/gemini-3-flash-preview",
        "--prompt", "direct",
        "--input", "data/polished/test_polished_variables_gemini.json",
        "--output", "predictions/gemini_normalized_direct.json"
    ],

    # Gemini - Normalized test data + formal prompt
    [
        "experiments/run_experiment.py",
        "--model", "google/gemini-3-flash-preview",
        "--prompt", "formal",
        "--input", "data/polished/test_polished_variables_gemini.json",
        "--output", "predictions/gemini_normalized_formal.json"
    ],

    # ===== OPUS EXPERIMENTS =====

    # Opus - Raw test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.5",
        "--prompt", "direct",
        "--input", "test_data/subtask 1/test_data_subtask_1.json",
        "--output", "predictions/opus_raw_direct.json"
    ],

    # Opus - Normalized test data + direct prompt
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.5",
        "--prompt", "direct",
        "--input", "data/polished/test_polished_variables_opus.json",
        "--output", "predictions/opus_normalized_direct.json"
    ],

    # Opus - Normalized test data + formal prompt
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.5",
        "--prompt", "formal",
        "--input", "data/polished/test_polished_variables_opus.json",
        "--output", "predictions/opus_normalized_formal.json"
    ],
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
            raise TypeError("Each task must be a string or list of strings.")
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
        print(f"[Runner] Unexpected error: {e}", file=sys.stderr)
        return 255, " ".join(full_cmd)

def main():
    if not TASKS:
        print("No tasks defined.")
        sys.exit(2)

    tasks = normalize_tasks(TASKS)

    results: List[Tuple[int, str]] = []
    for i, parts in enumerate(tasks, 1):
        print(f"\n[Runner] ({i}/{len(tasks)}) Starting: {parts[0]} {' '.join(parts[1:])}")
        rc, cmd_str = run_command(parts)
        status = "OK" if rc == 0 else f"ERROR (rc={rc})"
        print(f"[Runner] Finished: {status}")
        results.append((rc, cmd_str))

    print("\n===== SUMMARY =====")
    failed = 0
    for rc, cmd_str in results:
        mark = "OK" if rc == 0 else f"ERROR (rc={rc})"
        print(f"{mark}  |  {cmd_str}")
        if rc != 0:
            failed += 1

    sys.exit(0 if failed == 0 else min(failed, 255))

if __name__ == "__main__":
    main()
