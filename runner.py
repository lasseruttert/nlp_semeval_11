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
    
    # [
    #     "experiments/run_experiment.py",
    #     "--model", "google/gemini-3-flash-preview",
    #     "--prompt", "direct",
    #     "--input", "train_data/subtask 1/train_data.json",
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
    
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.5",
        "--prompt", "direct",
        "--input", "train_data/subtask 1/train_data.json",
        "--output", "predictions/anthropic_claude_opus_direct.json",
        "--evaluate",
        "--reference", "train_data/subtask 1/train_data.json",
    ],
    
    [
        "experiments/run_experiment.py",
        "--model", "anthropic/claude-opus-4.5",
        "--prompt", "reason",
        "--input", "train_data/subtask 1/train_data.json",
        "--output", "predictions/anthropic_claude_opus_reason.json",
        "--evaluate",
        "--reference", "train_data/subtask 1/train_data.json",
        "--limit", "300"
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
            raise TypeError("Jede Task muss String oder Liste von Strings sein.")
        if parts:
            out.append(parts)
    return out

def run_command(cmd_parts: List[str]) -> Tuple[int, str]:
    full_cmd = [sys.executable, *cmd_parts]
    try:
        rc = subprocess.run(
            full_cmd,
            check=False,
            env=os.environ.copy(),
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
