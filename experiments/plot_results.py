"""
Generate visualization plots from experiment results.

Creates two plots:
1. Qwen Prompt Comparison - Comparing 9 different prompts
2. Gemini vs Opus (Raw vs Normalized) - Comparing models across data types
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# University color palette
UNI_BLUE = '#004e9f'
UNI_YELLOW = '#fcba00'
UNI_GREY = '#909085'

# Script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_qwen_results():
    """Load Qwen experiment results for all prompts."""
    prompts = ['direct', 'cot', 'counterexample', 'venn', 'rules',
               'minimal', 'symbolic', 'comprehensive', 'fallacy']

    data = {'prompt': [], 'accuracy': [], 'content_effect': [], 'combined_score': []}

    for prompt in prompts:
        csv_path = os.path.join(SCRIPT_DIR, f'results_qwen_qwen3-vl-235b-a22b-instruct_{prompt}.csv')
        df = pd.read_csv(csv_path)
        data['prompt'].append(prompt)
        data['accuracy'].append(df['accuracy'].iloc[0])
        data['content_effect'].append(df['content_effect'].iloc[0])
        data['combined_score'].append(df['combined_score'].iloc[0])

    return pd.DataFrame(data)


def load_gemini_opus_results():
    """Load Gemini and Opus results for raw vs normalized comparison."""
    results = []

    # Gemini results
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)

    # Row 0: raw data (train_data.json), Row 1: normalized (polished_syllogisms_variables_opus.json)
    for idx, row in gemini_df.iterrows():
        is_normalized = 'polished' in row['input_file']
        results.append({
            'model': 'Gemini 3 Flash Preview',
            'data_type': 'Normalized' if is_normalized else 'Raw',
            'accuracy': row['accuracy'],
            'content_effect': row['content_effect'],
            'combined_score': row['combined_score'],
            'acc_plausible_valid': row['acc_plausible_valid'],
            'acc_implausible_valid': row['acc_implausible_valid'],
            'acc_plausible_invalid': row['acc_plausible_invalid'],
            'acc_implausible_invalid': row['acc_implausible_invalid'],
            'content_effect_intra': row['content_effect_intra_validity'],
            'content_effect_inter': row['content_effect_inter_validity']
        })

    # Opus results
    opus_path = os.path.join(SCRIPT_DIR, 'results_anthropic_claude-opus-4.5_direct.csv')
    opus_df = pd.read_csv(opus_path)

    for idx, row in opus_df.iterrows():
        is_normalized = 'polished' in row['input_file']
        results.append({
            'model': 'Claude Opus 4.5',
            'data_type': 'Normalized' if is_normalized else 'Raw',
            'accuracy': row['accuracy'],
            'content_effect': row['content_effect'],
            'combined_score': row['combined_score'],
            'acc_plausible_valid': row['acc_plausible_valid'],
            'acc_implausible_valid': row['acc_implausible_valid'],
            'acc_plausible_invalid': row['acc_plausible_invalid'],
            'acc_implausible_invalid': row['acc_implausible_invalid'],
            'content_effect_intra': row['content_effect_intra_validity'],
            'content_effect_inter': row['content_effect_inter_validity']
        })

    return pd.DataFrame(results)


def plot_qwen_prompts(df):
    """Create bar chart comparing Qwen prompts across metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))

    prompts = df['prompt'].tolist()
    x = np.arange(len(prompts))
    width = 0.25

    # Create bars for each metric
    bars1 = ax.bar(x - width, df['accuracy'], width, label='Accuracy (%)', color=UNI_BLUE)
    bars2 = ax.bar(x, df['content_effect'], width, label='Content Effect', color=UNI_YELLOW)
    bars3 = ax.bar(x + width, df['combined_score'], width, label='Combined Score', color=UNI_GREY)

    # Customize plot
    ax.set_xlabel('Prompt Template', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Qwen Model: Comparison of Prompt Templates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_qwen_prompts.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gemini_opus_comparison(df):
    """Create grouped bar chart comparing Gemini vs Opus (raw vs normalized)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Create grouped structure: Gemini Raw, Gemini Norm, Opus Raw, Opus Norm
    categories = ['Gemini 3 Flash\n(Raw)', 'Gemini 3 Flash\n(Normalized)',
                  'Claude Opus 4.5\n(Raw)', 'Claude Opus 4.5\n(Normalized)']

    # Extract data in order
    gemini_raw = df[(df['model'] == 'Gemini 3 Flash Preview') & (df['data_type'] == 'Raw')].iloc[0]
    gemini_norm = df[(df['model'] == 'Gemini 3 Flash Preview') & (df['data_type'] == 'Normalized')].iloc[0]
    opus_raw = df[(df['model'] == 'Claude Opus 4.5') & (df['data_type'] == 'Raw')].iloc[0]
    opus_norm = df[(df['model'] == 'Claude Opus 4.5') & (df['data_type'] == 'Normalized')].iloc[0]

    accuracy = [gemini_raw['accuracy'], gemini_norm['accuracy'],
                opus_raw['accuracy'], opus_norm['accuracy']]
    content_effect = [gemini_raw['content_effect'], gemini_norm['content_effect'],
                      opus_raw['content_effect'], opus_norm['content_effect']]
    combined_score = [gemini_raw['combined_score'], gemini_norm['combined_score'],
                      opus_raw['combined_score'], opus_norm['combined_score']]

    x = np.arange(len(categories))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, accuracy, width, label='Accuracy (%)', color=UNI_BLUE)
    bars2 = ax.bar(x, content_effect, width, label='Content Effect', color=UNI_YELLOW)
    bars3 = ax.bar(x + width, combined_score, width, label='Combined Score', color=UNI_GREY)

    # Customize plot
    ax.set_xlabel('Model and Data Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Raw vs Normalized Data: Gemini vs Claude Opus', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)

    # Add vertical separator between models
    ax.axvline(x=1.5, color='lightgray', linestyle='--', linewidth=1)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_gemini_opus_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_syllogism_accuracy(df):
    """Create bar chart showing accuracy on different syllogism types for Gemini and Opus."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Categories: 4 syllogism types x 4 model/data combinations
    syllogism_types = ['Plausible\nValid', 'Implausible\nValid',
                       'Plausible\nInvalid', 'Implausible\nInvalid']

    # Extract data
    gemini_raw = df[(df['model'] == 'Gemini 3 Flash Preview') & (df['data_type'] == 'Raw')].iloc[0]
    gemini_norm = df[(df['model'] == 'Gemini 3 Flash Preview') & (df['data_type'] == 'Normalized')].iloc[0]
    opus_raw = df[(df['model'] == 'Claude Opus 4.5') & (df['data_type'] == 'Raw')].iloc[0]
    opus_norm = df[(df['model'] == 'Claude Opus 4.5') & (df['data_type'] == 'Normalized')].iloc[0]

    # Prepare data arrays
    gemini_raw_acc = [gemini_raw['acc_plausible_valid'], gemini_raw['acc_implausible_valid'],
                      gemini_raw['acc_plausible_invalid'], gemini_raw['acc_implausible_invalid']]
    gemini_norm_acc = [gemini_norm['acc_plausible_valid'], gemini_norm['acc_implausible_valid'],
                       gemini_norm['acc_plausible_invalid'], gemini_norm['acc_implausible_invalid']]
    opus_raw_acc = [opus_raw['acc_plausible_valid'], opus_raw['acc_implausible_valid'],
                    opus_raw['acc_plausible_invalid'], opus_raw['acc_implausible_invalid']]
    opus_norm_acc = [opus_norm['acc_plausible_valid'], opus_norm['acc_implausible_valid'],
                     opus_norm['acc_plausible_invalid'], opus_norm['acc_implausible_invalid']]

    x = np.arange(len(syllogism_types))
    width = 0.2

    # Create bars with lighter shades for normalized
    bars1 = ax.bar(x - 1.5*width, gemini_raw_acc, width, label='Gemini (Raw)', color=UNI_BLUE)
    bars2 = ax.bar(x - 0.5*width, gemini_norm_acc, width, label='Gemini (Normalized)',
                   color=UNI_BLUE, alpha=0.5)
    bars3 = ax.bar(x + 0.5*width, opus_raw_acc, width, label='Opus (Raw)', color=UNI_YELLOW)
    bars4 = ax.bar(x + 1.5*width, opus_norm_acc, width, label='Opus (Normalized)',
                   color=UNI_YELLOW, alpha=0.5)

    # Customize plot
    ax.set_xlabel('Syllogism Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Raw vs Normalized Data: Accuracy by Syllogism Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(syllogism_types)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_syllogism_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def load_bert_csv_results():
    """Load BERT experiment results from CSV files.

    Returns dict with:
    - 'bert_base_original': results for bert-base on original data
    - 'bert_large_original': results for bert-large on original data
    - 'bert_base_best_normalized': best result among opus/gemini/qwen for bert-base
    - 'bert_large_best_normalized': best result among opus/gemini/qwen for bert-large
    """
    results = {}

    # Load all BERT CSV files
    bert_files = {
        'bert_base_original': 'bert_results_bert_base_uncased_original.csv',
        'bert_base_gemini': 'bert_results_bert_base_uncased_gemini.csv',
        'bert_base_opus': 'bert_results_bert_base_uncased_opus.csv',
        'bert_base_qwen': 'bert_results_bert_base_uncased_qwen.csv',
        'bert_large_original': 'bert_results_bert_large_uncased_original.csv',
        'bert_large_gemini': 'bert_results_bert_large_uncased_gemini.csv',
        'bert_large_opus': 'bert_results_bert_large_uncased_opus.csv',
        'bert_large_qwen': 'bert_results_bert_large_uncased_qwen.csv',
    }

    for key, filename in bert_files.items():
        csv_path = os.path.join(SCRIPT_DIR, filename)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results[key] = df.iloc[-1].to_dict()  # Get last row (most recent run)

    # Find best normalized result for bert_base (by combined_score)
    bert_base_normalized = ['bert_base_gemini', 'bert_base_opus', 'bert_base_qwen']
    best_base = max(
        [results[k] for k in bert_base_normalized if k in results],
        key=lambda x: x['combined_score']
    )
    results['bert_base_best_normalized'] = best_base

    # Find best normalized result for bert_large (by combined_score)
    bert_large_normalized = ['bert_large_gemini', 'bert_large_opus', 'bert_large_qwen']
    best_large = max(
        [results[k] for k in bert_large_normalized if k in results],
        key=lambda x: x['combined_score']
    )
    results['bert_large_best_normalized'] = best_large

    return results


def load_bert_results():
    """Load BERT experiment results from JSON files (legacy)."""
    import json

    results = {}

    # Load results from different models/learning rates (raw data)
    raw_path = os.path.join(SCRIPT_DIR, 'bert_different_models_and_lr.json')
    if os.path.exists(raw_path):
        with open(raw_path, 'r') as f:
            raw_results = json.load(f)
        results['raw'] = raw_results

    # Load results from normalized input experiments
    norm_path = os.path.join(SCRIPT_DIR, 'bert_normalized_input.json')
    if os.path.exists(norm_path):
        with open(norm_path, 'r') as f:
            norm_results = json.load(f)
        results['normalized'] = norm_results

    return results


def load_solver_results():
    """Load logic solver results from CSV."""
    solver_path = os.path.join(SCRIPT_DIR, 'solver.csv')
    return pd.read_csv(solver_path)


def plot_raw_accuracy_comparison():
    """Create bar chart comparing accuracy of LLMs vs fine-tuned BERT on raw data."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Data from experiments (raw/original data)
    # LLM results from CSVs
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_raw = gemini_df[~gemini_df['input_file'].str.contains('polished')].iloc[0]

    opus_path = os.path.join(SCRIPT_DIR, 'results_anthropic_claude-opus-4.5_direct.csv')
    opus_df = pd.read_csv(opus_path)
    opus_raw = opus_df[~opus_df['input_file'].str.contains('polished')].iloc[0]

    # BERT results from new CSV files
    bert_results = load_bert_csv_results()
    bert_base_original = bert_results.get('bert_base_original', {})
    bert_large_original = bert_results.get('bert_large_original', {})

    # Prepare data for all 4 categories
    categories = ['Gemini 3 Flash', 'Claude Opus 4.5',
                  'BERT', 'BERT-large']

    accuracy = [
        gemini_raw['accuracy'],
        opus_raw['accuracy'],
        bert_base_original.get('accuracy', 0),
        bert_large_original.get('accuracy', 0)
    ]

    x = np.arange(len(categories))
    width = 0.5

    # Create accuracy bars (single bar per category, centered)
    colors = [UNI_BLUE, UNI_BLUE, UNI_YELLOW, UNI_YELLOW]
    bars = ax.bar(x, accuracy, width, color=colors)

    # Customize plot
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Raw Data: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)

    # Add vertical separator between LLMs and BERT
    ax.axvline(x=1.5, color='lightgray', linestyle='--', linewidth=1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_raw_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_normalized_accuracy_comparison():
    """Create bar chart comparing accuracy on normalized data across all approaches."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # LLM results (Gemini on normalized)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_norm = gemini_df[gemini_df['input_file'].str.contains('polished')].iloc[0]

    # BERT results from new CSV files (best among opus/gemini/qwen)
    bert_results = load_bert_csv_results()
    bert_base_best = bert_results.get('bert_base_best_normalized', {})
    bert_large_best = bert_results.get('bert_large_best_normalized', {})

    # Logic solver on Gemini-normalized
    solver_df = load_solver_results()
    solver_gemini = solver_df[solver_df['model'] == 'gemini'].iloc[0]

    # Prepare data - include best BERT and BERT-large
    categories = ['Gemini 3 Flash', 'BERT', 'BERT-large', 'Logic Solver']
    accuracy = [
        gemini_norm['accuracy'],
        bert_base_best.get('accuracy', 0),
        bert_large_best.get('accuracy', 0),
        solver_gemini['accuracy']
    ]

    x = np.arange(len(categories))
    width = 0.5

    # Create bars with different colors
    colors = [UNI_BLUE, UNI_YELLOW, UNI_YELLOW, UNI_GREY]
    bars = ax.bar(x, accuracy, width, color=colors)

    # Customize plot
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Normalized Data: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(90, 100)

    # Add vertical separator between LLM and BERT/Solver
    ax.axvline(x=0.5, color='lightgray', linestyle='--', linewidth=1)
    ax.axvline(x=2.5, color='lightgray', linestyle='--', linewidth=1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_normalized_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_llm_vs_solver_accuracy():
    """Compare sub-accuracies between LLM, BERT, and Logic Solver on normalized data."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # LLM results (Gemini on normalized)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_norm = gemini_df[gemini_df['input_file'].str.contains('polished')].iloc[0]

    # BERT results (best normalized)
    bert_results = load_bert_csv_results()
    bert_best = bert_results.get('bert_base_best_normalized', {})

    # Logic solver on Gemini-normalized
    solver_df = load_solver_results()
    solver_gemini = solver_df[solver_df['model'] == 'gemini'].iloc[0]

    # Categories: 4 syllogism types
    categories = ['Plausible\nValid', 'Implausible\nValid',
                  'Plausible\nInvalid', 'Implausible\nInvalid']

    # Prepare data arrays
    gemini_acc = [gemini_norm['acc_plausible_valid'], gemini_norm['acc_implausible_valid'],
                  gemini_norm['acc_plausible_invalid'], gemini_norm['acc_implausible_invalid']]
    bert_acc = [bert_best.get('acc_plausible_valid', 0), bert_best.get('acc_implausible_valid', 0),
                bert_best.get('acc_plausible_invalid', 0), bert_best.get('acc_implausible_invalid', 0)]
    solver_acc = [solver_gemini['acc_plausible_valid'], solver_gemini['acc_implausible_valid'],
                  solver_gemini['acc_plausible_invalid'], solver_gemini['acc_implausible_invalid']]

    x = np.arange(len(categories))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, gemini_acc, width, label='Gemini 3 Flash', color=UNI_BLUE)
    bars2 = ax.bar(x, bert_acc, width, label='BERT', color=UNI_YELLOW)
    bars3 = ax.bar(x + width, solver_acc, width, label='Logic Solver', color=UNI_GREY)

    # Customize plot
    ax.set_xlabel('Syllogism Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Normalized Data: Accuracy by Syllogism Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_llm_vs_solver_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_llm_vs_solver():
    """Create bar chart comparing content effects between LLM, BERT, and Logic Solver on normalized data."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Define a 4th color (lighter blue)
    UNI_LIGHT_BLUE = '#6699cc'

    # LLM results (Gemini on normalized)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_norm = gemini_df[gemini_df['input_file'].str.contains('polished')].iloc[0]

    # BERT results (best normalized)
    bert_results = load_bert_csv_results()
    bert_best = bert_results.get('bert_base_best_normalized', {})

    # Logic solver on Gemini-normalized
    solver_df = load_solver_results()
    solver_gemini = solver_df[solver_df['model'] == 'gemini'].iloc[0]

    # Prepare data
    categories = ['Gemini 3 Flash', 'BERT', 'Logic Solver']
    intra_effect = [
        gemini_norm['content_effect_intra_validity'],
        bert_best.get('content_effect_intra_validity', 0),
        solver_gemini['content_effect_intra_validity']
    ]
    inter_effect = [
        gemini_norm['content_effect_inter_validity'],
        bert_best.get('content_effect_inter_validity', 0),
        solver_gemini['content_effect_inter_validity']
    ]
    content_effect = [
        gemini_norm['content_effect'],
        bert_best.get('content_effect', 0),
        solver_gemini['content_effect']
    ]
    combined_score = [
        gemini_norm['combined_score'],
        bert_best.get('combined_score', 0),
        solver_gemini['combined_score']
    ]

    x = np.arange(len(categories))
    width = 0.2

    # Create bars
    bars1 = ax.bar(x - 1.5*width, intra_effect, width, label='Intra-Validity Effect', color=UNI_BLUE)
    bars2 = ax.bar(x - 0.5*width, inter_effect, width, label='Inter-Validity Effect', color=UNI_YELLOW)
    bars3 = ax.bar(x + 0.5*width, content_effect, width, label='Content Effect', color=UNI_GREY)
    bars4 = ax.bar(x + 1.5*width, combined_score, width, label='Combined Score', color=UNI_LIGHT_BLUE)

    # Customize plot
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Normalized Data: Content Effect and Combined Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 55)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_llm_vs_solver.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_content_effect_breakdown(df):
    """Create bar chart showing content effect breakdown (intra vs inter validity)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Categories: 4 model/data combinations
    categories = ['Gemini\n(Raw)', 'Gemini\n(Normalized)',
                  'Opus\n(Raw)', 'Opus\n(Normalized)']

    # Extract data
    gemini_raw = df[(df['model'] == 'Gemini 3 Flash Preview') & (df['data_type'] == 'Raw')].iloc[0]
    gemini_norm = df[(df['model'] == 'Gemini 3 Flash Preview') & (df['data_type'] == 'Normalized')].iloc[0]
    opus_raw = df[(df['model'] == 'Claude Opus 4.5') & (df['data_type'] == 'Raw')].iloc[0]
    opus_norm = df[(df['model'] == 'Claude Opus 4.5') & (df['data_type'] == 'Normalized')].iloc[0]

    intra_validity = [gemini_raw['content_effect_intra'], gemini_norm['content_effect_intra'],
                      opus_raw['content_effect_intra'], opus_norm['content_effect_intra']]
    inter_validity = [gemini_raw['content_effect_inter'], gemini_norm['content_effect_inter'],
                      opus_raw['content_effect_inter'], opus_norm['content_effect_inter']]
    total_effect = [gemini_raw['content_effect'], gemini_norm['content_effect'],
                    opus_raw['content_effect'], opus_norm['content_effect']]

    x = np.arange(len(categories))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, intra_validity, width, label='Intra-Validity Effect', color=UNI_BLUE)
    bars2 = ax.bar(x, inter_validity, width, label='Inter-Validity Effect', color=UNI_YELLOW)
    bars3 = ax.bar(x + width, total_effect, width, label='Total Content Effect', color=UNI_GREY)

    # Customize plot
    ax.set_xlabel('Model and Data Type', fontsize=12)
    ax.set_ylabel('Content Effect', fontsize=12)
    ax.set_title('Raw vs Normalized Data: Content Effect Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')

    # Add vertical separator between models
    ax.axvline(x=1.5, color='lightgray', linestyle='--', linewidth=1)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_content_effect_breakdown.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_raw_content_effect_comparison():
    """Create bar chart comparing content effect and combined score on raw data: LLMs vs BERT."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Define a 4th color (lighter blue)
    UNI_LIGHT_BLUE = '#6699cc'

    # LLM results (raw data)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_raw = gemini_df[~gemini_df['input_file'].str.contains('polished')].iloc[0]

    opus_path = os.path.join(SCRIPT_DIR, 'results_anthropic_claude-opus-4.5_direct.csv')
    opus_df = pd.read_csv(opus_path)
    opus_raw = opus_df[~opus_df['input_file'].str.contains('polished')].iloc[0]

    # BERT results (original data)
    bert_results = load_bert_csv_results()
    bert_original = bert_results.get('bert_base_original', {})
    bert_large_original = bert_results.get('bert_large_original', {})

    # Prepare data
    categories = ['Gemini 3 Flash', 'Claude Opus 4.5', 'BERT', 'BERT-large']
    intra_effect = [
        gemini_raw['content_effect_intra_validity'],
        opus_raw['content_effect_intra_validity'],
        bert_original.get('content_effect_intra_validity', 0),
        bert_large_original.get('content_effect_intra_validity', 0)
    ]
    inter_effect = [
        gemini_raw['content_effect_inter_validity'],
        opus_raw['content_effect_inter_validity'],
        bert_original.get('content_effect_inter_validity', 0),
        bert_large_original.get('content_effect_inter_validity', 0)
    ]
    content_effect = [
        gemini_raw['content_effect'],
        opus_raw['content_effect'],
        bert_original.get('content_effect', 0),
        bert_large_original.get('content_effect', 0)
    ]
    combined_score = [
        gemini_raw['combined_score'],
        opus_raw['combined_score'],
        bert_original.get('combined_score', 0),
        bert_large_original.get('combined_score', 0)
    ]

    x = np.arange(len(categories))
    width = 0.2

    # Create bars
    bars1 = ax.bar(x - 1.5*width, intra_effect, width, label='Intra-Validity Effect', color=UNI_BLUE)
    bars2 = ax.bar(x - 0.5*width, inter_effect, width, label='Inter-Validity Effect', color=UNI_YELLOW)
    bars3 = ax.bar(x + 0.5*width, content_effect, width, label='Content Effect', color=UNI_GREY)
    bars4 = ax.bar(x + 1.5*width, combined_score, width, label='Combined Score', color=UNI_LIGHT_BLUE)

    # Customize plot
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Raw Data: Content Effect and Combined Score', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 55)

    # Add vertical separator between LLMs and BERT
    ax.axvline(x=1.5, color='lightgray', linestyle='--', linewidth=1)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_raw_content_effect.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all plots."""
    print("Loading Qwen results...")
    qwen_df = load_qwen_results()
    print(qwen_df)
    print()

    print("Loading Gemini/Opus results...")
    gemini_opus_df = load_gemini_opus_results()
    print(gemini_opus_df)
    print()

    print("Generating plots...")
    plot_qwen_prompts(qwen_df)
    plot_gemini_opus_comparison(gemini_opus_df)
    plot_syllogism_accuracy(gemini_opus_df)
    plot_content_effect_breakdown(gemini_opus_df)

    # New comparison plots
    print("\nGenerating combined approach comparison plots...")
    plot_raw_accuracy_comparison()
    plot_raw_content_effect_comparison()
    plot_normalized_accuracy_comparison()
    plot_llm_vs_solver()
    plot_llm_vs_solver_accuracy()

    print("\nDone! All plots generated successfully.")


if __name__ == '__main__':
    main()
