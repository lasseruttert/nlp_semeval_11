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
    fig, ax = plt.subplots(figsize=(14, 6))

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
    fig, ax = plt.subplots(figsize=(12, 6))

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
    ax.set_title('Gemini vs Claude Opus: Raw vs Normalized Data Comparison', fontsize=14, fontweight='bold')
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
    fig, ax = plt.subplots(figsize=(14, 6))

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
    ax.set_title('Accuracy by Syllogism Type: Gemini vs Claude Opus', fontsize=14, fontweight='bold')
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


def load_bert_results():
    """Load BERT experiment results from JSON files."""
    import json

    results = {}

    # Load results from different models/learning rates (raw data)
    raw_path = os.path.join(SCRIPT_DIR, 'bert_different_models_and_lr.json')
    with open(raw_path, 'r') as f:
        raw_results = json.load(f)
    results['raw'] = raw_results

    # Load results from normalized input experiments
    norm_path = os.path.join(SCRIPT_DIR, 'bert_normalized_input.json')
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
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data from experiments (raw/original data)
    # LLM results from CSVs
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_raw = gemini_df[~gemini_df['input_file'].str.contains('polished')].iloc[0]

    opus_path = os.path.join(SCRIPT_DIR, 'results_anthropic_claude-opus-4.5_direct.csv')
    opus_df = pd.read_csv(opus_path)
    opus_raw = opus_df[~opus_df['input_file'].str.contains('polished')].iloc[0]

    # BERT results from JSON
    bert_results = load_bert_results()
    bert_base_2e5 = next(r for r in bert_results['raw']
                         if r['model_name'] == 'bert-base-uncased' and r['learning_rate'] == 2e-5)
    bert_base_3e5 = next(r for r in bert_results['raw']
                         if r['model_name'] == 'bert-base-uncased' and r['learning_rate'] == 3e-5)

    # Prepare data for all 4 categories
    categories = ['Gemini 3 Flash', 'Claude Opus 4.5',
                  'BERT-base\n(lr=2e-5)', 'BERT-base\n(lr=3e-5)']

    accuracy = [gemini_raw['accuracy'], opus_raw['accuracy'],
                bert_base_2e5['test_accuracy'] * 100, bert_base_3e5['test_accuracy'] * 100]

    x = np.arange(len(categories))
    width = 0.5

    # Create accuracy bars (single bar per category, centered)
    colors = [UNI_BLUE, UNI_BLUE, UNI_YELLOW, UNI_YELLOW]
    bars = ax.bar(x, accuracy, width, color=colors)

    # Customize plot
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Raw Data: LLMs vs Fine-tuned BERT', fontsize=14, fontweight='bold')
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
    fig, ax = plt.subplots(figsize=(10, 6))

    # LLM results (Gemini on normalized)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_norm = gemini_df[gemini_df['input_file'].str.contains('polished')].iloc[0]

    # BERT-large on Gemini-normalized
    bert_results = load_bert_results()
    bert_large_gemini = next(r for r in bert_results['normalized']
                              if r['model_name'] == 'bert-large-uncased' and r['source_name'] == 'gemini')

    # Logic solver on Gemini-normalized
    solver_df = load_solver_results()
    solver_gemini = solver_df[solver_df['model'] == 'gemini'].iloc[0]

    # Prepare data
    categories = ['Gemini 3 Flash', 'BERT-large', 'Logic Solver']
    accuracy = [gemini_norm['accuracy'],
                bert_large_gemini['test_accuracy'] * 100,
                solver_gemini['accuracy']]

    x = np.arange(len(categories))
    width = 0.5

    # Create bars with different colors
    colors = [UNI_BLUE, UNI_YELLOW, UNI_GREY]
    bars = ax.bar(x, accuracy, width, color=colors)

    # Customize plot
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Normalized Data: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(90, 100)

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
    """Compare sub-accuracies between LLM and Logic Solver on normalized data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # LLM results (Gemini on normalized)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_norm = gemini_df[gemini_df['input_file'].str.contains('polished')].iloc[0]

    # Logic solver on Gemini-normalized
    solver_df = load_solver_results()
    solver_gemini = solver_df[solver_df['model'] == 'gemini'].iloc[0]

    # Categories: 4 syllogism types
    categories = ['Plausible\nValid', 'Implausible\nValid',
                  'Plausible\nInvalid', 'Implausible\nInvalid']

    # Prepare data arrays
    gemini_acc = [gemini_norm['acc_plausible_valid'], gemini_norm['acc_implausible_valid'],
                  gemini_norm['acc_plausible_invalid'], gemini_norm['acc_implausible_invalid']]
    solver_acc = [solver_gemini['acc_plausible_valid'], solver_gemini['acc_implausible_valid'],
                  solver_gemini['acc_plausible_invalid'], solver_gemini['acc_implausible_invalid']]

    x = np.arange(len(categories))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, gemini_acc, width, label='Gemini 3 Flash', color=UNI_BLUE)
    bars2 = ax.bar(x + width/2, solver_acc, width, label='Logic Solver', color=UNI_YELLOW)

    # Customize plot
    ax.set_xlabel('Syllogism Type', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('LLM vs Logic Solver: Accuracy by Syllogism Type', fontsize=14, fontweight='bold')
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
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    output_path = os.path.join(SCRIPT_DIR, 'plot_llm_vs_solver_accuracy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_llm_vs_solver():
    """Create bar chart comparing content effects between LLM and Logic Solver on normalized data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a 4th color (lighter blue)
    UNI_LIGHT_BLUE = '#6699cc'

    # LLM results (Gemini on normalized)
    gemini_path = os.path.join(SCRIPT_DIR, 'results_google_gemini-3-flash-preview_direct.csv')
    gemini_df = pd.read_csv(gemini_path)
    gemini_norm = gemini_df[gemini_df['input_file'].str.contains('polished')].iloc[0]

    # Logic solver on Gemini-normalized
    solver_df = load_solver_results()
    solver_gemini = solver_df[solver_df['model'] == 'gemini'].iloc[0]

    # Prepare data
    categories = ['Gemini 3 Flash', 'Logic Solver']
    intra_effect = [gemini_norm['content_effect_intra_validity'], solver_gemini['content_effect_intra_validity']]
    inter_effect = [gemini_norm['content_effect_inter_validity'], solver_gemini['content_effect_inter_validity']]
    content_effect = [gemini_norm['content_effect'], solver_gemini['content_effect']]
    combined_score = [gemini_norm['combined_score'], solver_gemini['combined_score']]

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
    ax.set_title('Content Effects: LLM vs Logic Solver', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 50)

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_title('Content Effect Breakdown: Gemini vs Claude Opus', fontsize=14, fontweight='bold')
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
    plot_normalized_accuracy_comparison()
    plot_llm_vs_solver()
    plot_llm_vs_solver_accuracy()

    print("\nDone! All plots generated successfully.")


if __name__ == '__main__':
    main()
