# Experiment Commands (PowerShell One-Liners)

This file contains ready-to-run experiment commands for testing different models and prompt templates.

## Quick Reference

- **Models**: Qwen (instruct/thinking), Claude, GPT-4o, DeepSeek, Llama
- **Prompts**: `direct` (simple), `cot` (chain-of-thought)
- **Data**: Train data for evaluation, test data for final predictions
- **All commands are one-liners** - just copy and paste!

---

## Core Experiments: Qwen Models

### Qwen Instruct - Direct Prompt
**Cost**: ~$0.08 | **Examples**: 960 | **Best for**: Baseline comparison, cheapest option
Tests the non-thinking Qwen model with simple direct prompt asking for validity judgment.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_instruct_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Qwen Instruct - Chain of Thought Prompt
**Cost**: ~$0.12 | **Examples**: 960 | **Best for**: Testing if explicit reasoning helps non-thinking models
Tests the non-thinking Qwen model with step-by-step reasoning prompt.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-instruct" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_instruct_cot.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Qwen Thinking - Direct Prompt ⭐ RECOMMENDED
**Cost**: ~$1.20 | **Examples**: 960 | **Best for**: High accuracy with internal reasoning
Uses thinking model with simple prompt. Model reasons internally (258 reasoning tokens). Most efficient for thinking models.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_thinking_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

### Qwen Thinking - Chain of Thought Prompt ⚠️ EXPENSIVE
**Cost**: ~$3.60 | **Examples**: 960 | **Best for**: Research only - likely redundant
Uses thinking model with CoT prompt. Generates 3.4x more reasoning tokens (872). Likely unnecessary since model already thinks internally.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "cot" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_thinking_cot.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

## Testing on Test Data (Final Predictions)

### Qwen Thinking - Direct Prompt on Test Data
**Cost**: ~$1.20+ | **Examples**: Unknown (test set) | **Best for**: Final submission
Once you've identified the best model/prompt, run on actual test data. No evaluation since test data has no labels.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "direct" --input "test_data/subtask 1/test_data_subtask_1.json" --output "predictions/final_test_predictions.json"
```

### Best Model - Direct Prompt on Test Data (Template)
**Cost**: Varies | **Examples**: Unknown (test set) | **Best for**: Final submission
Replace `{best_model}` and `{best_prompt}` with your highest-performing configuration from training experiments.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "{best_model}" --prompt "{best_prompt}" --input "test_data/subtask 1/test_data_subtask_1.json" --output "predictions/final_test_predictions.json"
```

---

## Custom Results CSV Path

### All Experiments to Single CSV
**Best for**: Centralized results tracking
All experiments append to one CSV file instead of separate files per model/prompt combination.

```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_instruct_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json" --results-csv "experiments/all_results.csv"
```

## Results Files

All experiments with `--evaluate` automatically create/append to CSV files:

**Default naming**: `experiments/results_{model}_{prompt}.csv`

Examples:
- `experiments/results_qwen_qwen3-vl-235b-a22b-instruct_direct.csv`
- `experiments/results_qwen_qwen3-vl-235b-a22b-instruct_cot.csv`
- `experiments/results_qwen_qwen3-vl-235b-a22b-thinking_direct.csv`
- `experiments/results_meta-llama_llama-3.1-70b-instruct_direct.csv`

**CSV Columns**:
- Experiment metadata: timestamp, model, prompt_template, input/output files, num_examples
- Main metrics: accuracy, content_effect, combined_score
- Subgroup accuracies: plausible_valid, implausible_valid, plausible_invalid, implausible_invalid
- Content effect breakdown: intra_validity, inter_validity

---

## Recommended Experiment Sequence

### Phase 1: Budget Exploration (Total: ~$1.00)
```
1. DeepSeek Chat - Direct ($0.04)
2. DeepSeek Chat - CoT ($0.08)
3. Qwen Instruct - Direct ($0.08)
4. Qwen Instruct - CoT ($0.12)
5. Gemini Pro - Direct ($0.12)
6. Llama 3.1 70B - Direct ($0.20)
7. Llama 3.1 70B - CoT ($0.32)
```

**Goal**: Identify if CoT helps, establish baseline performance

### Phase 2: Thinking Models (Total: ~$1.20-2.80)
```
8. Qwen Thinking - Direct ($1.20) ⭐ RECOMMENDED
9. (Optional) DeepSeek R1 - Direct ($1.60)
```

**Goal**: Test if thinking models significantly outperform instruct models

### Phase 3: Premium Comparison (Total: ~$0.18)
```
10. Claude 3.5 Sonnet - Direct, 100 examples ($0.10)
11. GPT-4o - Direct, 100 examples ($0.08)
```

**Goal**: Benchmark against top-tier models to see if extra cost is justified

### Phase 4: Final Prediction
```
12. Best model from above on test data (~$0.04-1.20+)
```

**Total estimated cost for all phases**: ~$2.38-4.18

---

## Key Insights from Test Results

### Thinking vs Instruct Models
- **Thinking models** reason internally (see `message.reasoning` field)
- **Direct prompt** is sufficient for thinking models
- **CoT prompt** with thinking models is redundant and 3x more expensive

### OpenRouter Handling
- Reasoning tokens are separated from content
- `message.content` contains only JSON response
- `message.reasoning` contains the internal reasoning process
- Current code works perfectly with both model types

### Prompt Strategy
- **Instruct models**: Try both direct and CoT prompts
- **Thinking models**: Use direct prompt only
- CoT might help non-thinking models organize their reasoning

---

## Quick Copy Commands (Most Important)

**Fastest baseline** (3 seconds):
```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "deepseek/deepseek-chat" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/deepseek_chat_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

**Best value** (good performance/cost):
```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "meta-llama/llama-3.1-70b-instruct" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/llama_70b_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

**Highest accuracy** (recommended for final):
```powershell
conda run -n semeval2026_task11 python experiments/run_experiment.py --model "qwen/qwen3-vl-235b-a22b-thinking" --prompt "direct" --input "train_data/subtask 1/train_data.json" --output "predictions/qwen_thinking_direct.json" --evaluate --reference "train_data/subtask 1/train_data.json"
```

---

## Notes

- 💡 **Thinking models**: Always use `direct` prompt - they already reason internally
- 💰 **Cost control**: Use `--limit` flag to test expensive models on subset first
- 📊 **Results tracking**: All runs automatically append to CSV for easy comparison
- ⚡ **Speed**: Budget models (DeepSeek, Llama) are fastest; thinking models are slowest
- 🎯 **Accuracy vs Cost**: Thinking models likely best accuracy but 15-45x more expensive
