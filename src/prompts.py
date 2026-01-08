"""
Prompt templates for syllogistic reasoning.
"""
from typing import Callable


def direct_prompt(syllogism: str) -> str:
    """
    Simple direct prompt asking for validity judgment.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY only. Completely ignore whether the argument sounds plausible or realistic.

KEY PRINCIPLE:
- VALID = conclusion MUST be true IF premises are true (regardless of content)
- INVALID = conclusion does NOT necessarily follow from premises (even if plausible)

WARNING - Content Bias:
- Plausible arguments can be INVALID: "Most doctors are smart. John is smart. Therefore John is a doctor."
- Absurd arguments can be VALID: "All cats are reptiles. All reptiles are purple. Therefore all cats are purple."

IGNORE: Real-world truth, plausibility, whether it "makes sense"
EVALUATE: Does conclusion follow by logical necessity from premises?

Argument:
{syllogism}

Analyze ONLY logical structure, not content. Assume you know nothing about the real world.

Respond with JSON:
{{
    "validity": true or false
}}"""

    return prompt


def chain_of_thought_prompt(syllogism: str) -> str:
    """
    Chain-of-thought prompt with step-by-step reasoning.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY only. Ignore plausibility and real-world knowledge completely.

KEY: Absurd content can be VALID. Realistic content can be INVALID. Judge only logical structure.

Argument:
{syllogism}

Follow these steps:

1. IDENTIFY: What are the premises and conclusion? (Ignore plausibility)

2. CONVERT to form: "All/No/Some X are Y"
   - Identify: major term (conclusion predicate), minor term (conclusion subject), middle term (in premises only)
   - Don't let content influence this step

3. CHECK structure:
   - Is middle term distributed in at least one premise?
   - Watch for fallacies: undistributed middle, affirming consequent, denying antecedent, illicit major/minor

4. EVALUATE: If premises are true (even if absurd), MUST conclusion be true?
   - Valid argument can have false premises/conclusion
   - Invalid argument can have true premises/conclusion

5. DECIDE: VALID = conclusion follows necessarily (regardless of content)

IGNORE: Real-world truth, plausibility, whether it "makes sense"

Respond with JSON:
{{
    "validity": true or false
}}"""

    return prompt


def normalization_replace_prompt(syllogism: str) -> str:
    """
    Normalize 3-sentence syllogism into strict categorical A/E/I/O form
    and replace concrete terms with variables A,B,C.

    If not possible as a 3-term categorical syllogism, output a fixed
    canonical syllogism whose validity encodes the model's validity
    judgment of the original.
    """

    prompt = f"""/no_think
You are a logic expert. Output must be STRICTLY formatted.

You receive EXACTLY THREE sentences in order:
1) Major premise
2) Minor premise
3) Conclusion

GOAL:
Produce EXACTLY three sentences, separated by single spaces, each ending with a period.
No extra text. No markdown. No JSON. No labels.

ALLOWED SENTENCE TEMPLATES (ONLY these four):
1. All X are Y.
2. No X are Y.
3. Some X are Y.
4. Some X are not Y.

ABSOLUTE FORMAT RULES:
- Use ONLY: "All", "No", "Some", "are", "not", and term placeholders (A/B/C) plus punctuation.
- The word "not" may appear ONLY in: "Some X are not Y."
- NEVER output: "All X are not Y." or "No X are not Y." (rewrite them canonically).
- NEVER use conjunctions/disjunctions inside a term (no "and", "or", commas, parentheses).
- Preserve the original sentence order (premise1, premise2, conclusion) IF conversion is possible.

STEP 1 — CATEGORICAL NORMALIZATION (meaning-preserving):
Rewrite each input sentence into exactly one of the four allowed templates.
Do not weaken/strengthen quantifiers:
- "No X are Y" must NOT become "Some X are not Y"
- "All X are Y" must NOT become "Some X are Y"
Fix common non-canonical forms:
- "All X are not Y"  -> "No X are Y"
- "No X are not Y"   -> "All X are Y"

STEP 2 — VARIABLE REPLACEMENT (3-term only):
Replace ALL concrete class terms with variables using ONLY A, B, C.
- Same original term => same variable everywhere
- Different terms => different variables
- Use EXACTLY 3 variables iff the argument is a standard 3-term categorical syllogism
- If more than 3 distinct class terms are REQUIRED, conversion is NOT POSSIBLE (see fallback)

DETECT “NOT POSSIBLE” (fallback required) if ANY occurs:
- You would need >3 variables (more than 3 distinct class terms)
- Any sentence requires a compound term (e.g., "B and C", "not B" as a term)
- Any sentence is relational/non-categorical (e.g., "loves", "equals", "taller than", "if...then", "most", "exactly one", etc.)
- You cannot express it without adding/removing information

FALLBACK (CRITICAL):
If conversion is NOT POSSIBLE, DO THIS INSTEAD:
1) Decide whether the ORIGINAL 3-sentence argument is logically VALID (premises entail conclusion).
2) Output EXACTLY one of these fixed canonical syllogisms:
- If VALID output:   "All A are B. All B are C. All A are C."
- If INVALID output: "All A are B. All C are B. All A are C."

TINY EXAMPLE (follow EXACTLY):
Input: All cats are mammals. No mammals are reptiles. Therefore no cats are reptiles.
Output: All A are B. No B are C. No A are C.

NOW PROCESS THIS SYLLOGISM (three sentences):
{syllogism}

OUTPUT:
Exactly three sentences separated by spaces, each ending with a period.
"""
    return prompt



# Dictionary of available prompt templates
PROMPT_TEMPLATES: dict[str, Callable[[str], str]] = {
    "direct": direct_prompt,
    "chain_of_thought": chain_of_thought_prompt,
    "cot": chain_of_thought_prompt,
    "normalization": normalization_prompt,
    "normalize": normalization_prompt,
    "normalization_replace": normalization_replace_prompt,
    "normalize_replace": normalization_replace_prompt,
}


def get_prompt_template(template_name: str) -> Callable[[str], str]:
    """
    Get a prompt template by name.

    Args:
        template_name: Name of the template ('direct' or 'chain_of_thought'/'cot')

    Returns:
        Prompt template function

    Raises:
        ValueError: If template name is not recognized
    """
    if template_name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(
            f"Unknown prompt template: {template_name}. "
            f"Available templates: {available}"
        )

    return PROMPT_TEMPLATES[template_name]
