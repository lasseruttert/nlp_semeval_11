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

Respond with ONLY: true or false"""

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

Respond with ONLY: true or false"""

    return prompt


def counterexample_prompt(syllogism: str) -> str:
    """
    Prompt that asks the model to find counterexamples to test validity.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY by testing for counterexamples.

KEY PRINCIPLE:
- VALID = NO possible counterexample exists (conclusion MUST be true if premises are true)
- INVALID = At least one counterexample exists (premises can be true while conclusion is false)

IMPORTANT: Ignore real-world plausibility. Test ONLY logical structure.

METHOD - Counterexample Testing:
1. Assume the premises are true
2. Try to construct a scenario where premises are true BUT conclusion is false
3. If you can find such a scenario → INVALID
4. If no such scenario is possible → VALID

EXAMPLE (INVALID):
"All cats are animals. All dogs are animals. Therefore all cats are dogs."
- Counterexample: Cats and dogs are both animals, but cats are not dogs
- Premises true, conclusion false → INVALID

EXAMPLE (VALID):
"All A are B. All B are C. Therefore all A are C."
- Try to make premises true but conclusion false: impossible
- No counterexample exists → VALID

Argument:
{syllogism}

Test for counterexamples. Ignore whether the content is realistic.

Respond with ONLY: true or false"""

    return prompt


def venn_diagram_prompt(syllogism: str) -> str:
    """
    Prompt that asks the model to visualize using Venn diagrams.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY using Venn diagram visualization.

KEY: Ignore real-world content. Focus only on set relationships.

Argument:
{syllogism}

METHOD - Venn Diagram Analysis:
1. IDENTIFY three terms (subject, predicate, middle term)
2. VISUALIZE with three overlapping circles representing these terms
3. DRAW what the premises require:
   - "All X are Y" → X circle entirely within Y circle
   - "No X are Y" → X and Y circles don't overlap
   - "Some X are Y" → X and Y circles must overlap (at least one element in intersection)
   - "Some X are not Y" → Part of X circle outside Y circle
4. CHECK: Given the premise constraints, does the conclusion ALWAYS hold?
   - If YES in all valid arrangements → VALID
   - If NO in some arrangement → INVALID

IGNORE: Real-world truth, plausibility
EVALUATE: Does the diagram forced by premises guarantee the conclusion?

Respond with ONLY: true or false"""

    return prompt


def syllogistic_rules_prompt(syllogism: str) -> str:
    """
    Prompt that applies formal syllogistic rules.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY using classical syllogistic rules.

IGNORE real-world content completely. Apply formal rules only.

Argument:
{syllogism}

STEP-BY-STEP VALIDATION:

1. IDENTIFY TERMS:
   - Major term (predicate of conclusion)
   - Minor term (subject of conclusion)
   - Middle term (appears in both premises, not in conclusion)

2. CHECK DISTRIBUTION:
   - In "All X are Y": X is distributed, Y is not
   - In "No X are Y": both X and Y are distributed
   - In "Some X are Y": neither X nor Y is distributed
   - In "Some X are not Y": Y is distributed, X is not

3. APPLY VALIDITY RULES (if ANY rule violated → INVALID):
   - Rule 1: Middle term must be distributed in at least one premise
   - Rule 2: If a term is distributed in conclusion, it must be distributed in its premise
   - Rule 3: From two negative premises, no valid conclusion
   - Rule 4: If one premise is negative, conclusion must be negative
   - Rule 5: From two particular premises, no valid conclusion

4. DECIDE:
   - All rules satisfied → VALID
   - Any rule violated → INVALID

Respond with ONLY: true or false"""

    return prompt


def minimal_prompt(syllogism: str) -> str:
    """
    Minimal prompt without much guidance.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""Evaluate the logical validity of this argument. Ignore real-world plausibility.

Argument:
{syllogism}

Is this argument logically valid? (Valid means: IF the premises are true, the conclusion MUST be true)

Respond with ONLY: true or false"""

    return prompt


def symbolic_logic_prompt(syllogism: str) -> str:
    """
    Prompt that converts to symbolic logic notation.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY using symbolic logic.

IGNORE content and plausibility. Analyze structure only.

Argument:
{syllogism}

METHOD - Symbolic Analysis:

1. CONVERT to symbolic form:
   - "All X are Y" → ∀x(X(x) → Y(x))
   - "No X are Y" → ∀x(X(x) → ¬Y(x))
   - "Some X are Y" → ∃x(X(x) ∧ Y(x))
   - "Some X are not Y" → ∃x(X(x) ∧ ¬Y(x))

2. IDENTIFY logical structure:
   - What are the predicates (A, B, C)?
   - How are they connected?

3. TEST logical entailment:
   - Assume premises are true
   - Does conclusion follow by logical necessity?
   - Can you derive conclusion from premises?

4. WATCH for common fallacies:
   - Undistributed middle
   - Illicit major/minor
   - Affirming the consequent
   - Denying the antecedent

IMPORTANT: Absurd content can be VALID. Plausible content can be INVALID.

Respond with ONLY: true or false"""

    return prompt


def comprehensive_prompt(syllogism: str) -> str:
    """
    Comprehensive prompt with examples, rules, multiple strategies, and warnings.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY only. COMPLETELY IGNORE real-world plausibility, truth, or whether the argument "makes sense."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY PRINCIPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID = If premises are true, conclusion MUST be true (by logical necessity)
INVALID = Conclusion does NOT necessarily follow, even if premises are true

CRITICAL: Content is IRRELEVANT. Only structure matters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES - Study these carefully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID EXAMPLES (absurd content, but logically sound):

1. "All cats are reptiles. All reptiles are purple. Therefore all cats are purple."
   → VALID (perfect transitive structure, even though content is false)

2. "No unicorns are real things. All real things exist. Therefore no unicorns exist."
   → VALID (logical structure is sound, regardless of content paradoxes)

3. "All A are B. All B are C. Therefore all A are C."
   → VALID (classic Barbara syllogism - always valid)

INVALID EXAMPLES (plausible content, but logically flawed):

1. "Most doctors are smart. John is smart. Therefore John is a doctor."
   → INVALID (undistributed middle fallacy - many smart people aren't doctors)

2. "All cats are mammals. All cats are animals. Therefore all mammals are animals."
   → INVALID (illicit major - "mammals" not distributed in premise but is in conclusion)

3. "Some birds can fly. Penguins are birds. Therefore penguins can fly."
   → INVALID (existential fallacy - "some" doesn't mean "all")

4. "No fish are mammals. Some fish live in water. Therefore some mammals don't live in water."
   → INVALID (doesn't follow logically, even though conclusion is true in reality)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAL RULES - Check each one
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DISTRIBUTION RULES:
- "All X are Y" → X distributed, Y undistributed
- "No X are Y" → Both X and Y distributed
- "Some X are Y" → Neither distributed
- "Some X are not Y" → Y distributed, X undistributed

VALIDITY RULES (violate ANY = INVALID):
1. Middle term must be distributed at least once
2. If term distributed in conclusion, must be distributed in premise
3. Cannot have two negative premises
4. If one premise negative, conclusion must be negative
5. Cannot have two particular premises ("some")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTIPLE VALIDATION STRATEGIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy 1 - COUNTEREXAMPLE TEST:
- Try to make premises true but conclusion false
- If possible → INVALID
- If impossible → VALID

Strategy 2 - VENN DIAGRAM:
- Draw three overlapping circles
- Mark what premises require
- Check if conclusion must be true in all valid arrangements

Strategy 3 - FORMAL RULES:
- Identify three terms and check distribution
- Apply all 5 validity rules above

Strategy 4 - STRUCTURAL PATTERN:
- Identify the syllogistic form (Barbara, Celarent, etc.)
- Known valid forms: Barbara (AAA-1), Celarent (EAE-1), Darii (AII-1), etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON CONTENT BIAS TRAPS - AVOID THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRAP 1: "This sounds right, so it must be valid"
→ NO! Plausible arguments can be invalid.

TRAP 2: "This is absurd, so it must be invalid"
→ NO! Absurd arguments can be perfectly valid.

TRAP 3: "The conclusion is true in reality"
→ IRRELEVANT! Valid arguments can have false conclusions.

TRAP 4: "Most/many means all"
→ NO! "Most A are B" ≠ "All A are B" in formal logic.

TRAP 5: "This is a good argument in real life"
→ IRRELEVANT! We're checking FORMAL validity, not persuasiveness.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Argument to evaluate:
{syllogism}

EVALUATE using multiple strategies:
1. Identify the three terms and check distribution
2. Test for counterexamples
3. Apply the 5 formal validity rules
4. Visualize with Venn diagrams if helpful

REMEMBER:
- Assume you know NOTHING about the real world
- Content is completely irrelevant
- Only logical structure matters
- Absurd can be valid, plausible can be invalid

Respond with ONLY: true or false"""

    return prompt


def fallacy_focus_prompt(syllogism: str) -> str:
    """
    Prompt focused on identifying common logical fallacies.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY by checking for common fallacies. Ignore real-world plausibility completely.

Argument:
{syllogism}

COMMON FALLACIES TO CHECK:

1. UNDISTRIBUTED MIDDLE
   - Middle term (appears only in premises) must be distributed at least once
   - INVALID: "All cats are animals. All dogs are animals. Therefore all cats are dogs."
   - The middle term "animals" is never distributed (always predicate of "All")

2. ILLICIT MAJOR
   - Major term (predicate of conclusion) distributed in conclusion but not in major premise
   - INVALID: "All A are B. Some B are C. Therefore all C are A."

3. ILLICIT MINOR
   - Minor term (subject of conclusion) distributed in conclusion but not in minor premise
   - INVALID: "All cats are mammals. All cats are animals. Therefore all mammals are animals."

4. AFFIRMING THE CONSEQUENT
   - INVALID: "If A then B. B is true. Therefore A is true."
   - Not a categorical syllogism but similar error

5. DENYING THE ANTECEDENT
   - INVALID: "If A then B. A is false. Therefore B is false."
   - Not a categorical syllogism but similar error

6. EXISTENTIAL FALLACY
   - Drawing particular conclusion from universal premises when existence is uncertain
   - INVALID: "All unicorns are magical. All unicorns have horns. Therefore some magical things have horns."

7. EXCLUSIVE PREMISES
   - Two negative premises cannot yield valid conclusion
   - INVALID: "No A are B. No B are C. Therefore [any conclusion]."

8. NEGATIVE CONCLUSION FROM AFFIRMATIVE PREMISES
   - If both premises positive, conclusion must be positive
   - INVALID: "All A are B. All B are C. Therefore some A are not C."

9. AFFIRMATIVE CONCLUSION FROM NEGATIVE PREMISE
   - If one premise negative, conclusion must be negative
   - INVALID: "No A are B. All C are A. Therefore all C are B."

10. QUANTIFIER SHIFT
    - "Some" does not mean "all" or "most"
    - INVALID: "Some birds fly. Penguins are birds. Therefore penguins fly."

EVALUATION PROCESS:
1. Identify the three terms
2. Check each fallacy above
3. If ANY fallacy present → INVALID
4. If NO fallacies → Likely VALID (verify with formal rules)

IGNORE: Real-world truth, plausibility, whether it makes sense
EVALUATE: Is there a logical fallacy in the structure?

Respond with ONLY: true or false"""

    return prompt


def reasoning_then_answer_prompt(syllogism: str) -> str:
    """
    Prompt that explicitly encourages reasoning before answering.

    Args:
        syllogism: The complete syllogism text

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: Evaluate FORMAL LOGICAL VALIDITY only. Ignore real-world plausibility completely.

Argument:
{syllogism}

INSTRUCTIONS:
1. Think through the logical structure step by step
2. Identify any fallacies or structural issues
3. Make your determination based purely on logical form, not content

Please reason through this problem, explaining your analysis.
After your reasoning, provide your final answer on the last line as either: true or false

KEY REMINDERS:
- Absurd content can be VALID if the structure is sound
- Plausible content can be INVALID if the structure is flawed
- Only the logical form matters, not real-world truth"""

    return prompt


def formal_abstract_prompt(syllogism: str) -> str:
    """
    Prompt specifically for normalized syllogisms using abstract variables (A, B, C).
    Emphasizes pure formal reasoning without any real-world associations.

    Args:
        syllogism: The normalized syllogism with abstract variables

    Returns:
        Formatted prompt string
    """
    prompt = f"""CRITICAL: You are evaluating a FORMAL LOGICAL ARGUMENT using ABSTRACT VARIABLES.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The input uses ABSTRACT VARIABLES (A, B, C, etc.) that have NO REAL-WORLD MEANING.
- A, B, C are PURELY SYMBOLIC placeholders
- They represent arbitrary sets with NO properties except their logical relationships
- Do NOT assign any concrete interpretation to these variables
- Do NOT apply any real-world knowledge or associations

This is FORMAL SYMBOLIC LOGIC - evaluate ONLY the abstract structure.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARGUMENT (Abstract Form)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{syllogism}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID = The conclusion follows by LOGICAL NECESSITY from the premises
INVALID = The conclusion does NOT necessarily follow

CRITICAL REMINDERS:
1. Treat A, B, C as completely abstract - no meaning beyond their logical relations
2. Apply ONLY formal logic rules - no heuristics, no intuition, no content
3. The variables could represent ANYTHING - focus on structure alone

DISTRIBUTION (for categorical syllogisms):
- "All X are Y" → X distributed, Y undistributed
- "No X are Y" → Both X and Y distributed
- "Some X are Y" → Neither X nor Y distributed
- "Some X are not Y" → Y distributed, X undistributed

VALIDITY RULES (if ANY violated → INVALID):
1. Middle term (in premises only) must be distributed at least once
2. If term distributed in conclusion, must be distributed in premise
3. Two negative premises cannot yield valid conclusion
4. If one premise negative, conclusion must be negative
5. Two particular ("some") premises cannot yield valid conclusion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IDENTIFY the three terms and their positions
2. CHECK distribution of each term in each premise
3. APPLY the 5 formal validity rules
4. VERIFY using counterexample test (can premises be true while conclusion false?)

THINK PURELY ABSTRACTLY:
- Imagine A, B, C as arbitrary sets
- No semantic content whatsoever
- Only set-theoretic relationships matter

Respond with ONLY: true or false"""

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
    "counterexample": counterexample_prompt,
    "venn_diagram": venn_diagram_prompt,
    "venn": venn_diagram_prompt,
    "syllogistic_rules": syllogistic_rules_prompt,
    "rules": syllogistic_rules_prompt,
    "minimal": minimal_prompt,
    "symbolic_logic": symbolic_logic_prompt,
    "symbolic": symbolic_logic_prompt,
    "comprehensive": comprehensive_prompt,
    "detailed": comprehensive_prompt,
    "fallacy_focus": fallacy_focus_prompt,
    "fallacy": fallacy_focus_prompt,
    "reasoning_then_answer": reasoning_then_answer_prompt,
    "reason": reasoning_then_answer_prompt,
    "formal_abstract": formal_abstract_prompt,
    "formal": formal_abstract_prompt,
    "abstract": formal_abstract_prompt,
    "normalized": formal_abstract_prompt,
    "normalization": normalization_replace_prompt,
    "normalize": normalization_replace_prompt,
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
