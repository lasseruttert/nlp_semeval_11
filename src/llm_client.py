"""
OpenRouter API client for LLM interactions.
"""
import os
import json
import re
from typing import Optional, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


def clean_json_response(content: str) -> str:
    """
    Clean markdown code blocks and fix common JSON issues from LLM response.

    Args:
        content: Raw LLM response that may contain markdown formatting

    Returns:
        Cleaned JSON string
    """
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    content = content.strip()

    # Pattern to match code blocks with optional language specifier
    pattern = r'^```(?:json)?\s*\n(.*?)\n```$'
    match = re.match(pattern, content, re.DOTALL)

    if match:
        content = match.group(1).strip()

    # Try to parse as-is first
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass

    # Fix control characters inside string values only
    # Scan through the JSON and escape control chars when inside quotes
    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(content):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == '\\':
            result.append(char)
            escape_next = True
            continue

        if char == '"':
            result.append(char)
            in_string = not in_string
            continue

        if in_string:
            # Escape control characters inside strings
            if char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif ord(char) < 32:  # Other control characters
                result.append(f'\\u{ord(char):04x}')
            else:
                result.append(char)
        else:
            result.append(char)

    fixed_content = ''.join(result)

    # Remove trailing commas before closing braces/brackets
    fixed_content = re.sub(r',\s*}', '}', fixed_content)
    fixed_content = re.sub(r',\s*]', ']', fixed_content)

    # Try to parse the fixed content
    try:
        json.loads(fixed_content)
        return fixed_content
    except json.JSONDecodeError:
        # Return the best effort fix
        return fixed_content


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API.
    Uses OpenAI SDK with OpenRouter base URL.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet"):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (if None, reads from OPENROUTER_API_KEY env var)
            model: Model identifier to use (default: Claude 3.5 Sonnet)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: float = 0.0,
        max_tokens: int = 5000,
    ) -> T:
        """
        Generate a structured response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model class for the expected response
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            Instance of response_model with the LLM's response
        """
        # Simple system message for logical reasoning
        system_message = (
            "You are a logical reasoning expert. "
            "Respond with ONLY the word 'true' or 'false', nothing else."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract the response content
            content = response.choices[0].message.content

            # Check for empty response
            if not content or content.strip() == "":
                finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
                raise ValueError(
                    f"Empty response from LLM.\n"
                    f"Finish reason: {finish_reason}\n"
                    f"Prompt length: {len(prompt)} chars\n"
                    f"This could indicate:\n"
                    f"- Content filtering (check finish_reason='content_filter')\n"
                    f"- Max tokens too low (check finish_reason='length')\n"
                    f"- API rate limiting or errors\n"
                    f"- Prompt too long for model context window"
                )

            # Extract true/false from the response
            # Look for true or false (case insensitive) in the response
            content_lower = content.strip().lower()

            # Try exact match first
            if content_lower == 'true':
                validity_value = True
            elif content_lower == 'false':
                validity_value = False
            else:
                # Search for true/false anywhere in the response
                import re
                # Look for standalone true/false (word boundaries)
                true_match = re.search(r'\btrue\b', content_lower)
                false_match = re.search(r'\bfalse\b', content_lower)

                if true_match and not false_match:
                    validity_value = True
                elif false_match and not true_match:
                    validity_value = False
                elif true_match and false_match:
                    # Both found - use the first one
                    validity_value = true_match.start() < false_match.start()
                else:
                    # No true/false found, try to parse as JSON as fallback
                    try:
                        cleaned_content = clean_json_response(content)
                        response_data = json.loads(cleaned_content)

                        # Try to find validity field
                        def find_validity(obj):
                            if isinstance(obj, dict):
                                if 'validity' in obj:
                                    return obj['validity']
                                for value in obj.values():
                                    result = find_validity(value)
                                    if result is not None:
                                        return result
                            return None

                        validity_value = find_validity(response_data)
                        if validity_value is None:
                            raise ValueError(f"Could not find true/false or validity field in response: {content}")
                    except:
                        raise ValueError(f"Could not extract true/false from response: {content}")

            return response_model(validity=validity_value)

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API: {e}")

    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate a simple text response from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            The LLM's text response
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error calling OpenRouter API: {e}")
