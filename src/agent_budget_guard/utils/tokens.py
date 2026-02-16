"""Token counting utilities using tiktoken."""

from typing import List, Dict, Any, Optional
import tiktoken


def count_message_tokens(messages: List[Dict[str, Any]], encoding_name: str) -> int:
    """Count tokens in a list of messages including formatting overhead.

    This function accounts for the message formatting tokens that OpenAI
    adds when processing chat completions (role labels, separators, etc.).

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        encoding_name: Name of the tiktoken encoding (e.g., "o200k_base", "cl100k_base")

    Returns:
        Total number of input tokens including formatting overhead

    Raises:
        ValueError: If encoding name is invalid
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError as e:
        raise ValueError(f"Invalid encoding name: {encoding_name}") from e

    # Token overhead per message varies by model, but 3 is a safe estimate
    # This accounts for <|start|>role/content<|end|> formatting
    tokens_per_message = 3

    # Additional token if 'name' field is present
    tokens_per_name = 1

    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message

        for key, value in message.items():
            # Convert value to string and encode
            num_tokens += len(encoding.encode(str(value)))

            if key == "name":
                num_tokens += tokens_per_name

    # Every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3

    return num_tokens


def estimate_completion_tokens(
    max_tokens: Optional[int],
    input_tokens: int,
    model: str,
    is_reasoning_model: bool = False
) -> int:
    """Estimate the number of completion (output) tokens.

    Args:
        max_tokens: Maximum tokens specified by user (if any)
        input_tokens: Number of input tokens
        model: Model name (for default max_tokens lookup)
        is_reasoning_model: Whether this is an o-series reasoning model

    Returns:
        Conservative estimate of completion tokens

    Note:
        For o-series models, this applies a 3-5x safety margin due to
        hidden reasoning tokens that are billed but not returned.
    """
    if max_tokens is not None:
        estimated = max_tokens
    else:
        # Conservative estimate: 150% of input tokens
        # Capped at reasonable defaults based on model
        model_defaults = {
            "gpt-5.2": 4096,
            "gpt-5.1": 4096,
            "gpt-5": 4096,
            "gpt-5-mini": 4096,
            "gpt-5-nano": 4096,
            "gpt-4.1": 4096,
            "gpt-4o": 4096,
            "gpt-4o-mini": 4096,
            "o1": 4096,
            "o3": 4096,
            "o4-mini": 4096,
            "gpt-4-turbo": 4096,
            "gpt-4": 4096,
            "gpt-3.5-turbo": 4096,
        }

        # Get base model name (without version suffixes)
        base_model = model.split("-")[0] + "-" + model.split("-")[1] if "-" in model else model

        # Find matching default
        default_max = 4096
        for model_prefix, max_val in model_defaults.items():
            if model.startswith(model_prefix):
                default_max = max_val
                break

        estimated = min(int(input_tokens * 1.5), default_max)

    # CRITICAL: O-series models use hidden reasoning tokens
    # Apply safety margin of 4x to account for internal reasoning
    if is_reasoning_model:
        estimated = int(estimated * 4)

    return estimated


def count_string_tokens(text: str, encoding_name: str) -> int:
    """Count tokens in a plain text string.

    Args:
        text: Text to count tokens for
        encoding_name: Name of the tiktoken encoding

    Returns:
        Number of tokens in the text

    Raises:
        ValueError: If encoding name is invalid
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError as e:
        raise ValueError(f"Invalid encoding name: {encoding_name}") from e

    return len(encoding.encode(text))
