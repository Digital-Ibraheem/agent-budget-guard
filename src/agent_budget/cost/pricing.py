"""Pricing configuration loader and manager."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..exceptions import PricingDataError


class PricingTable:
    """Manages OpenAI model pricing data.

    Loads pricing information from pricing.json and provides lookup methods
    for model pricing, encodings, and metadata.

    Attributes:
        _data: Raw pricing configuration data
        _models: Model pricing dictionary
        _aliases: Model alias mappings
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize PricingTable.

        Args:
            config_path: Optional path to pricing JSON file. If None, uses default
                        pricing.json from the package's config directory.

        Raises:
            PricingDataError: If pricing file cannot be loaded or is malformed
        """
        if config_path is None:
            # Use default pricing.json from package
            config_path = Path(__file__).parent.parent / "config" / "pricing.json"
        else:
            config_path = Path(config_path)

        try:
            with open(config_path, "r") as f:
                self._data: Dict[str, Any] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise PricingDataError(f"Failed to load pricing configuration: {e}") from e

        self._models: Dict[str, Dict[str, Any]] = self._data.get("models", {})
        self._aliases: Dict[str, str] = self._data.get("model_aliases", {})

        if not self._models:
            raise PricingDataError("Pricing configuration contains no models")

    def _resolve_model(self, model: str) -> str:
        """Resolve model alias to canonical model name.

        Args:
            model: Model name or alias

        Returns:
            Canonical model name

        Raises:
            PricingDataError: If model not found in pricing data
        """
        # Check if it's an alias first
        if model in self._aliases:
            return self._aliases[model]

        # Check if it's a known model
        if model in self._models:
            return model

        # Try to match versioned models like gpt-4-0314 or gpt-4o-mini-2024-07-18 to their base model.
        # E.g., gpt-4o-mini-2024-07-18 -> gpt-4o-mini
        parts = model.split("-")
        # Try stripping numeric 'version/date' suffixes step by step
        for i in range(len(parts), 0, -1):
            candidate = "-".join(parts[:i])
            if candidate in self._models:
                return candidate

        raise PricingDataError(
            f"Model '{model}' not found in pricing configuration. "
            f"Available models: {', '.join(sorted(self._models.keys()))}"
        )

    def get_input_price(self, model: str, tier: str = "standard", cached: bool = False) -> float:
        """Get input token price for a model in USD per 1,000 tokens.

        Args:
            model: Model name (can be an alias)
            tier: Pricing tier ("standard" or "batch")
            cached: Whether to use cached input pricing (if available)

        Returns:
            Price per 1,000 input tokens in USD

        Raises:
            PricingDataError: If model not found or price missing
        """
        canonical_model = self._resolve_model(model)
        model_data = self._models[canonical_model]

        # Get pricing for the specified tier (default to standard)
        if tier not in model_data:
            tier = "standard"

        tier_data = model_data.get(tier, {})

        # Try to get cached input price if requested and available
        if cached and "cached_input_price_per_1k" in tier_data:
            return float(tier_data["cached_input_price_per_1k"])

        # Otherwise get regular input price
        if "input_price_per_1k" not in tier_data:
            raise PricingDataError(
                f"Input price not found for model '{canonical_model}' tier '{tier}'"
            )

        return float(tier_data["input_price_per_1k"])

    def get_output_price(self, model: str, tier: str = "standard") -> float:
        """Get output token price for a model in USD per 1,000 tokens.

        Args:
            model: Model name (can be an alias)
            tier: Pricing tier ("standard" or "batch")

        Returns:
            Price per 1,000 output tokens in USD

        Raises:
            PricingDataError: If model not found or price missing
        """
        canonical_model = self._resolve_model(model)
        model_data = self._models[canonical_model]

        # Get pricing for the specified tier (default to standard)
        if tier not in model_data:
            tier = "standard"

        tier_data = model_data.get(tier, {})

        if "output_price_per_1k" not in tier_data:
            raise PricingDataError(
                f"Output price not found for model '{canonical_model}' tier '{tier}'"
            )

        return float(tier_data["output_price_per_1k"])

    def get_model_encoding(self, model: str) -> str:
        """Get tiktoken encoding name for a model.

        Args:
            model: Model name (can be an alias)

        Returns:
            Encoding name (e.g., "o200k_base", "cl100k_base")
        """
        canonical_model = self._resolve_model(model)

        # Infer encoding based on model name
        # GPT-5, GPT-4.1, GPT-4o, and o-series use o200k_base
        # Older GPT-4 and GPT-3.5 use cl100k_base
        if canonical_model.startswith(("gpt-5", "gpt-4.1", "gpt-4o", "o1", "o3", "o4")):
            return "o200k_base"
        elif canonical_model.startswith(("gpt-4", "gpt-3.5")):
            return "cl100k_base"
        else:
            # Default to o200k_base for unknown models
            return "o200k_base"

    def is_reasoning_model(self, model: str) -> bool:
        """Check if a model is an o-series reasoning model.

        O-series models (o1, o3, o4-mini) use hidden reasoning tokens that
        significantly increase costs beyond visible output.

        Args:
            model: Model name (can be an alias)

        Returns:
            True if model is a reasoning model (o-series)
        """
        try:
            canonical_model = self._resolve_model(model)
            # O-series models start with "o" followed by a digit
            return canonical_model.startswith(("o1", "o3", "o4"))
        except PricingDataError:
            # If model not found, assume it's not a reasoning model
            return False

    def get_max_tokens(self, model: str) -> int:
        """Get maximum output tokens for a model.

        Args:
            model: Model name (can be an alias)

        Returns:
            Maximum number of output tokens

        Raises:
            PricingDataError: If model not found
        """
        canonical_model = self._resolve_model(model)
        model_data = self._models[canonical_model]

        # Try max_output_tokens first, fallback to max_tokens or default
        return int(model_data.get("max_output_tokens", model_data.get("max_tokens", 4096)))

    def get_context_window(self, model: str) -> int:
        """Get context window size for a model.

        Args:
            model: Model name (can be an alias)

        Returns:
            Context window size in tokens

        Raises:
            PricingDataError: If model not found
        """
        canonical_model = self._resolve_model(model)
        model_data = self._models[canonical_model]

        return int(model_data.get("context_window", 128000))
