"""Experimental auto-tier classification for models."""

import re
from dataclasses import dataclass
from enum import Enum


class AutoTier(Enum):
    """Automatically determined tier."""

    T1 = "T1"  # < 5B parameters
    T2 = "T2"  # 5B - 10B parameters
    T3 = "T3"  # 10B - 20B parameters
    T4 = "T4"  # 20B+ parameters
    UNKNOWN = "UNKNOWN"


@dataclass
class ModelAnalysis:
    """Analysis result for a model."""

    model_key: str
    estimated_params_b: float  # Parameters in billions
    tier: AutoTier
    estimated_vram_q4: float  # VRAM in GB for Q4_K_M
    estimated_vram_q8: float  # VRAM in GB for Q8_0
    confidence: str  # "high", "medium", "low"
    reasoning: str


class AutoTierClassifier:
    """
    Experimental auto-tier classification for LM Studio models.
    
    Analyzes model names to estimate:
    - Parameter count
    - VRAM requirements
    - Appropriate tier assignment
    
    Supports naming conventions from:
    - TheBloke
    - Bartowski
    - MaziyarPanahi
    - Unsloth
    - And other common quantizers
    """

    # Parameter size patterns in model names
    PARAM_PATTERNS = [
        (r"(\d+)b(?!\d)", 1.0),  # "7b" -> 7B
        (r"(\d+)-b(?!\d)", 1.0),  # "7-b" -> 7B
        (r"(\d+)billion(?!\d)", 1.0),  # "7billion" -> 7B
        (r"(\d+)m(?!b|\d)", 0.001),  # "500m" -> 0.5B
        (r"phi-3-mini", 3.8),  # Phi-3-mini specific
        (r"phi-3-small", 7.0),  # Phi-3-small specific
        (r"phi-3-medium", 14.0),  # Phi-3-medium specific
        (r"qwen2\.5-0\.5b", 0.5),
        (r"qwen2\.5-1\.5b", 1.5),
        (r"qwen2\.5-3b", 3.0),
        (r"qwen2\.5-7b", 7.0),
        (r"qwen2\.5-14b", 14.0),
        (r"qwen2\.5-32b", 32.0),
        (r"qwen2\.5-72b", 72.0),
        (r"llama-3\.2-1b", 1.0),
        (r"llama-3\.2-3b", 3.0),
        (r"llama-3-8b", 8.0),
        (r"llama-3-70b", 70.0),
        (r"llama-3\.1-8b", 8.0),
        (r"llama-3\.1-70b", 70.0),
        (r"llama-3\.1-405b", 405.0),
        (r"mistral-7b", 7.0),
        (r"mixtral-8x7b", 46.7),  # MoE: 8 experts of 7B
        (r"mixtral-8x22b", 141.0),  # MoE: 8 experts of 22B
        (r"gemma-2b", 2.0),
        (r"gemma-7b", 7.0),
        (r"gemma-2-2b", 2.0),
        (r"gemma-2-9b", 9.0),
        (r"gemma-2-27b", 27.0),
        (r"yi-6b", 6.0),
        (r"yi-9b", 9.0),
        (r"yi-34b", 34.0),
        (r"command-r", 35.0),
        (r"command-r-plus", 104.0),
        (r"deepseek-coder-1\.3b", 1.3),
        (r"deepseek-coder-6\.7b", 6.7),
        (r"deepseek-coder-33b", 33.0),
        (r"deepseek-v2", 236.0),  # MoE
        (r"deepseek-v3", 671.0),  # MoE
        (r"starcoder2-3b", 3.0),
        (r"starcoder2-7b", 7.0),
        (r"starcoder2-15b", 15.0),
        (r"codellama-7b", 7.0),
        (r"codellama-13b", 13.0),
        (r"codellama-34b", 34.0),
        (r"codellama-70b", 70.0),
    ]

    def __init__(self, hardware_vram_gb: float = 8.0):
        """
        Initialize auto-tier classifier.

        Args:
            hardware_vram_gb: Available VRAM in GB
        """
        self.hardware_vram_gb = hardware_vram_gb

    def analyze_model(self, model_key: str) -> ModelAnalysis:
        """
        Analyze a model and determine its tier.

        Args:
            model_key: Model identifier

        Returns:
            ModelAnalysis with tier and VRAM estimates
        """
        model_lower = model_key.lower()

        # Skip embedding models - they're not for chat/completion
        if "embedding" in model_lower or "embed-" in model_lower:
            return ModelAnalysis(
                model_key=model_key,
                estimated_params_b=0,
                tier=AutoTier.T4,  # Put in T4 so they're not used for chat
                estimated_vram_q4=0,
                estimated_vram_q8=0,
                confidence="high",
                reasoning="Embedding model - not for chat/completion",
            )

        # Try to extract parameter count
        params_b = self._extract_params(model_lower)
        confidence = "high" if params_b > 0 else "low"
        reasoning = ""

        if params_b <= 0:
            # Couldn't determine from name
            params_b = 7.0  # Default assumption
            confidence = "low"
            reasoning = "Parameter count not detected, assuming 7B"
        else:
            reasoning = f"Detected ~{params_b}B parameters from model name"

        # Determine tier
        tier = self._params_to_tier(params_b)

        # Estimate VRAM (Q4_K_M is ~0.7 bytes per param, Q8 is ~1.0)
        vram_q4 = params_b * 0.7
        vram_q8 = params_b * 1.0

        # Adjust for MoE models (active params are lower)
        if "mixtral" in model_lower or "mo" in model_lower:
            vram_q4 *= 0.6  # MoE models use fewer active params
            reasoning += "; MoE model detected"

        return ModelAnalysis(
            model_key=model_key,
            estimated_params_b=params_b,
            tier=tier,
            estimated_vram_q4=round(vram_q4, 2),
            estimated_vram_q8=round(vram_q8, 2),
            confidence=confidence,
            reasoning=reasoning,
        )

    def _extract_params(self, model_key: str) -> float:
        """
        Extract parameter count from model name.

        Args:
            model_key: Model identifier (lowercase)

        Returns:
            Parameter count in billions, or 0 if not found
        """
        for pattern, multiplier in self.PARAM_PATTERNS:
            match = re.search(pattern, model_key)
            if match:
                if multiplier == 1.0:
                    return float(match.group(1))
                else:
                    return multiplier

        return 0.0

    def _params_to_tier(self, params_b: float) -> AutoTier:
        """
        Convert parameter count to tier.

        Args:
            params_b: Parameters in billions

        Returns:
            AutoTier value
        """
        if params_b < 5:
            return AutoTier.T1
        elif params_b < 10:
            return AutoTier.T2
        elif params_b < 20:
            return AutoTier.T3
        else:
            return AutoTier.T4

    def classify_local_models(
        self,
        models: list[dict[str, Any]],
    ) -> list[ModelAnalysis]:
        """
        Classify a list of local models.

        Args:
            models: List of model dicts from LM Studio API or CLI

        Returns:
            List of ModelAnalysis results
        """
        results = []
        for model in models:
            # Handle both camelCase (lms CLI JSON) and snake_case (REST API)
            model_key = model.get(
                "modelKey",  # camelCase from lms ls --json
                model.get("model_key", model.get("name", ""))
            )
            analysis = self.analyze_model(model_key)
            results.append(analysis)
        return results

    def get_models_for_tier(
        self,
        models: list[dict[str, Any]],
        target_tier: AutoTier,
    ) -> list[ModelAnalysis]:
        """
        Filter models by tier.

        Args:
            models: List of model dicts
            target_tier: Target tier

        Returns:
            List of matching ModelAnalysis results
        """
        analyses = self.classify_local_models(models)
        return [a for a in analyses if a.tier == target_tier]

    def get_best_model_for_tier(
        self,
        models: list[dict[str, Any]],
        target_tier: AutoTier,
    ) -> ModelAnalysis | None:
        """
        Get the best model for a tier (largest that fits in VRAM).

        Args:
            models: List of model dicts
            target_tier: Target tier

        Returns:
            Best ModelAnalysis or None
        """
        tier_models = self.get_models_for_tier(models, target_tier)

        if not tier_models:
            return None

        # Sort by params (descending) and filter by VRAM
        tier_models.sort(key=lambda x: x.estimated_params_b, reverse=True)

        for model in tier_models:
            if model.estimated_vram_q4 <= self.hardware_vram_gb:
                return model

        # If none fit, return smallest
        return tier_models[-1] if tier_models else None

    def generate_tier_config(
        self,
        models: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate tier configuration from local models.

        Args:
            models: List of model dicts

        Returns:
            Tier configuration dict
        """
        analyses = self.classify_local_models(models)

        config = {"T1": [], "T2": [], "T3": [], "T4": []}

        for analysis in analyses:
            tier = analysis.tier.value
            if tier in config:
                config[tier].append({
                    "model_key": analysis.model_key,
                    "params_b": analysis.estimated_params_b,
                    "vram_q4": analysis.estimated_vram_q4,
                    "confidence": analysis.confidence,
                })

        return config
