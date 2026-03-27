"""Tests for auto-tier classification."""

import pytest

from henrycli.auto_tier import AutoTier, AutoTierClassifier, ModelAnalysis


class TestAutoTierClassifier:
    """Tests for AutoTierClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier with default hardware."""
        return AutoTierClassifier(hardware_vram_gb=8.0)

    def test_extract_params_7b(self, classifier):
        """Test extracting 7B parameter count."""
        params = classifier._extract_params("TheBloke/phi-3-mini-7b-instruct-GGUF")
        assert params == 7.0

    def test_extract_params_13b(self, classifier):
        """Test extracting 13B parameter count."""
        params = classifier._extract_params("TheBloke/codellama-13b-instruct-GGUF")
        assert params == 13.0

    def test_extract_params_34b(self, classifier):
        """Test extracting 34B parameter count."""
        params = classifier._extract_params("TheBloke/yi-34b-chat-GGUF")
        assert params == 34.0

    def test_extract_params_phi3_mini(self, classifier):
        """Test extracting Phi-3-mini specific params."""
        params = classifier._extract_params("TheBloke/phi-3-mini-4k-instruct-GGUF")
        assert params == 3.8

    def test_extract_params_qwen(self, classifier):
        """Test extracting Qwen parameter counts."""
        # Pattern matching is case-insensitive in model names
        params = classifier._extract_params("qwen/qwen2.5-7b-instruct-gguf")
        assert params == 7.0

    def test_extract_params_unknown(self, classifier):
        """Test extracting from unknown model name."""
        params = classifier._extract_params("unknown-model-name")
        assert params == 0.0

    def test_params_to_tier_t1(self, classifier):
        """Test tier assignment for small models."""
        assert classifier._params_to_tier(3.0) == AutoTier.T1
        assert classifier._params_to_tier(4.0) == AutoTier.T1

    def test_params_to_tier_t2(self, classifier):
        """Test tier assignment for medium models."""
        assert classifier._params_to_tier(7.0) == AutoTier.T2
        assert classifier._params_to_tier(9.0) == AutoTier.T2

    def test_params_to_tier_t3(self, classifier):
        """Test tier assignment for large models."""
        assert classifier._params_to_tier(13.0) == AutoTier.T3
        assert classifier._params_to_tier(15.0) == AutoTier.T3

    def test_params_to_tier_t4(self, classifier):
        """Test tier assignment for very large models."""
        assert classifier._params_to_tier(32.0) == AutoTier.T4
        assert classifier._params_to_tier(70.0) == AutoTier.T4

    def test_analyze_model_7b(self, classifier):
        """Test analyzing a 7B model."""
        analysis = classifier.analyze_model("TheBloke/phi-3-7b-instruct-GGUF")

        assert analysis.model_key == "TheBloke/phi-3-7b-instruct-GGUF"
        assert analysis.estimated_params_b == 7.0
        assert analysis.tier == AutoTier.T2
        assert analysis.estimated_vram_q4 > 0
        assert analysis.confidence == "high"

    def test_analyze_model_unknown(self, classifier):
        """Test analyzing unknown model."""
        analysis = classifier.analyze_model("unknown-model")

        assert analysis.model_key == "unknown-model"
        assert analysis.estimated_params_b == 7.0  # Default
        assert analysis.confidence == "low"

    def test_classify_local_models(self, classifier):
        """Test classifying multiple models."""
        models = [
            {"model_key": "TheBloke/phi-3-mini-4k-instruct-GGUF"},
            {"model_key": "TheBloke/codellama-13b-instruct-GGUF"},
            {"model_key": "TheBloke/yi-34b-chat-GGUF"},
        ]

        analyses = classifier.classify_local_models(models)

        assert len(analyses) == 3
        assert analyses[0].tier == AutoTier.T1  # Phi-3-mini
        assert analyses[1].tier == AutoTier.T3  # CodeLlama-13B
        assert analyses[2].tier == AutoTier.T4  # Yi-34B

    def test_get_models_for_tier(self, classifier):
        """Test filtering models by tier."""
        models = [
            {"model_key": "TheBloke/phi-3-mini-4k-instruct-GGUF"},
            {"model_key": "TheBloke/qwen2.5-7b-instruct-GGUF"},
            {"model_key": "TheBloke/codellama-13b-instruct-GGUF"},
        ]

        t2_models = classifier.get_models_for_tier(models, AutoTier.T2)

        assert len(t2_models) == 1
        assert "qwen2.5-7b" in t2_models[0].model_key.lower()

    def test_generate_tier_config(self, classifier):
        """Test generating tier configuration."""
        models = [
            {"model_key": "TheBloke/phi-3-mini-4k-instruct-GGUF"},
            {"model_key": "TheBloke/qwen2.5-7b-instruct-GGUF"},
        ]

        config = classifier.generate_tier_config(models)

        assert "T1" in config
        assert "T2" in config
        assert "T3" in config
        assert "T4" in config

        # Phi-3-mini should be T1
        assert len(config["T1"]) >= 1

    def test_moe_model_detection(self, classifier):
        """Test MoE model VRAM adjustment."""
        analysis = classifier.analyze_model("TheBloke/mixtral-8x7b-instruct-GGUF")

        # MoE models should have reduced VRAM estimate
        assert "MoE" in analysis.reasoning
        # 46.7B params * 0.7 * 0.6 (MoE adjustment) ≈ 19.6 GB
        assert analysis.estimated_vram_q4 < 46.7 * 0.7
