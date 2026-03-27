"""Performance benchmarks for HenryCLI."""

import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from henrycli.auto_tier import AutoTierClassifier
from henrycli.context.filestore import FileStore
from henrycli.context.manager import ContextManager
from henrycli.downloader import DocumentDownloader
from henrycli.lmstudio import LMStudioClient, ChatMessage
from henrycli.models.config import ModelConfig
from henrycli.models.pool import ModelPool
from henrycli.plugins import PluginManager


# Sample data for benchmarks
SAMPLE_MODEL_NAMES = [
    "TheBloke/phi-3-mini-4k-instruct-GGUF",
    "TheBloke/qwen2.5-7b-instruct-GGUF",
    "TheBloke/codellama-13b-instruct-GGUF",
    "TheBloke/yi-34b-chat-GGUF",
    "bartowski/Llama-3.1-8B-Instruct-GGUF",
    "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
    "Qwen/Qwen2.5-72B-Instruct-GGUF",
    "unsloth/gemma-2-9b-it-GGUF",
]

SAMPLE_LARGE_TEXT = "Hello, World! " * 10000  # ~130KB


class TestAutoTierBenchmarks:
    """Benchmarks for auto-tier classification."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return AutoTierClassifier(hardware_vram_gb=8.0)

    def test_benchmark_extract_params(self, benchmark, classifier):
        """Benchmark parameter extraction from model names."""
        def extract_all():
            for model_name in SAMPLE_MODEL_NAMES:
                classifier._extract_params(model_name.lower())
        benchmark(extract_all)

    def test_benchmark_analyze_model(self, benchmark, classifier):
        """Benchmark single model analysis."""
        model_name = "TheBloke/phi-3-mini-4k-instruct-GGUF"
        benchmark(classifier.analyze_model, model_name)

    def test_benchmark_classify_local_models(self, benchmark, classifier):
        """Benchmark classifying multiple models."""
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES * 10]
        benchmark(classifier.classify_local_models, models)

    def test_benchmark_generate_tier_config(self, benchmark, classifier):
        """Benchmark generating tier configuration."""
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES * 10]
        benchmark(classifier.generate_tier_config, models)


class TestFileStoreBenchmarks:
    """Benchmarks for filestore operations."""

    @pytest.fixture
    def temp_filestore(self, tmp_path):
        """Create temporary filestore."""
        return FileStore(base_dir=tmp_path)

    def test_benchmark_offload_small(self, benchmark, temp_filestore):
        """Benchmark offloading small content."""
        content = "Small content" * 100
        benchmark(temp_filestore.offload, content)

    def test_benchmark_offload_large(self, benchmark, temp_filestore):
        """Benchmark offloading large content."""
        content = SAMPLE_LARGE_TEXT
        benchmark(temp_filestore.offload, content)

    def test_benchmark_load_content(self, benchmark, temp_filestore):
        """Benchmark loading content."""
        ref = temp_filestore.offload(SAMPLE_LARGE_TEXT)
        benchmark(temp_filestore.load, ref)

    def test_benchmark_load_preview(self, benchmark, temp_filestore):
        """Benchmark loading preview."""
        ref = temp_filestore.offload("\n".join([f"Line {i}" for i in range(1000)]))
        benchmark(temp_filestore.load_preview, ref, 10)


class TestContextManagerBenchmarks:
    """Benchmarks for context manager operations."""

    @pytest.fixture
    def temp_context_manager(self, tmp_path):
        """Create temporary context manager."""
        return ContextManager(base_dir=tmp_path)

    def test_benchmark_create_context(self, benchmark, temp_context_manager):
        """Benchmark creating contexts."""
        benchmark(
            temp_context_manager.create_context,
            "test-agent",
            "test-model",
            "Test task",
        )

    def test_benchmark_add_messages(self, benchmark, temp_context_manager):
        """Benchmark adding messages."""
        temp_context_manager.create_context(
            agent_id="test",
            model="test-model",
            task="Test task",
        )
        benchmark(temp_context_manager.add_message, "user", "Test message")

    def test_benchmark_context_usage_ratio(self, benchmark, temp_context_manager):
        """Benchmark context usage calculation."""
        temp_context_manager.create_context(
            agent_id="test",
            model="test-model",
            task="Test task",
        )
        benchmark(temp_context_manager.get_context_usage_ratio, 16000, 32768)


class TestPluginManagerBenchmarks:
    """Benchmarks for plugin manager operations."""

    @pytest.fixture
    def plugin_manager(self):
        """Create plugin manager."""
        return PluginManager()

    def test_benchmark_enable_tool(self, benchmark, plugin_manager):
        """Benchmark enabling tools."""
        benchmark(plugin_manager.enable_tool, "duckduckgo")

    def test_benchmark_get_tool_definitions(self, benchmark, plugin_manager):
        """Benchmark getting tool definitions."""
        benchmark(plugin_manager.get_tool_definitions)

    def test_benchmark_list_plugins(self, benchmark, plugin_manager):
        """Benchmark listing plugins."""
        benchmark(plugin_manager.list_plugins)


class TestModelConfigBenchmarks:
    """Benchmarks for model configuration."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config."""
        config_path = tmp_path / "config.yaml"
        return ModelConfig(config_path=config_path)

    def test_benchmark_get_tier(self, benchmark, temp_config):
        """Benchmark getting tier configuration."""
        benchmark(temp_config.get_tier, "T2")

    def test_benchmark_get_default_model(self, benchmark, temp_config):
        """Benchmark getting default model."""
        benchmark(temp_config.get_default_model, "T1")

    def test_benchmark_get_model_vram(self, benchmark, temp_config):
        """Benchmark getting model VRAM."""
        model = "phi-3-mini-4k-instruct-q4_k_m"
        benchmark(temp_config.get_model_vram, model)


class TestDownloaderBenchmarks:
    """Benchmarks for document downloader."""

    @pytest.fixture
    def temp_downloader(self, tmp_path):
        """Create temporary downloader."""
        return DocumentDownloader(rag_directory=tmp_path)

    def test_benchmark_generate_filename(self, benchmark, temp_downloader):
        """Benchmark filename generation."""
        url = "https://example.com/file.pdf"
        benchmark(temp_downloader._generate_filename, url)

    def test_benchmark_convert_github_url(self, benchmark, temp_downloader):
        """Benchmark GitHub URL conversion."""
        url = "https://github.com/user/repo/blob/main/file.py"
        benchmark(temp_downloader._convert_github_url, url)

    def test_benchmark_extract_arxiv_id(self, benchmark, temp_downloader):
        """Benchmark arXiv ID extraction."""
        url = "https://arxiv.org/abs/2301.12345"
        benchmark(temp_downloader._extract_arxiv_id, url)


# Performance threshold tests
class TestPerformanceThresholds:
    """Test that operations meet performance thresholds."""

    def test_extract_params_performance(self):
        """Test parameter extraction meets threshold (<10ms per model)."""
        classifier = AutoTierClassifier()
        start = time.perf_counter()
        for model_name in SAMPLE_MODEL_NAMES:
            classifier._extract_params(model_name.lower())
        duration = time.perf_counter() - start

        # Should complete in under 10ms per model
        assert duration < len(SAMPLE_MODEL_NAMES) * 0.01

    def test_filestore_offload_performance(self, tmp_path):
        """Test filestore offload meets threshold (<1ms per file)."""
        filestore = FileStore(base_dir=tmp_path)
        content = "Test content" * 100

        start = time.perf_counter()
        for _ in range(100):
            filestore.offload(content)
        duration = time.perf_counter() - start

        # Should complete in under 100ms
        assert duration < 0.1

    def test_classifier_batch_performance(self):
        """Test batch classification meets threshold (<50ms for 80 models)."""
        classifier = AutoTierClassifier()
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES * 10]

        start = time.perf_counter()
        classifier.classify_local_models(models)
        duration = time.perf_counter() - start

        # Should complete in under 50ms
        assert duration < 0.05
