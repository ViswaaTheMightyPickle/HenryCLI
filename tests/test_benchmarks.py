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

    def benchmark_extract_params(self, classifier):
        """Benchmark parameter extraction from model names."""
        for model_name in SAMPLE_MODEL_NAMES:
            classifier._extract_params(model_name.lower())

    def benchmark_analyze_model(self, classifier):
        """Benchmark single model analysis."""
        for model_name in SAMPLE_MODEL_NAMES:
            classifier.analyze_model(model_name)

    def benchmark_classify_local_models(self, classifier):
        """Benchmark classifying multiple models."""
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES * 10]
        classifier.classify_local_models(models)

    def benchmark_generate_tier_config(self, classifier):
        """Benchmark generating tier configuration."""
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES * 10]
        classifier.generate_tier_config(models)


class TestFileStoreBenchmarks:
    """Benchmarks for filestore operations."""

    @pytest.fixture
    def temp_filestore(self, tmp_path):
        """Create temporary filestore."""
        return FileStore(base_dir=tmp_path)

    def benchmark_offload_small(self, temp_filestore):
        """Benchmark offloading small content."""
        content = "Small content" * 100
        for _ in range(100):
            temp_filestore.offload(content)

    def benchmark_offload_large(self, temp_filestore):
        """Benchmark offloading large content."""
        content = SAMPLE_LARGE_TEXT
        for _ in range(10):
            temp_filestore.offload(content)

    def benchmark_offload_json(self, temp_filestore):
        """Benchmark offloading JSON data."""
        data = {"key": "value", "nested": {"data": [1, 2, 3]}}
        for _ in range(100):
            temp_filestore.offload_json(data)

    def benchmark_load_content(self, temp_filestore):
        """Benchmark loading content."""
        ref = temp_filestore.offload(SAMPLE_LARGE_TEXT)
        for _ in range(50):
            temp_filestore.load(ref)

    def benchmark_load_preview(self, temp_filestore):
        """Benchmark loading preview."""
        ref = temp_filestore.offload("\n".join([f"Line {i}" for i in range(1000)]))
        for _ in range(50):
            temp_filestore.load_preview(ref, lines=10)


class TestContextManagerBenchmarks:
    """Benchmarks for context manager operations."""

    @pytest.fixture
    def temp_context_manager(self, tmp_path):
        """Create temporary context manager."""
        return ContextManager(base_dir=tmp_path)

    def benchmark_create_context(self, temp_context_manager):
        """Benchmark creating contexts."""
        for i in range(50):
            temp_context_manager.create_context(
                agent_id=f"test-{i}",
                model="test-model",
                task=f"Test task {i}",
            )

    def benchmark_add_messages(self, temp_context_manager):
        """Benchmark adding messages."""
        temp_context_manager.create_context(
            agent_id="test",
            model="test-model",
            task="Test task",
        )
        for i in range(100):
            temp_context_manager.add_message("user", f"Message {i}")
            temp_context_manager.add_message("assistant", f"Response {i}")

    def benchmark_update_semantic_stream(self, temp_context_manager):
        """Benchmark updating semantic stream."""
        temp_context_manager.create_context(
            agent_id="test",
            model="test-model",
            task="Test task",
        )
        for i in range(50):
            temp_context_manager.update_semantic_stream(
                current_step=f"Step {i}",
                next_steps=[f"Next {i}"],
            )

    def benchmark_context_usage_ratio(self, temp_context_manager):
        """Benchmark context usage calculation."""
        temp_context_manager.create_context(
            agent_id="test",
            model="test-model",
            task="Test task",
        )
        for tokens in range(1000, 32000, 1000):
            temp_context_manager.get_context_usage_ratio(tokens, 32768)


class TestPluginManagerBenchmarks:
    """Benchmarks for plugin manager operations."""

    @pytest.fixture
    def plugin_manager(self):
        """Create plugin manager."""
        return PluginManager()

    def benchmark_enable_disable_tool(self, plugin_manager):
        """Benchmark enabling/disabling tools."""
        for _ in range(100):
            plugin_manager.enable_tool("duckduckgo")
            plugin_manager.disable_tool("duckduckgo")

    def benchmark_get_tool_definitions(self, plugin_manager):
        """Benchmark getting tool definitions."""
        for _ in range(50):
            plugin_manager.get_tool_definitions()

    def benchmark_configure_rag(self, plugin_manager):
        """Benchmark configuring RAG."""
        for i in range(20):
            plugin_manager.configure_rag(
                documents_dir=f"/tmp/docs-{i}",
                vector_store_dir=f"/tmp/rag-{i}",
            )

    def benchmark_list_plugins(self, plugin_manager):
        """Benchmark listing plugins."""
        for _ in range(50):
            plugin_manager.list_plugins()


class TestModelConfigBenchmarks:
    """Benchmarks for model configuration."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create temporary config."""
        config_path = tmp_path / "config.yaml"
        return ModelConfig(config_path=config_path)

    def benchmark_get_tier(self, temp_config):
        """Benchmark getting tier configuration."""
        for _ in range(100):
            temp_config.get_tier("T2")

    def benchmark_get_default_model(self, temp_config):
        """Benchmark getting default model."""
        for tier in ["T1", "T2", "T3", "T4"]:
            for _ in range(25):
                temp_config.get_default_model(tier)

    def benchmark_get_model_vram(self, temp_config):
        """Benchmark getting model VRAM."""
        models = temp_config.get_all_models()
        for _ in range(50):
            for model in models:
                temp_config.get_model_vram(model)


class TestDownloaderBenchmarks:
    """Benchmarks for document downloader."""

    @pytest.fixture
    def temp_downloader(self, tmp_path):
        """Create temporary downloader."""
        return DocumentDownloader(rag_directory=tmp_path)

    def benchmark_generate_filename(self, temp_downloader):
        """Benchmark filename generation."""
        urls = [
            f"https://example.com/file{i}.pdf"
            for i in range(100)
        ]
        for url in urls:
            temp_downloader._generate_filename(url)

    def benchmark_convert_github_url(self, temp_downloader):
        """Benchmark GitHub URL conversion."""
        urls = [
            f"https://github.com/user/repo/blob/main/file{i}.py"
            for i in range(100)
        ]
        for url in urls:
            temp_downloader._convert_github_url(url)

    def benchmark_extract_arxiv_id(self, temp_downloader):
        """Benchmark arXiv ID extraction."""
        urls = [
            f"https://arxiv.org/abs/2301.{i:05d}"
            for i in range(100)
        ]
        for url in urls:
            temp_downloader._extract_arxiv_id(url)

    def benchmark_list_downloaded(self, temp_downloader):
        """Benchmark listing downloaded files."""
        # Create some files
        for i in range(50):
            (temp_downloader.get_rag_directory() / f"file{i}.txt").write_text(f"content {i}")

        for _ in range(20):
            temp_downloader.list_downloaded()


class TestEndToEndBenchmarks:
    """End-to-end benchmarks for common workflows."""

    @pytest.fixture
    def setup_components(self, tmp_path):
        """Set up all components for end-to-end tests."""
        config = ModelConfig(config_path=tmp_path / "config.yaml")
        context = ContextManager(base_dir=tmp_path / "contexts")
        filestore = FileStore(base_dir=tmp_path / "filestore")
        classifier = AutoTierClassifier(hardware_vram_gb=8.0)
        plugins = PluginManager()

        return {
            "config": config,
            "context": context,
            "filestore": filestore,
            "classifier": classifier,
            "plugins": plugins,
        }

    def benchmark_task_analysis_workflow(self, setup_components):
        """Benchmark complete task analysis workflow."""
        classifier = setup_components["classifier"]

        # Simulate analyzing multiple models for task routing
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES]
        analyses = classifier.classify_local_models(models)

        # Classify each by tier
        for analysis in analyses:
            classifier._params_to_tier(analysis.estimated_params_b)

    def benchmark_context_with_offload(self, setup_components):
        """Benchmark context management with file offloading."""
        context = setup_components["context"]
        filestore = setup_components["filestore"]

        # Create context
        context.create_context(
            agent_id="benchmark",
            model="test-model",
            task="Benchmark task",
        )

        # Add messages and offload large content
        for i in range(20):
            context.add_message("user", f"User message {i}")
            context.add_message("assistant", f"Assistant response {i}")

            # Offload large content
            if i % 5 == 0:
                ref = filestore.offload(SAMPLE_LARGE_TEXT, prefix=f"batch-{i}")
                context.update_semantic_stream(
                    key_decision=f"Offloaded content to {ref}",
                )

    def benchmark_plugin_configuration_workflow(self, setup_components):
        """Benchmark plugin configuration workflow."""
        plugins = setup_components["plugins"]

        # Configure all plugins
        for plugin in ["duckduckgo", "visit_website", "big_rag"]:
            plugins.enable_tool(plugin)
            params = plugins.get_tool_parameters(plugin)
            if params:
                for key in params:
                    plugins.set_tool_parameter(plugin, key, f"value-{key}")

        # Get definitions
        plugins.get_tool_definitions()


# Performance thresholds (in seconds)
# These are maximum acceptable times for each benchmark
PERFORMANCE_THRESHOLDS = {
    "benchmark_extract_params": 0.01,  # 10ms per model
    "benchmark_analyze_model": 0.005,  # 5ms per analysis
    "benchmark_classify_local_models": 0.05,  # 50ms for 80 models
    "benchmark_offload_small": 0.1,  # 100ms for 100 small files
    "benchmark_offload_large": 0.5,  # 500ms for 10 large files
    "benchmark_generate_filename": 0.01,  # 10ms for 100 filenames
}


@pytest.fixture
def benchmark_timer():
    """Create a benchmark timer."""
    class Timer:
        def __init__(self):
            self.times = {}

        def record(self, name, duration):
            self.times[name] = duration

        def check_threshold(self, name, threshold):
            if name in self.times:
                return self.times[name] < threshold
            return True

    return Timer()


class TestPerformanceThresholds:
    """Test that benchmarks meet performance thresholds."""

    def test_extract_params_performance(self):
        """Test parameter extraction meets threshold."""
        classifier = AutoTierClassifier()
        start = time.perf_counter()
        for model_name in SAMPLE_MODEL_NAMES:
            classifier._extract_params(model_name.lower())
        duration = time.perf_counter() - start

        # Should complete in under 10ms per model
        assert duration < len(SAMPLE_MODEL_NAMES) * 0.01

    def test_filestore_offload_performance(self, tmp_path):
        """Test filestore offload meets threshold."""
        filestore = FileStore(base_dir=tmp_path)
        content = "Test content" * 100

        start = time.perf_counter()
        for _ in range(100):
            filestore.offload(content)
        duration = time.perf_counter() - start

        # Should complete in under 100ms
        assert duration < 0.1

    def test_classifier_batch_performance(self):
        """Test batch classification meets threshold."""
        classifier = AutoTierClassifier()
        models = [{"model_key": name} for name in SAMPLE_MODEL_NAMES * 10]

        start = time.perf_counter()
        classifier.classify_local_models(models)
        duration = time.perf_counter() - start

        # Should complete in under 50ms
        assert duration < 0.05
