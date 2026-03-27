# HenryCLI

Multi-agent LLM orchestration for LM Studio with dynamic model loading/unloading.

## Overview

HenryCLI is a CLI tool that orchestrates multiple local LLM models via LM Studio. It intelligently routes tasks to appropriate models based on complexity and task type, automatically managing model loading/unloading to work within VRAM constraints.

**Target Hardware**: RTX 4060 Laptop (8GB VRAM, 32GB RAM)
**Model Range**: 4B - 30B parameters (GGUF Q4_K_M quantization)

## Features

- **Hierarchical Agent System**: Router agent (4B) analyzes tasks and routes to specialist agents
- **Dynamic Model Switching**: Automatically switches models based on task requirements
- **Context Management**: Dual-stream architecture preserves state across model swaps
- **Smart Compression**: Offloads large content to filesystem, keeps context within limits
- **Tier-Based Routing**: 4 model tiers (T1-T4) for different task complexities
- **LM Studio Plugins**: Support for DuckDuckGo, Visit Website, and BigRAG plugins
- **Auto Model Loading**: Load/unload models programmatically via LM Studio API
- **Auto-Tier Classification**: [Experimental] Automatically classify models by parameter count

## Installation

### Quick Install (Mac/Linux)

```bash
# Clone the repository
git clone https://github.com/ViswaaTheMightyPickle/HenryCLI
cd HenryCLI

# Run the installation script
./install.sh
```

The installer will:
- Check Python version (3.10+)
- Install HenryCLI and dependencies
- Set up configuration directories
- Optionally install LM Studio plugins

### Manual Install

```bash
# Install with pip
pip install -e ".[dev]"

# Verify installation
henry version
henry health
```

### Prerequisites

1. **Python 3.10+**
2. **LM Studio** installed and running (https://lmstudio.ai/)
3. **Models downloaded** in LM Studio

## Quick Start

1. **Start LM Studio** and load a model (e.g., Phi-3-mini-4k)

2. **Analyze a task**:
   ```bash
   henry analyze "Write a Python function to sort a list of dictionaries by a key"
   ```

3. **Run a task**:
   ```bash
   henry run "Explain the time complexity of quicksort"
   ```

4. **Force a specific tier**:
   ```bash
   henry run "Build a REST API with authentication" --tier T3
   ```

## Commands

### `henry analyze <task>`
Analyze a task without executing. Shows task type, complexity, and recommended tier.

```bash
henry analyze "Refactor the authentication module to use JWT"
```

### `henry run <task>`
Execute a task with automatic agent routing.

```bash
henry run "Write unit tests for the user service"
```

Options:
- `-t, --tier`: Force specific tier (T1-T4)
- `-i, --interactive`: Interactive mode (prompts for model switches)

### `henry health`
Check LM Studio connection and list loaded models.

```bash
henry health
```

### `henry models`
Manage model configuration.

```bash
henry models --list      # List all configured models
henry models --stats     # Show model statistics and VRAM usage
```

### `henry context`
Manage context storage.

```bash
henry context --show     # Show active contexts
henry context --clear    # Clear all contexts
```

### `henry plugins`
Manage LM Studio plugins.

```bash
henry plugins --list              # List all plugins
henry plugins --enable duckduckgo # Enable DuckDuckGo search
henry plugins --configure-rag     # Configure BigRAG plugin
```

### `henry load <model>`
Load a model in LM Studio.

```bash
henry load "TheBloke/phi-3-mini-4k-instruct-GGUF"
henry load "bartowski/Qwen2.5-7B-Instruct-GGUF" --gpu auto
```

### `henry unload`
Unload models from LM Studio.

```bash
henry unload --all       # Unload all models
henry unload <instance>  # Unload specific model
```

### `henry discover`
[EXPERIMENTAL] Discover and classify local models into tiers.

```bash
henry discover                      # Show classified models
henry discover --auto-configure     # Auto-generate tier config
```

### `henry tui`
Launch the Terminal User Interface (TUI).

```bash
henry tui
```

The TUI provides:
- Real-time model status display
- Interactive task input with analysis
- Model tier discovery and loading
- Task history tracking
- Keyboard shortcuts for common actions

**Keyboard Shortcuts:**
- `Ctrl+Q` - Quit
- `Ctrl+N` - New task
- `Ctrl+R` - Refresh models
- `Ctrl+L` - Clear log
- `F1` - Help
- `Escape` - Clear input

### `henry get <url>`
Download files to the RAG documents directory.

```bash
# Download a GitHub file
henry get https://github.com/user/repo/blob/main/file.py

# Download an arXiv paper
henry get https://arxiv.org/abs/2301.12345

# Download a PDF
henry get https://example.com/document.pdf

# List downloaded files
henry get --list

# Delete a file
henry get --delete filename.pdf
```

### `henry config`
Manage configuration.

```bash
henry config --show      # Show current configuration
henry config --edit      # Show config file path for editing
```

### `henry version`
Show version information.

## Model Tiers

| Tier | Model Size | VRAM | Use Case | Resident |
|------|------------|------|----------|----------|
| T1 | 3B-4B | ~2.5 GB | Routing, simple Q&A | Yes |
| T2 | 7B-9B | ~4.5 GB | General tasks, writing | No |
| T3 | 13B-15B | ~8 GB | Code, debugging | No |
| T4 | 20B-30B | ~16 GB | Deep reasoning | No (CPU offload) |

### Recommended Models

**T1 (Router)**:
- `phi-3-mini-4k-instruct-q4_k_m`
- `qwen2.5-3b-instruct-q4_k_m`

**T2 (General)**:
- `qwen2.5-7b-instruct-q4_k_m`
- `llama-3.1-8b-instruct-q4_k_m`

**T3 (Code)**:
- `deepseek-coder-6.7b-instruct-q4_k_m`
- `codellama-13b-instruct-q4_k_m`

**T4 (Reasoning)**:
- `qwen2.5-32b-instruct-q4_k_m`
- `command-r-q4_k_m`

## Configuration

Configuration is stored at `~/.henrycli/models/config.yaml`.

### Example Configuration

```yaml
tiers:
  T1:
    default: "phi-3-mini-4k-instruct-q4_k_m"
    vram_gb: 2.5
    resident: true
  T2:
    default: "qwen2.5-7b-instruct-q4_k_m"
    vram_gb: 4.5
    resident: false
  T3:
    default: "deepseek-coder-6.7b-instruct-q4_k_m"
    vram_gb: 8.0
    resident: false
  T4:
    default: "qwen2.5-32b-instruct-q4_k_m"
    vram_gb: 16.0
    resident: false
    cpu_offload: true

hardware:
  vram_gb: 8
  ram_gb: 32
  gpu: "RTX 4060 Laptop"

context:
  compression_threshold: 0.85
  offload_token_limit: 20000
  keep_recent_messages: 10
```

## Architecture

### Agent Hierarchy

```
User Input → Router Agent (T1) → Task Analysis
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
              Code Agent      Research       Reasoning
              (T3, 13B)       Agent (T2)     Agent (T4)
```

### Context Management

HenryCLI uses a **dual-stream architecture**:

- **Semantic Stream**: Compressed conversation in LLM context (summary, recent messages)
- **Runtime Stream**: Full state stored on filesystem (complete history, artifacts)

This allows efficient context preservation across model swaps.

## LM Studio Plugins

HenryCLI supports LM Studio plugins for enhanced capabilities:

### DuckDuckGo Search
Web search capability for research tasks.

```bash
# Install plugin
lms get danielsig/duckduckgo

# Enable in HenryCLI
henry plugins --enable duckduckgo
```

### Visit Website
Extract content from webpages.

```bash
# Install plugin
lms get danielsig/visit-website

# Enable in HenryCLI
henry plugins --enable visit_website
```

### BigRAG (Document Search)
RAG-based document retrieval from local files.

```bash
# Install plugin
lms get picklerick/big-rag-rust-accelerated

# Configure in HenryCLI
henry plugins --configure-rag
# Follow prompts for documents directory and vector store

# Download documents to RAG directory
henry get https://github.com/user/repo/blob/main/doc.md
henry get https://arxiv.org/abs/2301.12345
henry get --list  # List downloaded files
```

Documents downloaded with `henry get` are stored in `~/.henrycli/rag-docs/` and can be used with BigRAG for document-aware conversations.

### Plugin Configuration

Plugins are managed via `henry plugins`:

```bash
# List all plugins
henry plugins --list

# Enable a plugin
henry plugins --enable duckduckgo

# Disable a plugin
henry plugins --disable big_rag

# Configure BigRAG
henry plugins --configure-rag
```

### Model Switching

LM Studio doesn't have a direct model load API. HenryCLI:
1. Polls `/v1/models` endpoint to detect model changes
2. Prompts user to switch models in LM Studio (interactive mode)
3. Automatically falls back to smaller models if larger ones fail

## Development

### Run Tests

```bash
pytest
```

### Run with Coverage

```bash
pytest --cov=src/henrycli --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check src tests

# Type checking
mypy src/henrycli
```

### Project Structure

```
HenryCLI/
├── src/henrycli/
│   ├── __init__.py
│   ├── cli.py              # CLI interface
│   ├── lmstudio.py         # LM Studio API client
│   ├── model_switcher.py   # Model switching logic
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py         # Base agent class
│   │   └── router.py       # Router agent
│   ├── context/
│   │   ├── __init__.py
│   │   ├── manager.py      # Context manager
│   │   └── filestore.py    # Filesystem storage
│   └── models/
│       ├── __init__.py
│       ├── config.py       # Model configuration
│       └── pool.py         # Model pool manager
├── tests/
├── config/
├── pyproject.toml
├── agents.md               # Architecture documentation
└── README.md
```

## Troubleshooting

### LM Studio Connection Failed

1. Ensure LM Studio is running
2. Check server is enabled (Port 1234)
3. Run `henry health` to verify connection

### Model Switch Timeout

1. Ensure target model is downloaded in LM Studio
2. Use interactive mode: `henry run "task" -i`
3. Manually load model in LM Studio before running

### Out of Memory

1. Close other GPU applications
2. Use smaller tier: `henry run "task" --tier T2`
3. Reduce context in config: `compression_threshold: 0.75`

### Context Too Large

HenryCLI automatically compresses context when it exceeds 85% of the model's limit. To force compression:
1. Lower threshold in config
2. Use `henry context --clear` to reset

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read the architecture documentation in `agents.md` before contributing.

## Author

- **ViswaaTheMightyPickle**
- Email: thereddragonspeaks22919@protonmail.com
- GitHub: https://github.com/ViswaaTheMightyPickle/HenryCLI

## Version

0.1.0 (Alpha) - Initial release
