# HenryCLI - Agent System Architecture

A CLI tool for orchestrating local LLM models via LM Studio with dynamic model loading/unloading for multi-agent task execution.

## Hardware Target

- **GPU**: RTX 4060 Laptop (8GB VRAM)
- **RAM**: 32GB System Memory
- **Model Range**: 4B - 30B parameters
- **Inference Speed**: ~50 tok/sec (4B) to ~5 tok/sec (30B)

---

## Core Problem

Local models require loading/unloading due to VRAM constraints. A multi-agent system must:
1. Route tasks to appropriate model sizes
2. Minimize model swap overhead
3. Preserve context across agent boundaries
4. Balance speed vs. capability per task

---

## Key Optimizations (Research-Backed)

### 1. Quantization Strategy (GGUF Format)

| Model Size | Q4_K_M VRAM | Q5_K_M VRAM | Recommended |
|------------|-------------|-------------|-------------|
| 4B         | ~2.5 GB     | ~3 GB       | Q4_K_M (always resident) |
| 7B         | ~4.5 GB     | ~5.5 GB     | Q4_K_M (hot cache) |
| 14B        | ~8 GB       | ~9 GB       | Q4_K_M (requires CPU offload) |
| 30B        | ~16 GB      | ~18 GB      | Q4_K_M (heavy CPU offload) |

**Recommendations:**
- Use **Q4_K_M** as default - best quality/speed/VRAM balance
- Avoid Q2 - noticeably degraded quality
- Q5_K_M only if VRAM headroom available
- For 30B models: expect 10-20x slower inference with CPU offload

### 2. Hierarchical Routing (HAPS-Inspired)

Two-level routing architecture:
- **High-Level Router** (1B-4B): Selects agent architecture
- **Low-Level Router**: Task-specific parameter adaptation via prompt engineering

```
Task → Router (1B-4B) → Architecture Selection → Specialist Agent → Output
                          ↓
                    Prompt Template Injection (task-specific)
```

### 3. Context Management (LangChain Deep Agents Pattern)

**Threshold-Based Compression:**
- 85% context window → trigger compression
- Offload tool inputs/outputs to filesystem
- Replace with file path references + previews

**Rolling Summarization:**
- Generate structured summary (intent, artifacts, next steps)
- Preserve full conversation to filesystem
- Agent can recover details via file search

### 4. State Serialization (CaveAgent Pattern)

**Dual-Stream Architecture:**
- **Semantic Stream**: Conversation history (text)
- **Runtime Stream**: External state store (filesystem/variables)

Benefits:
- Avoids serializing complex data into context
- Persistent state across model swaps
- Object references instead of full serialization

---

## Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      TASK INPUT                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ROUTER AGENT (4B model, always resident)                    │
│  - Analyzes task complexity                                  │
│  - Decomposes into subtasks                                  │
│  - Assigns subtasks to specialist agents                     │
│  - Aggregates results                                        │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  CODE AGENT      │ │  RESEARCH AGENT  │ │  WRITING AGENT   │
│  (7B-15B)        │ │  (4B-7B)         │ │  (4B-7B)         │
│  - Complex logic │ │  - Fact lookup   │ │  - Documentation │
│  - Debugging     │ │  - Summarization │ │  - Explanation   │
│  - Architecture  │ │  - Quick queries │ │  - Simple tasks  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
            │               │               │
            ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│  SPECIALIST SUB-AGENTS (loaded on-demand, 15B-30B)           │
│  - Deep reasoning tasks                                      │
│  - Complex mathematical problems                             │
│  - Multi-file refactoring                                    │
│  - Security analysis                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Tiers

| Tier | Model Size | VRAM Usage | Use Case | Always Resident |
|------|------------|------------|----------|-----------------|
| T1   | 3B-4B      | ~2-3 GB    | Routing, simple Q&A, classification | Yes |
| T2   | 7B-9B      | ~5-6 GB    | Code generation, documentation, research | No |
| T3   | 14B-15B    | ~8-9 GB*   | Complex code, debugging, architecture | No |
| T4   | 20B-30B    | ~12-16 GB* | Deep reasoning, security, math | No |

*Requires partial offload to RAM for 8GB VRAM constraint

---

## Model Management

### Loading Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL POOL MANAGER                        │
├─────────────────────────────────────────────────────────────┤
│  Resident Models (GPU):                                      │
│    - T1 Router (4B Q4_K_M, ~2.5GB VRAM)                      │
│                                                              │
│  Hot Cache (GPU, swapped on-demand):                         │
│    - T2 General (7B Q4_K_M, ~4.5GB VRAM)                     │
│    - T3 Code (7B-13B Q4_K_M, ~8GB VRAM)                      │
│                                                              │
│  Warm Cache (RAM, quick GPU load):                           │
│    - GGUF files memory-mapped, ~30 sec load                  │
│                                                              │
│  Cold Storage (disk):                                        │
│    - All other models                                        │
└─────────────────────────────────────────────────────────────┘
```

### LM Studio Model Switching (API Polling Pattern)

Based on LM Studio Batch Prompt Automator analysis:

```python
async def switch_model(target_model: str) -> bool:
    """
    LM Studio doesn't have direct model load API.
    Strategy: Poll /v1/models endpoint to detect changes.
    """
    # Step 1: Save current context
    await save_context()
    
    # Step 2: Prompt user to switch model (or auto via LM Studio API if available)
    # Note: LM Studio v1 API may support model switching via settings
    
    # Step 3: Poll until model is loaded
    max_retries = 60  # 30 seconds with 500ms polling
    for i in range(max_retries):
        response = await lmstudio.get("/v1/models")
        if target_model in response.models:
            await restore_context(target_model)
            return True
        await asyncio.sleep(0.5)
    
    return False
```

**Key Insight:** LM Studio model switching requires:
1. Strict polling to ensure model is fully unloaded before next load
2. Cross-reference v0 and v1 API endpoints for accurate RAM state
3. Queue mechanism for sequential model operations

### State Preservation

When unloading a model:
1. Serialize conversation history to filesystem
2. Save runtime state (variables, file references) to separate file
3. Store pending subtasks and execution pointers
4. Write context metadata (model, timestamp, task_id)

When loading a model:
1. Restore conversation history from filesystem
2. Load runtime state (file references, variable signatures)
3. Resume from pending subtasks
4. Reconstruct agent working memory

---

## Agent Types

### 1. Router Agent
- **Model**: 4B Q4_K_M (e.g., Phi-3-mini-4k, Qwen2.5-3B-Instruct)
- **VRAM**: ~2.5 GB (always resident)
- **Responsibility**: Task decomposition and routing
- **Prompt Template**:
  ```
  You are a task router. Analyze the input and:
  1. Classify task type: [code|research|writing|reasoning]
  2. Estimate complexity: [simple|moderate|complex]
  3. Decompose into subtasks if needed
  4. Assign each subtask to appropriate agent tier
  
  Output JSON: { "type": "...", "complexity": "...", "subtasks": [...] }
  ```

### 2. Code Agent
- **Model**: 7B-13B Q4_K_M (e.g., DeepSeek-Coder-6.7B, CodeLlama-13B)
- **VRAM**: ~4.5-8 GB (loaded on-demand)
- **Responsibility**: Code generation, debugging, refactoring
- **Capabilities**:
  - Single-file edits
  - Multi-file understanding
  - Test generation
  - Error diagnosis
- **Context Management**: Offload large file contents to filesystem, use file path references

### 3. Research Agent
- **Model**: 4B-7B Q4_K_M (e.g., Llama-3.2-3B-Instruct, Phi-3-mini)
- **VRAM**: ~2.5-4.5 GB (can coexist with router in some configs)
- **Responsibility**: Information gathering, summarization
- **Capabilities**:
  - File content analysis
  - Pattern recognition
  - Documentation lookup
  - Quick fact extraction
- **Optimization**: Use rolling summarization for long documents

### 4. Reasoning Agent (On-Demand)
- **Model**: 20B-30B Q4_K_M (e.g., Qwen2.5-32B-Instruct, Command-R)
- **VRAM**: ~16-18 GB (requires significant CPU offload)
- **Responsibility**: Complex multi-step reasoning
- **Capabilities**:
  - Architecture decisions
  - Security analysis
  - Mathematical proofs
  - Cross-domain synthesis
- **Performance Note**: Expect 5-10 tok/sec with CPU offload on 8GB VRAM

### 5. Compression Agent (Background)
- **Model**: 4B Q4_K_M (shares router model)
- **Responsibility**: Context compression and summarization
- **Triggers**:
  - Context window > 85% capacity
  - Tool output > 20,000 tokens
  - Before model switch
- **Actions**:
  - Generate structured summary
  - Offload to filesystem
  - Replace with file references

---

## LM Studio API Integration

### Endpoints Used

```bash
# Get loaded models
GET http://localhost:1234/v1/models

# Load a model (via settings change - requires restart or manual)
# Note: LM Studio doesn't have direct load/unload API
# Workaround: Use model picker + context preservation

# Chat completion
POST http://localhost:1234/v1/chat/completions
{
  "model": "selected-model",
  "messages": [...],
  "stream": true
}
```

### Model Switching Workflow

Since LM Studio lacks programmatic model loading:

1. **Manual Mode** (v1): User confirms model switch via CLI prompt
2. **Semi-Auto Mode** (v2): CLI detects model change needed, prompts user
3. **Auto Mode** (future): Integration with LM Studio server API if available

---

## Data Flow

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  User    │────▶│  Router  │────▶│ Specialist│────▶│  Output  │
│  Input   │     │  Agent   │     │  Agent    │     │  Merge   │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                      │                  │
                      │                  │
                      ▼                  ▼
               ┌──────────┐       ┌──────────┐
               │ Context  │       │ Context  │
               │  Store   │       │  Store   │
               └──────────┘       └──────────┘
```

---

## Context Management

### Dual-Stream Architecture (CaveAgent Pattern)

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXT MANAGER                           │
├─────────────────────────────────────────────────────────────┤
│  Semantic Stream (LLM Context):                              │
│    - Conversation history (compressed)                       │
│    - Current task instructions                               │
│    - Recent tool outputs (< 20k tokens)                      │
│                                                              │
│  Runtime Stream (Filesystem):                                │
│    - Full conversation logs                                  │
│    - Large tool inputs/outputs                               │
│    - Generated artifacts (code, files)                       │
│    - State variables and references                          │
└─────────────────────────────────────────────────────────────┘
```

### Context File Structure

```json
{
  "agent_id": "code-agent-001",
  "model": "deepseek-coder-6.7b-instruct-q4_k_m",
  "task": "refactor authentication module",
  "semantic_stream": {
    "conversation_summary": "Refactoring login flow...",
    "current_step": "Updating test cases",
    "next_steps": ["update docs", "run integration tests"]
  },
  "runtime_stream": {
    "full_history_path": "~/.henrycli/contexts/active/code-agent-001/history.json",
    "artifacts": ["auth.py", "test_auth.py"],
    "file_references": {
      "auth_module": "~/.henrycli/filestore/auth_module_full.txt",
      "test_output": "~/.henrycli/filestore/test_output.txt"
    }
  },
  "context_usage": {
    "tokens_used": 28000,
    "tokens_limit": 32768,
    "compression_triggered": true
  },
  "timestamp": "2026-03-27T10:30:00Z"
}
```

### Compression Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Context usage | > 85% | Rolling summarization |
| Tool output | > 20,000 tokens | Offload to filesystem |
| Tool input (edit) | Large payloads | Replace with file reference |
| Before model switch | Any usage | Full state serialization |

### Rolling Summarization Process

```python
async def compress_context(conversation: list) -> dict:
    """
    Compress conversation using rolling summarization.
    """
    # Step 1: Save full conversation to filesystem
    history_path = save_to_filestore(conversation)
    
    # Step 2: Generate structured summary
    summary = await router_agent.generate_summary(
        conversation,
        fields=["intent", "artifacts", "decisions", "next_steps"]
    )
    
    # Step 3: Return compressed context
    return {
        "summary": summary,
        "full_history_reference": history_path,
        "recent_messages": conversation[-10:]  # Keep last 10 messages
    }
```

### Context Directory Structure

```
~/.henrycli/
├── contexts/
│   ├── active/
│   │   ├── {task_id}/
│   │   │   ├── context.json        # Current state
│   │   │   ├── history.json        # Full conversation
│   │   │   └── artifacts/          # Generated files
│   │   └── ...
│   ├── completed/
│   │   └── {task_id}/              # Archived after completion
│   └── archived/
│       └── {YYYY-MM}/              # Compressed after 7 days
├── filestore/
│   ├── {file_hash}_full.txt        # Offloaded large content
│   └── ...
├── models/
│   └── config.json                 # Model tier assignments
└── logs/
    └── sessions/
        └── {YYYY-MM-DD}.log        # Session logs
```

---

---

## Task Execution Flow

```python
# Pseudocode for agent orchestration with optimizations

class HenryOrchestrator:
    def __init__(self):
        self.current_model = "phi-3-mini-4k"  # T1 router always loaded
        self.context_manager = ContextManager()
        self.model_pool = ModelPool()
    
    async def execute_task(self, user_input: str) -> Result:
        # Step 1: Router analyzes task (T1 model, always loaded)
        analysis = await self.router_agent.analyze(user_input)
        
        # Step 2: Create execution plan with model assignments
        plan = self.create_plan(analysis)
        
        # Step 3: Execute subtasks with appropriate agents
        results = []
        for subtask in plan.subtasks:
            agent = self.select_agent(subtask.type, subtask.complexity)
            target_model = agent.model
            
            # Check if model switch needed
            if target_model != self.current_model:
                # Compress context before switch
                if self.context_manager.usage > 0.85:
                    await self.context_manager.compress()
                
                # Save current agent state
                await self.context_manager.save_state(self.current_model)
                
                # Switch model (with polling)
                success = await self.model_pool.switch_model(target_model)
                if not success:
                    # Fallback to smaller model
                    target_model = self.get_fallback_model(subtask.complexity)
                    await self.model_pool.switch_model(target_model)
                
                # Restore new agent state
                await self.context_manager.restore_state(target_model)
                self.current_model = target_model
            
            # Execute subtask
            result = await agent.execute(subtask)
            results.append(result)
        
        # Step 4: Router aggregates results
        final = await self.router_agent.aggregate(results)
        
        return final
    
    async def run_with_context_management(self, agent, task):
        """
        Execute agent with automatic context compression.
        """
        while not task.is_complete:
            # Check context usage
            if self.context_manager.usage > 0.85:
                await self.context_manager.compress()
            
            # Execute step
            step_result = await agent.step(task)
            
            # Check for large outputs
            if len(step_result.output) > 20000:
                # Offload to filesystem
                file_ref = self.context_manager.offload(step_result.output)
                step_result.output_ref = file_ref
                step_result.output = step_result.output[:1000]  # Keep preview
            
            task.update(step_result)
        
        return task.result
```

### Model Selection Algorithm (HAPS-Inspired)

```python
def select_agent(task_type: str, complexity: str) -> Agent:
    """
    Hierarchical model selection based on HAPS routing.
    """
    # High-level: Select architecture based on task type
    architecture_map = {
        "code": "T3",      # 7B-13B code-specialized
        "research": "T2",  # 4B-7B general purpose
        "writing": "T2",   # 4B-7B general purpose
        "reasoning": "T4"  # 20B-30B deep reasoning
    }
    
    # Low-level: Adjust based on complexity
    complexity_adjustments = {
        "simple": -1,   # Downgrade tier
        "moderate": 0,  # Keep default
        "complex": +1   # Upgrade tier
    }
    
    base_tier = architecture_map.get(task_type, "T2")
    adjustment = complexity_adjustments.get(complexity, 0)
    
    final_tier = adjust_tier(base_tier, adjustment)
    
    return get_agent_for_tier(final_tier)
```

---

## Configuration

### Default Model Assignments (`~/.henrycli/models/config.json`)

```json
{
  "tiers": {
    "T1": {
      "models": [
        "phi-3-mini-4k-instruct-q4_k_m",
        "qwen2.5-3b-instruct-q4_k_m",
        "llama-3.2-3b-instruct-q4_k_m"
      ],
      "default": "phi-3-mini-4k-instruct-q4_k_m",
      "purpose": "routing",
      "vram_gb": 2.5,
      "resident": true
    },
    "T2": {
      "models": [
        "qwen2.5-7b-instruct-q4_k_m",
        "llama-3.1-8b-instruct-q4_k_m",
        "mistral-7b-instruct-v0.3-q4_k_m"
      ],
      "default": "qwen2.5-7b-instruct-q4_k_m",
      "purpose": "general",
      "vram_gb": 4.5,
      "resident": false
    },
    "T3": {
      "models": [
        "deepseek-coder-6.7b-instruct-q4_k_m",
        "codellama-13b-instruct-q4_k_m",
        "starcoder2-7b-q4_k_m"
      ],
      "default": "deepseek-coder-6.7b-instruct-q4_k_m",
      "purpose": "code",
      "vram_gb": 8.0,
      "resident": false
    },
    "T4": {
      "models": [
        "qwen2.5-32b-instruct-q4_k_m",
        "command-r-q4_k_m",
        "yi-34b-chat-q4_k_m"
      ],
      "default": "qwen2.5-32b-instruct-q4_k_m",
      "purpose": "reasoning",
      "vram_gb": 16.0,
      "resident": false,
      "cpu_offload": true
    }
  },
  "hardware": {
    "vram_gb": 8,
    "ram_gb": 32,
    "gpu": "RTX 4060 Laptop",
    "gpu_layers_max": 99,
    "cpu_offload_threshold": 0.9
  },
  "context": {
    "compression_threshold": 0.85,
    "offload_token_limit": 20000,
    "keep_recent_messages": 10
  },
  "performance": {
    "model_switch_timeout_sec": 60,
    "model_switch_poll_interval_ms": 500,
    "inference_timeout_sec": 300
  }
}
```

### Hardware-Specific Tuning

For RTX 4060 Laptop (8GB VRAM):

```json
{
  "gpu_layers_default": 99,
  "gpu_layers_t4": 20,
  "main_gpu": 0,
  "tensor_split": [0.8, 0.2],
  "flash_attn": true,
  "kv_cache_type": "q4_0"
}
```

**Explanation:**
- `gpu_layers_t4: 20` - Keep only first 20 layers on GPU for 30B models
- `tensor_split` - 80% GPU, 20% CPU for large models
- `kv_cache_type: q4_0` - Quantized KV cache for longer context
- `flash_attn: true` - Enable flash attention for speed

---

## CLI Commands

```bash
# Main task execution
henry "Refactor the authentication module"

# Interactive mode
henry --interactive

# Specify agent tier manually
henry --tier T3 "Write a binary search implementation"

# List available models
henry models list

# Set default model for tier
henry models set-default --tier T2 --model "qwen2.5-7b"

# View context
henry context show --task-id abc123

# Clear all contexts
henry context clear

# Show session log
henry logs --today
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] LM Studio API client (OpenAI-compatible)
- [ ] Model polling mechanism for switch detection
- [ ] GGUF quantization config system
- [ ] Context serialization/deserialization
- [ ] Filesystem-based filestore for offloading
- [ ] Basic CLI interface with argparse/typer

### Phase 2: Router Agent
- [ ] Task classification logic (type, complexity)
- [ ] HAPS-inspired hierarchical routing
- [ ] Subtask decomposition
- [ ] Agent selection algorithm
- [ ] Result aggregation
- [ ] JSON output parsing

### Phase 3: Context Management
- [ ] Dual-stream architecture (semantic + runtime)
- [ ] Rolling summarization with router agent
- [ ] Threshold-based compression triggers
- [ ] File reference system for offloaded content
- [ ] Context state save/restore
- [ ] History recovery from filesystem

### Phase 4: Specialist Agents
- [ ] Code agent implementation
- [ ] Research agent implementation
- [ ] Reasoning agent (T4) with CPU offload support
- [ ] Compression agent (background)
- [ ] Model switching workflow with polling
- [ ] Fallback tier selection

### Phase 5: Optimization
- [ ] Model hot caching (RAM)
- [ ] Parallel subtask execution (same model)
- [ ] Predictive prefetching
- [ ] Performance metrics and logging
- [ ] GPU layer tuning per tier
- [ ] KV cache quantization

### Phase 6: Advanced Features
- [ ] Learning from past task assignments
- [ ] Custom agent definitions
- [ ] Plugin system for new agent types
- [ ] Multi-session context linking
- [ ] Interactive mode with streaming
- [ ] Progress indicators for model switches

---

## Error Handling

| Error | Recovery Strategy |
|-------|-------------------|
| Model fails to load | Retry with smaller quantization, fallback to lower tier |
| Context corruption | Load from last known good checkpoint (filesystem) |
| LM Studio unavailable | Queue tasks, notify user, retry with exponential backoff |
| VRAM exhaustion | Force unload non-essential models, use CPU offload, tier downgrade |
| Task timeout | Save partial progress, allow resume, notify user |
| Model switch timeout | Poll extended (120s), fallback to current model with warning |
| Context compression fails | Use simple truncation, keep recent messages only |
| Filestore write fails | Retry 3x, use temp directory, alert user |
| JSON parsing error (router) | Retry with stricter prompt, use regex fallback extraction |
| GPU out of memory | Reduce gpu_layers, switch to CPU-only for current task |

### Graceful Degradation Path

```
T4 (30B) → T3 (13B) → T2 (7B) → T1 (4B)
   ↓          ↓          ↓         ↓
 CPU offload  GPU full   GPU full  GPU + batch
```

When a model fails:
1. Save error context to logs
2. Select fallback tier
3. Notify user of degradation
4. Continue with smaller model
5. Offer retry with original model after task

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Router response time | < 2 seconds | T1 model always resident |
| Model switch overhead | < 30 seconds | RAM cache (warm) |
| Model switch overhead | < 120 seconds | Disk load (cold) |
| Context save/restore | < 1 second | Filesystem operations |
| Context compression | < 5 seconds | Router-generated summary |
| Task completion (simple) | < 30 seconds | T1-T2 models |
| Task completion (moderate) | < 2 minutes | T2-T3 models |
| Task completion (complex) | < 5 minutes | T3-T4 models with CPU offload |
| T4 inference speed | 5-10 tok/sec | 30B with CPU offload |
| T3 inference speed | 15-25 tok/sec | 7B-13B on GPU |
| T1 inference speed | 40-60 tok/sec | 4B on GPU |

### Bottleneck Analysis

| Bottleneck | Mitigation |
|------------|------------|
| Model loading time | Hot cache for frequently-used models |
| CPU offload slowdown | Use only for T4 (complex reasoning) |
| Context compression | Async compression during user review |
| VRAM exhaustion | Aggressive offloading, tier downgrade |
| Sequential subtasks | Parallel execution for same-model tasks |

### Optimization Opportunities

1. **Parallel Subtask Execution**: Run subtasks requiring same model in parallel
2. **Predictive Prefetching**: Load next model while user reviews output
3. **Tier Downgrade Fallback**: Auto-fallback to smaller model if larger fails
4. **Batch Context Compression**: Compress multiple contexts together
5. **Model Warm Cache**: Keep last-used T2/T3 models in RAM (not GPU)

---

## Contributors

- **Primary Developer**: ViswaaTheMightyPickle
- **Contact**: thereddragonspeaks22919@protonmail.com
- **GitHub**: https://github.com/ViswaaTheMightyPickle/HenryCLI

---

## Research References

This architecture incorporates findings from:

1. **HAPS: Hierarchical LLM Routing with Joint Architecture and Parameter Search** (arXiv:2601.05903)
   - Two-level routing: architecture selection + parameter adaptation
   - Reward-augmented routing optimization
   - [Paper](https://arxiv.org/html/2601.05903v1)

2. **CaveAgent: Transforming LLMs into Stateful Runtime Operators** (arXiv:2601.01569)
   - Dual-stream context architecture (semantic + runtime)
   - Persistent kernel for state preservation
   - Serializable runtime environments
   - [Paper](https://arxiv.org/html/2601.01569v1)

3. **LangChain Deep Agents: Context Management**
   - Threshold-based compression (85% context window)
   - Filesystem abstraction for offloaded content
   - Rolling summarization with file references
   - [Blog](https://blog.langchain.com/context-management-for-deepagents/)

4. **GGUF Quantization Best Practices**
   - Q4_K_M as optimal quality/VRAM balance
   - CPU offload strategies for large models
   - [Guide](https://www.kunalganglani.com/blog/running-local-llms-2026-hardware-setup-guide/)

5. **LM Studio Batch Prompt Automator**
   - API polling pattern for model switch detection
   - Sequential model queue mechanism
   - [GitHub](https://github.com/skiranjotsingh/lmstudio-batch-prompt-automator)

6. **Optimizing Local LLM Inference for 8GB VRAM GPUs**
   - Quantization recommendations
   - KV cache optimization
   - [Article](https://hackernoon.com/optimizing-local-llm-inference-for-8gb-vram-gpus)

---

## License

MIT License - See LICENSE file for details

---

## Version

- **Document Version**: 1.1.0 (Research-Optimized)
- **Last Updated**: 2026-03-27
- **Changes from 1.0.0**:
  - Added GGUF quantization strategy (Q4_K_M recommended)
  - Integrated HAPS hierarchical routing architecture
  - Implemented CaveAgent dual-stream context management
  - Added LangChain-inspired context compression
  - LM Studio API polling pattern for model switching
  - Hardware-specific tuning for RTX 4060 Laptop (8GB VRAM)
  - Expanded error handling with graceful degradation
  - Detailed performance targets with bottleneck analysis
