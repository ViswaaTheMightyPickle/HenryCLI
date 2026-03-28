# EDITS.md - HenryCLI Architecture Rework Plan

## Analysis Date: 2026-03-28
## Status: ✅ COMPLETED + NEW FEATURES

All critical issues have been addressed and all 159 tests pass.
New `henry init` command and TUI rework completed.

---

## Executive Summary

After comprehensive testing of HenryCLI commands and comparison with LM Studio (lms) CLI, several critical mismatches and bugs were identified and fixed. This document outlines the issues and the completed rework.

---

## Completed Changes

### Phase 1: Critical Bug Fixes ✅

1. **Fixed `henry discover` model name extraction** ✅
   - File: `src/henrycli/auto_tier.py`
   - Change: Handle camelCase JSON fields from `lms ls --json`
   - Test: Verified model names display correctly

2. **Added CLI fallback for `henry load`** ✅
   - File: `src/henrycli/lmstudio.py`
   - Change: Uses subprocess to `lms load` when REST API fails
   - Test: Verified model loading works with CLI fallback

### Phase 2: Missing Core Features ✅

3. **Added `henry download` command** ✅
   - File: `src/henrycli/cli.py`, `src/henrycli/lmstudio.py`
   - Change: Wraps `lms get` functionality
   - Test: Verified model download works

4. **Added `henry server` command group** ✅
   - File: `src/henrycli/cli.py`, `src/henrycli/lmstudio.py`
   - Commands: `--status`, `--start`, `--stop`
   - Test: Verified server management works

5. **Fixed model listing confusion** ✅
   - File: `src/henrycli/cli.py`
   - Change: Added `--local` flag to show installed models
   - Test: Verified output matches `lms ls`

6. **Added `henry import-model` command** ✅
   - File: `src/henrycli/cli.py`, `src/henrycli/lmstudio.py`
   - Change: Wraps `lms import` functionality

### Phase 3: Initialization & TUI Rework ✅

7. **Added `henry init` command** ✅
   - File: `src/henrycli/cli.py`
   - Change: Complete initialization workflow
   - Features:
     - Discovers all local models (API + CLI fallback)
     - Classifies models into tiers (T1-T4) by parameter count
     - Auto-generates and saves tier configuration to `~/.henrycli/models/config.yaml`
     - Optionally loads T1 routing model automatically
   - Usage:
     - `henry init` - Full init with auto-load
     - `henry init --no-load` - Init without loading model
     - `henry init --use-cli` - Force CLI fallback for discovery
   - Test: Run `henry init` to verify full workflow

8. **Reworked TUI for better UX** ✅
   - File: `src/henrycli/tui.py`
   - Changes:
     - Improved CSS layout with better sizing and scrolling
     - Enhanced model discovery with CLI fallback
     - Fixed load buttons to actually load models
     - Added status bar auto-update on model load/unload
     - Better tier display showing model names and counts
     - Improved log output with auto-scroll
     - Added initialization state tracking
   - Features:
     - Auto-discovers models on startup
     - One-click model loading by tier
     - Real-time status updates
     - Task history tracking
   - Usage: Run `henry tui` to launch

---

## Architecture Overview

### HenryCLI Command Structure

```
henry init           # Initialize and auto-configure (NEW)
henry run <task>     # Execute task with agent routing
henry analyze <task> # Analyze task without executing
henry discover       # Discover and classify local models
henry models         # Show configured/local models
henry load <model>   # Load a model
henry unload         # Unload models
henry download       # Download model from Hub
henry server         # Manage LM Studio server
henry import-model   # Import local model file
henry tui            # Launch terminal UI
```

### Model Tier System

- **T1** (< 5B params): Routing agent, resident in VRAM
- **T2** (5-10B params): General tasks
- **T3** (10-20B params): Code generation
- **T4** (20B+ params): Complex reasoning, CPU offload

### Configuration Flow

1. `henry init` discovers local models
2. Auto-classifies by parameter count extraction from model names
3. Generates tier configuration with VRAM estimates
4. Saves to `~/.henrycli/models/config.yaml`
5. Optionally loads T1 routing model

---

## Testing Checklist

- [x] All 159 existing tests pass
- [ ] Manual test: `henry init` with models present
- [ ] Manual test: `henry init --no-load`
- [ ] Manual test: `henry tui` launch and model loading
- [ ] Manual test: `henry discover --use-cli` fallback
