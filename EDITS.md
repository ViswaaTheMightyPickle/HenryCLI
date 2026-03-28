# EDITS.md - HenryCLI Architecture Rework Plan

## Analysis Date: 2026-03-28
## Status: ✅ COMPLETED

All critical issues have been addressed and all 159 tests pass.

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
