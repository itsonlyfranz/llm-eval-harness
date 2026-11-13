#!/usr/bin/env bash
# Reproduces every run reported in the README. Requires a running Ollama server with qwen3:8b pulled
# (OLLAMA_MODELS=~/models ollama serve; OLLAMA_MODELS=~/models ollama pull qwen3:8b) and AGENT_REPOS set.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/llm-eval}"
"$PY" run --label baseline
"$PY" run --label baseline-repeat
"$PY" run --label truncated --num-predict 40
"$PY" compare baseline baseline-repeat || true
"$PY" compare baseline truncated || true
"$PY" dashboard
