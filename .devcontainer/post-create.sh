#!/usr/bin/env bash
set -euo pipefail

if [[ -f "pyproject.toml" ]]; then
  uv sync
else
  echo "pyproject.toml not found; skipping uv sync."
fi
