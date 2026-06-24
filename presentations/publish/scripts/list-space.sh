#!/usr/bin/env bash
set -euo pipefail
SPACE="${1:-evalstate/presentations}"
hf spaces list "$SPACE" --recursive
