#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SPACE="${1:-evalstate/presentations}"
OUT="publish/current-space"
INFO="publish/current-space-info.json"
rm -rf "$OUT"
mkdir -p "$OUT"
hf spaces info "$SPACE" --json > "$INFO"
hf download "$SPACE" --repo-type space --local-dir "$OUT"
