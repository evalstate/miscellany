#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SPACE="${SPACE:-evalstate/presentations}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Publish presentations}"
publish/scripts/build.sh
publish/scripts/verify.sh
hf upload "$SPACE" publish/site . --repo-type space --commit-message "$COMMIT_MESSAGE" --delete '*'
