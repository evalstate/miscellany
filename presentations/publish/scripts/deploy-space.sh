#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
SPACE="${SPACE:-evalstate/presentations}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Publish presentations}"
if [ ! -f publish/site/index.html ]; then
  echo "publish/site is missing; run publish/scripts/build.sh first" >&2
  exit 1
fi
publish/scripts/verify.sh
hf upload "$SPACE" publish/site . --repo-type space --commit-message "$COMMIT_MESSAGE" --delete '*'
