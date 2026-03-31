#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

npx --yes @marp-team/marp-cli@latest \
  --theme-set ./theme/freud.css ./theme/structure.css ./theme/schema.css ./theme/evalstate-extensions.css ./theme/ny-noir.css \
  --html \
  --allow-local-files \
  -o ./presentation.html \
  ./presentation.md
