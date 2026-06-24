#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python - <<'PY'
import json
from pathlib import Path
root = Path.cwd()
site = root / 'publish/site'
manifest = json.loads((root / 'publish/manifest.json').read_text())
missing = []
for item in manifest['items']:
    if item.get('status') != 'published':
        continue
    entry = site / item['publishPath'] / item['entry']
    if not entry.exists():
        missing.append(str(entry.relative_to(root)))
if not (site / 'index.html').exists():
    missing.append('publish/site/index.html')
if not (site / 'README.md').exists():
    missing.append('publish/site/README.md')
if missing:
    raise SystemExit('Missing built files:\n- ' + '\n- '.join(missing))
print('ok:', site.relative_to(root))
PY
