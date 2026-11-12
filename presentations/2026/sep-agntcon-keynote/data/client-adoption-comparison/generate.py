"""Completed-source adapter and display normalization; no rendering dependencies.
Run with Python 3; --analysis-root can relocate the pinned data-analysis checkout.
Only writes data artifacts beside this file. Never invokes snapshot.build/render.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ['claude-code', 'Anthropic/ClaudeAI', 'chat-ui-mcp', 'openai-mcp',
         'openai-mcp (Codex)', 'codex-mcp-client']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--analysis-root', type=Path, default=Path('/home/evalstate/source/data-analysis'))
analysis = parser.parse_args().analysis_root.resolve()
PINS = {'scripts/plot_client_migration_focus.py': '179cb529a72b975f58a91cc7d602b30e68337c3fd3d27f0281c5d02e9aca8d24', 'scripts/build_protocol_adoption_dashboard.py': 'd76cc0a348f7487572d350aa7f09b33a9ff10cc22bb3505968e79eab391c2456', 'scripts/plot_client_migration_snapshot.py': 'f936a77541ec718e92dcdd138cf4da462104370bc987a4ab385b760d11d6f063', 'scripts/client_protocol_current_remote.py': '7a47e9aaf4128acb6398872ed357b234822f2499eac8b4cd24db891768a236c1', 'summaries/manifest.json': '553056a83e98bfdd10978c1a001877ce06a48e2aff8d3316824d1771c8f4cf4d', 'summaries/client-protocol-20260916-v1/COMPLETE': '1af2609082426572d53133ca18f1ea474c237e846e957a0aab7ac650076284aa', 'summaries/client-protocol-20260916-v1/manifest.json': '98e98ec622637e6e3ecb0cee56611bd8a50c9339e7ac144c3c5aadf4571ac3a7', 'summaries/client-protocol-20260916-v1/daily_client_protocol.csv': 'b59d866ac529e633e06bf703dc5efd839728633acaf0f839a185573007b09bf1', 'summaries/client-protocol-20260916-v1/daily_protocol.csv': 'ef30f681f85b4c992020cdd0e13aa2744cdc569287427646d3f84ba73d50f43d', 'summaries/client-protocol-20260916-v1/job.json': 'a15bb0e1b85c9ea9e7cdcddc5a65bf9b0aca0ef435ad55d7f37a34af2d0f1c4f', 'summaries/client-protocol-20260916-v1/verify.py': '052d70e1c78334eb42dc898c4ed8b496d6c71c7e4830b8d02fcf9f526e2807a3', 'summaries/client-protocol-20260916-v1/verification.json': '4b25a1de60fc2d2d3fb5fdfb7f883dc942230fe7c9c92da36f6e036a9614ce51', 'summaries/client-protocol-20260916-v1/README.md': '95c7484af677ed59d134672584361f733c77c9711d125abbd7869149f06a39b5'}
for name, digest in PINS.items():
    assert hashlib.sha256((analysis / name).read_bytes()).hexdigest() == digest, name
sys.path.insert(0, str(analysis / 'scripts'))
from plot_client_migration_focus import prepare, sha, MINIMUM
from build_protocol_adoption_dashboard import validate_partial, series_data
from plot_client_migration_snapshot import comparisons
source = analysis / 'summaries/client-protocol-20260916-v1'
allowlist = json.loads((analysis / 'summaries/manifest.json').read_text())['valid_protocols']
manifest, days, counts, valid, modern = prepare(source, allowlist)
assert manifest['revision'] == 'cbbf32b5f6bf8f3d73fcc9f76ba91166644fd0c3'
assert manifest['script_sha256'] == sha(analysis / 'scripts/client_protocol_current_remote.py')
assert validate_partial(manifest, days) is None
assert days[0] == '2026-07-28' and days[-1] == '2026-09-15' and len(days) == 50
series = series_data(days, counts, valid, modern, None)
summary = comparisons(series, MINIMUM)
# Independently checked integer KPIs from the publication verification report.
expected = [(199797,393660),(68118,71512),(40194,40408),(41687,41687),
            (26783,82112),(2507,16341),(0,1309)]
verified = json.loads((source / 'verification.json').read_text())
for s, (m, n) in zip(summary, expected, strict=True):
    assert (s['last7_modern_calls'], s['last7_valid_calls']) == (m, n)
    assert s['last7_share'] == m/n
    v = verified['metrics']['Overall' if s['client'] == 'overall' else s['client']]
    for prefix, key in [('current', 'sep15'), ('previous', 'sep14'), ('last7', 'sep9_15')]:
        for field in ('share', 'modern_calls', 'valid_calls'):
            assert s[prefix+'_'+field] == v[key][field]
daily = [dict(client=s['id'], date=r['date'], partial=r['partial'],
              daily_share=r['daily_share'], rolling_share=r['rolling_share'],
              rolling_start=r['rolling_start'], rolling_end=r['rolling_end'], **r['counts'])
         for s in series for r in s['rows']]
def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
for name, rows in [('daily.csv', daily), ('day_on_day.csv', summary)]:
    with (ROOT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
metadata = dict(source_revision=manifest['revision'], window=manifest['window'],
                partial_day=None, snapshot_publication_utc=manifest['snapshot_last_modified_utc'])
write_json('stats.json', dict(**metadata, metrics=summary, minimum_display_valid_calls=MINIMUM))
write_json('provenance.json', dict(**metadata,
    source='summaries/client-protocol-20260916-v1', source_manifest=manifest,
    source_manifest_sha256=sha(source/'manifest.json'), complete_sha256=sha(source/'COMPLETE'),
    input_hashes=manifest['outputs'], verified_inputs_sha256=PINS,
    adapter_sha256=sha(__file__), valid_protocols=allowlist,
    metric='2026-07-28 protocol calls / reviewed-valid protocol calls; count weighted; fixed five excluded; missing hashes retained',
    comparison='Completed Sep15 versus Sep14; last7 is Sep9–15 inclusive.',
    coverage_limits=manifest.get('coverage_limits'),
    outputs_sha256={n:sha(ROOT/n) for n in ('daily.csv','day_on_day.csv','stats.json')}))

def read_csv(name):
    def cell(key, value):
        if value == '':
            return None
        if key == 'partial':
            assert value in ('True', 'False')
            return value == 'True'
        if key in ('client', 'label') or key.endswith(('date', 'start', 'end')):
            return value
        return float(value) if 'share' in key or key == 'change_pp' else int(value)
    with (ROOT / name).open(newline='', encoding='utf-8') as f:
        return [{k: cell(k, v) for k, v in r.items()} for r in csv.DictReader(f)]


provenance = json.loads((ROOT / 'provenance.json').read_text())
stats = json.loads((ROOT / 'stats.json').read_text())
# Verify source inputs consumed by the deck; source renderings are not imported.
for name in ('daily.csv', 'day_on_day.csv', 'stats.json'):
    digest = provenance['outputs_sha256'][name]
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
assert stats['window'] == provenance['window'] == ['2026-07-28', '2026-09-15']
assert stats['partial_day'] is provenance['partial_day'] is None
assert stats['snapshot_publication_utc'] == provenance['snapshot_publication_utc']
assert stats['source_revision'] == provenance['source_revision']
minimum = stats['minimum_display_valid_calls']
assert minimum == 100
start, end = map(date.fromisoformat, stats['window'])
dates = [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]
daily, summary = read_csv('daily.csv'), read_csv('day_on_day.csv')
assert summary == stats['metrics']
assert [s['client'] for s in summary] == ['overall', *ORDER]
assert set(r['client'] for r in daily) == {'overall', *ORDER}
series = []
for s in summary:
    rows = sorted((r for r in daily if r['client'] == s['client']), key=lambda r: r['date'])
    assert [r['date'] for r in rows] == dates
    normalized = []
    for r in rows:
        assert r['partial'] == (r['date'] == stats['partial_day'])
        rolling_end = date.fromisoformat(r['date']) - timedelta(days=r['partial'])
        rolling_start = rolling_end - timedelta(days=6)
        assert r['rolling_start'] == rolling_start.isoformat()
        assert r['rolling_end'] == rolling_end.isoformat()
        complete = rolling_start >= start
        for field in ('daily_share', 'rolling_share'):
            assert r[field] is None or 0 <= r[field] <= 1
        if not complete:
            assert r['rolling_share'] is None and r['rolling_valid_calls'] is None
        # Copy supplied rates; suppress only as the source contact-sheet renderer does.
        normalized.append({**r,
            'daily_share': r['daily_share'] if r['valid_calls'] >= minimum else None,
            'trailing7_share': r['rolling_share'] if complete and (r['rolling_valid_calls'] or 0) >= minimum else None})
    assert [s['last7_start'], s['last7_end']] == ['2026-09-09', '2026-09-15']
    assert s['last7_share'] == normalized[-1]['trailing7_share']
    assert s['current_share'] == normalized[-1]['daily_share']
    assert s['previous_share'] == normalized[-2]['daily_share']
    assert s['last7_valid_calls'] == rows[-1]['rolling_valid_calls']
    assert s['last7_modern_calls'] == rows[-1]['rolling_modern_calls']
    series.append({'id': s['client'], 'summary': s, 'rows': normalized})
output = {'window': stats['window'], 'partial_day': stats['partial_day'],
          'snapshot_publication_utc': stats['snapshot_publication_utc'],
          'minimum_display_valid_calls': minimum,
          'last7_window': ['2026-09-09', '2026-09-15'],
          'dates': dates, 'overall': series[0], 'series': series[1:]}
(ROOT / 'normalized.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print('Validated consumed source data hashes, seven identities, 50 dates/client, completed rolling windows, threshold and exact supplied KPIs.')
