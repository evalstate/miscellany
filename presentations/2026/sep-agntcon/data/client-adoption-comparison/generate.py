"""Normalize supplied CSV cells, never recompute rates or denominators.
Run: python data/client-adoption-comparison/generate.py
Original source artifacts remain byte-for-byte unchanged.
"""
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = ['claude-code', 'Anthropic/ClaudeAI', 'chat-ui-mcp', 'openai-mcp',
         'openai-mcp (Codex)', 'codex-mcp-client']


def read_csv(name):
    with (ROOT / name).open(newline='', encoding='utf-8') as f:
        return [{key: (value if key in ('date', 'client') else
                       None if value == '' else
                       float(value) if 'share' in key else int(value))
                 for key, value in row.items()} for row in csv.DictReader(f)]


provenance = json.loads((ROOT / 'provenance.json').read_text())
stats = json.loads((ROOT / 'stats.json').read_text())
assert provenance['focus_clients'] == ORDER
for name in ['daily.csv', 'summary.csv', 'stats.json', 'README.md']:
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == provenance['output_hashes'][name]
assert stats['window'] == ['2026-07-28', '2026-09-14']
start, end = map(date.fromisoformat, stats['window'])
dates = [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]
daily, summary = read_csv('daily.csv'), read_csv('summary.csv')
series = []
for client in ORDER:
    rows = sorted((r for r in daily if r['client'] == client), key=lambda r: r['date'])
    assert [r['date'] for r in rows] == dates
    s = next(r for r in summary if r['client'] == client)
    assert s == next(r for r in stats['clients'] if r['client'] == client)
    for r in rows:
        for field in ['daily_share', 'trailing7_share']:
            assert r[field] is None or 0 <= r[field] <= 1
        if r['valid_calls'] < 100:
            assert r['daily_share'] is None
        if r['trailing7_days'] < 7 or r['trailing7_valid_calls'] < 100:
            assert r['trailing7_share'] is None
    series.append({'id': client, 'summary': s, 'rows': rows})
assert [f"{s['summary']['last7_share'] * 100:.1f}" for s in series] == ['95.2', '99.3', '100.0', '28.4', '15.1', '0.0']
output = {'window': stats['window'], 'last7_window': ['2026-09-08', '2026-09-14'],
          'dates': dates, 'series': series}
(ROOT / 'normalized.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print('Validated source hashes, identities, 49 dates/client, supplied nulls and last7 KPIs.')
