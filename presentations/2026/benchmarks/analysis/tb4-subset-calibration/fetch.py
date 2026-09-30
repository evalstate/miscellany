"""Fetch every trial associated with every Terminal-Bench 4.0 leaderboard row.

Writes data/rows/<row_id>.json  = {row, trial_ids, trials:[job-trial records]}
"""
import json, subprocess, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

D = pathlib.Path(__file__).parent / 'data'
(D / 'rows').mkdir(parents=True, exist_ok=True)
(D / 'jobs').mkdir(exist_ok=True)


def hj(*args):
    out = subprocess.run(['harbor', 'hub', *args, '--json'], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def paged(*args, limit=1000):
    items, page = [], 1
    while True:
        r = hj(*args, '--limit', str(limit), '--page', str(page))
        items += r['items']
        tp = r.get('total_pages') or (1 if len(r['items']) < limit else page + 1)
        if page >= tp or not r['items']:
            return items
        page += 1


def job_trials(job):
    f = D / 'jobs' / f'{job}.json'
    if f.exists():
        return json.loads(f.read_text())
    items = paged('job', 'trials', job, '--include-retries')
    f.write_text(json.dumps(items))
    return items


def do_row(row):
    f = D / 'rows' / f"{row['id']}.json"
    if f.exists() and '--force' not in sys.argv:
        return f.read_text() and row['id']
    ids = [t['trial_id'] for t in paged('leaderboard', 'row', 'trial', 'list', row['id'])]
    have, jobs = {}, []
    missing = list(ids)
    while missing:
        job = hj('trial', 'show', missing[0])['job_id']
        if job in jobs:
            raise RuntimeError(f'trial {missing[0]} not listed in its own job {job}')
        jobs.append(job)
        for t in job_trials(job):
            have[t['id']] = t
        missing = [i for i in ids if i not in have]
    f.write_text(json.dumps(dict(row=row, jobs=jobs, trial_ids=ids, trials=[have[i] for i in ids])))
    return row['id']


if __name__ == '__main__':
    lb = json.loads((D / 'leaderboard.json').read_text())
    with ThreadPoolExecutor(8) as ex:
        for rid in ex.map(do_row, lb['rows']):
            print('ok', rid, flush=True)
