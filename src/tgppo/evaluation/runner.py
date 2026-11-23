"""Evaluate fixed Actor checkpoints and relpscost together, by instance and seed."""
from pathlib import Path
import re

from tgppo.config import fingerprint_inputs, read_config
from tgppo.paths import output_root
from tgppo.solver.protocol import fingerprint, sha256
from tgppo.solver.stack import preflight
from tgppo.training.budget import lock
from tgppo.training.runner import make_job, read, write


def evaluation_dir(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Invalid evaluation name')
    return output_root() / 'evaluation' / name


def model_records(c, seed, episode, actors):
    from tgppo.runtime import read_checkpoint
    from tgppo.training.runner import VARIANTS
    records = []
    for variant in VARIANTS:
        path = Path(actors.get(variant, output_root() / 'training' / (
            '{}-s{}'.format(variant, seed)) / 'actor-{:04d}.pt'.format(episode))).resolve()
        digest = sha256(path)
        model = read_checkpoint(path, allow_actor_only=True)
        cfg = model['config']
        progress = model.get('provenance', {}).get('progress', {})
        kind = 'b0' if variant == 'b0' else 'mlp'
        if (model.get('schema_version') != 'actor-only-v1' or
                (cfg.get('variant'), cfg.get('actor_kind'), cfg.get('server_training_seed'),
                 cfg.get('task_time_limit'), progress.get('next_episode'), progress.get('phase')) !=
                (variant, kind, seed, c['time_limit'], episode, 'ready') or
                not all(x.isfinite().all().item() for x in model['actor'].values())):
            raise ValueError('Expected matching fixed {} Actor: {}'.format(variant, path))
        if sha256(path) != digest:
            raise ValueError('Actor changed while being read: ' + str(path))
        records.append(dict(label=variant, path=str(path), sha256=digest))
    return records


def run(c, name, seed, episode, actors, shard, workers, resume):
    from tgppo.runtime import run_jobs
    if not 0 <= shard < c['evaluation_shards'] or not 1 <= workers:
        raise ValueError('Invalid shard or worker count')
    models = model_records(c, seed, episode, actors)
    out = evaluation_dir(name)
    info = preflight(c, 'cpu', workers)
    suite = dict(training_seed=seed, episode=episode, models=models,
                 inputs=fingerprint_inputs(c, 'test'), test=c['test'],
                 solver_seeds=c['solver_seeds'], labels=['relpscost'] + [x['label'] for x in models],
                 time_limit=c['time_limit'], shards=c['evaluation_shards'])
    with lock(out / 'suite.lock'):
        suite_file = out / 'suite.json'
        if suite_file.exists() and read(suite_file) != suite:
            raise ValueError('Frozen evaluation inputs changed; use a new --name')
        write(suite_file, suite)
    shard_dir = out / ('shard-{}'.format(shard))
    with lock(shard_dir / 'execution.lock', blocking=False):
        for name in c['test'][shard::c['evaluation_shards']]:
            limit = c['evaluation_large_memory_bytes'] if name == 'mspp16' else c['evaluation_memory_bytes']
            active_workers = 1 if name == 'mspp16' else workers
            cohort = dict(stack=info['stack'], host=info['host'], cores=info['cores'], workers=active_workers)
            for solver_seed in c['solver_seeds']:
                panel = shard_dir / name / str(solver_seed)
                result = panel / 'panel.json'
                if result.exists():
                    continue
                work = panel / 'cohorts' / fingerprint(cohort)[:24]
                if panel.exists() and not resume:
                    raise ValueError('Incomplete panel retained; use --resume')
                for model in models:
                    if sha256(model['path']) != model['sha256']:
                        raise ValueError('Frozen Actor changed: ' + model['label'])
                jobs = []
                labels = [('relpscost', None)] + [(m['label'], m['path']) for m in models]
                offset = solver_seed % len(labels)
                for label, checkpoint in labels[offset:] + labels[:offset]:
                    mode = 'eval' if checkpoint else 'native'
                    job = make_job(c, name, solver_seed, seed if checkpoint else 0, mode,
                                   'eval-{}-{}-s{}'.format(label, name, solver_seed), checkpoint)
                    job.update(address_space_limit_bytes=limit)
                    jobs.append((label, job))
                results = {}
                attempts = sorted(work.glob('attempt-*'))
                for attempt in attempts:
                    for _, job in jobs:
                        path = attempt / (job['job_id'] + '.json')
                        if path.exists():
                            row = read(path)
                            if valid_row(row, job, c):
                                results.setdefault(job['job_id'], row)
                missing = [j for _, j in jobs if j['job_id'] not in results]
                if missing:
                    attempt = work / 'attempt-{:04d}'.format(len(attempts))
                    write(attempt / 'environment.json', cohort)
                    rows = run_jobs(missing, attempt, active_workers, c['watchdog'])
                    if any(not valid_row(row, job, c) for row, job in
                           zip(rows, sorted(missing, key=lambda j: j['job_id']))):
                        raise RuntimeError('Invalid evaluation task retained: ' + str(attempt))
                    results.update((r['job_id'], r) for r in rows)
                write(result, dict(instance=name, solver_seed=solver_seed, cohort=cohort,
                    rows={label: results[j['job_id']] for label, j in jobs}))
                print('Completed {} solver seed {}'.format(name, solver_seed), flush=True)


def valid_row(row, job, c):
    return (row.get('valid') is True and row.get('task_terminal') is True and
            all(row.get(k) == job[k] for k in ('job_id', 'mode', 'instance_name', 'solver_seed', 'sampling_seed',
                                               'address_space_limit_bytes')) and
            row.get('pdi_horizon_seconds') == c['time_limit'] and
            row.get('worker_cpu_complete') and not row.get('cuda_initialized') and
            not row.get('fallback_action_count', 0) and
            (job['mode'] == 'native' or row.get('callback_seconds', {}).get('critic') == 0))


def summarize(name):
    from tgppo.evaluation.metrics import aggregate
    out = evaluation_dir(name)
    suite = read(out / 'suite.json')
    panels = []
    for index, instance in enumerate(suite['test']):
        for seed in suite['solver_seeds']:
            path = out / ('shard-{}'.format(index % suite['shards'])) / instance / str(seed) / 'panel.json'
            if path.exists():
                panel = read(path)
                if (panel['instance'], panel['solver_seed'], set(panel['rows'])) != (
                        instance, seed, set(suite['labels'])):
                    raise ValueError('Invalid panel: ' + str(path))
                panels.append(panel)
    expected = len(suite['test']) * len(suite['solver_seeds'])
    result = dict(complete=len(panels) == expected, complete_panels=len(panels), expected_panels=expected)
    if result['complete']:
        result.update(aggregate(panels, suite, read_config('benchmark_subsets.json')))
        write(out / 'summary.json', result)
    return result
