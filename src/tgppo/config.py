"""Validate protocol and fingerprint inputs before starting an experiment."""
import json
import math
import re

from tgppo.paths import PROJECT_ROOT, data_paths
from tgppo.solver.protocol import sha256


def read_config(name):
    return json.loads((PROJECT_ROOT / 'configs' / name).read_text())


def protocol():
    c = read_config('protocol.json')
    if (c['schema'], c['experiment'], c['optimizer'], c['actor_update_mode'],
            c['gae_implementation'], c['batch_episodes']) != (
            'tgppo-alter-protocol-v1', 'tgppo-alter-v1', 'AdamW', 'train',
            'standard_episodic_v1', 1):
        raise ValueError('Expected single-episode AdamW / train-mode Actor / episodic GAE protocol')
    names = c['train'] + c['test']
    if not c['train'] or not c['test'] or len(set(names)) != len(names) or any(
            not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', n) for n in names):
        raise ValueError('Training and test instances must be unique, safe names')
    if not c['solver_seeds'] or len(set(c['solver_seeds'])) != len(c['solver_seeds']):
        raise ValueError('Provide distinct solver seeds')
    if c['episodes'] < 1 or c['milestones'] != sorted(set(c['milestones'])) or (
            c['milestones'][0], c['milestones'][-1]) != (0, c['episodes']):
        raise ValueError('Milestones must start at 0 and end at the final episode')
    if not 1 <= c['time_limit'] < c['watchdog'] or c['evaluation_shards'] < 1:
        raise ValueError('Invalid time limit, watchdog or shard count')
    return c


def fingerprint_inputs(c, split, include_expert=False):
    paths = data_paths()
    folder = paths[split + '_instances']
    cutoffs = read_config('cutoffs.json')
    result = {'instances': {}, 'configs': {}, 'sources': {}}
    for name in c[split]:
        cutoff = cutoffs.get(name, {})
        if cutoff.get('status') != 'optimal' or not cutoff.get('source') or not math.isfinite(cutoff.get('cutoff', math.nan)):
            raise ValueError('Missing verified optimal cutoff: ' + name)
        result['instances'][name] = sha256(folder / (name + '.mps.gz'))
    for name in ('protocol.json', 'hparams.json', 'cutoffs.json', 'benchmark_subsets.json'):
        result['configs'][name] = sha256(PROJECT_ROOT / 'configs' / name)
    for path in sorted((PROJECT_ROOT / 'src/tgppo').rglob('*.py')):
        result['sources'][str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    if include_expert:
        path = paths['expert_dataset']
        digest = sha256(path)
        if digest != c['expert_dataset_sha256']:
            raise ValueError('Expert dataset does not match the configured SHA256: ' + str(path))
        result['expert_dataset'] = digest
    return result
