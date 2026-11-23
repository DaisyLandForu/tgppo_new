"""Canonical complete-episode trainer, copied from the verified research protocol."""
import hashlib
import json
from pathlib import Path
import time
import uuid
from tgppo.paths import PROJECT_ROOT as ROOT, data_paths, output_root
from tgppo.training.budget import budget, lock
from tgppo.solver.stack import preflight
RUN = output_root()
VARIANTS = ('b0', 'mlp', 'mlp_bc')
SEEDS = (11, 22)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def schedule(seed, c):
    import numpy as np
    rng = np.random.RandomState(seed)
    pairs = [(n, s) for n in c['train'] for s in c['solver_seeds']]
    result = []
    while len(result) < c['episodes']:
        result.extend(pairs[i] for i in rng.permutation(len(pairs)))
    return result[:c['episodes']]


def configuration(c, variant, seed, device=None):
    from tgppo.runtime import config_with_protocol
    device = device or c['learner_device']
    cfg = read(ROOT / 'configs/hparams.json')
    for key in ('optimizer', 'optimizer_betas', 'optimizer_eps', 'weight_decay',
                'actor_width', 'inducing_points', 'expert_dataset_sha256', 'bc_steps',
                'bc_batch_size', 'bc_lr', 'bc_selection_milestones', 'attention_cell_budget'):
        cfg[key] = c[key]
    for key in ('actor_update_mode', 'gae_implementation', 'update_organization', 'alignment_choice'):
        cfg[key] = c[key]
    cfg.update(learner_device=device, actor_kind='b0' if variant == 'b0' else 'mlp',
               server_training_seed=seed, task_time_limit=c['time_limit'], variant=variant,
               experiment=c['experiment'], experiment_protocol_sha256=fp(c),
               expert_pretraining='bc' if variant == 'mlp_bc' else 'none')
    return config_with_protocol(cfg)


def state_digest(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cached_job(job, folder, identity, c, resume=False):
    from tgppo.runtime import run_jobs
    folder = Path(folder)
    with lock(folder / 'job.lock'):
        path = folder / 'identity.json'
        if path.exists() and read(path) != identity:
            raise ValueError('Cached task identity changed: ' + str(folder))
        write(path, identity)
        attempts = sorted(folder.glob('attempt-*'))
        for a in reversed(attempts):
            f = a / (job['job_id'] + '.json')
            if not f.exists():
                continue
            row = read(f)
            if row.get('valid') and (job['mode'] != 'collect' or (a / (job['job_id']+'.pkl')).exists()):
                return row, a
        if attempts and not resume:
            raise ValueError('Previous failed/incomplete task retained; use --resume: ' + str(folder))
        attempt = folder / ('attempt-{:03d}'.format(len(attempts)))
        write(attempt / 'job.json', job)
        rows = run_jobs([job], attempt, 1, c['watchdog'])
        if len(rows) != 1 or not rows[0].get('valid'):
            raise RuntimeError('Invalid task retained; no automatic retry: ' + str(attempt))
        return rows[0], attempt


def make_job(c, name, solver_seed, seed, mode, job_id, checkpoint=None, baseline=None):
    from tgppo.runtime import make_job as build
    directory = data_paths()['train_instances' if name in c['train'] else 'test_instances']
    job = build(job_id, directory, name, c['time_limit'], solver_seed, seed,
                read(ROOT / 'configs/cutoffs.json'), mode, checkpoint, baseline)
    # Actor construction, BC, policy length and model label do not consume action RNG.
    from tgppo.runtime import action_seed
    job['sampling_seed'] = action_seed(seed, job_id if mode == 'collect' else 'paper-test-{}-s{}'.format(name, solver_seed))
    job['address_space_limit_bytes'] = c['worker_address_space_bytes']
    return job


def native(c, name, solver_seed, info, resume=False):
    from tgppo.runtime import job_identity
    from tgppo.solver.protocol import validate_baseline
    job = make_job(c, name, solver_seed, 0, 'native', 'native-'+name+'-s'+str(solver_seed))
    ident = job_identity(job)
    signature = dict(identity=ident, stack=info['stack'])
    folder = RUN / 'normalization' / fp(info['stack']) / fp(ident)
    with lock(folder / 'baseline.lock'):
        file = folder / 'baseline.json'
        if file.exists():
            bundle = read(file)
            if bundle['signature'] != signature:
                raise ValueError('Normalization identity mismatch')
        else:
            row, _ = cached_job(job, folder / 'task', signature, c, resume)
            bundle = dict(signature=signature, identity=ident, result=row,
                          host=info['host'], horizon_seconds=c['time_limit'])
            write(file, bundle)
        validate_baseline(bundle, ident)
    return bundle


def ready_save(out, agent, cfg, progress, c):
    from tgppo.runtime import save_checkpoint, export_actor
    n = progress['next_episode']
    if n in c['milestones']:
        cp = out / ('episode-{:04d}.pt'.format(n))
        save_checkpoint(cp, agent, cfg, progress)
        export_actor(cp, out / ('actor-{:04d}.pt'.format(n)))
    save_checkpoint(out / 'latest.pt', agent, cfg, progress)


def initial(c, variant, seed, out, device=None):
    import torch
    from tgppo import runtime as rt
    device = device or c['learner_device']
    cfg = configuration(c, variant, seed, device)
    rt.seed_all(seed)
    agent = rt.build_agent(cfg)
    rng = rt.rng_state()
    before = state_digest(agent.actor_network.state_dict())
    if variant == 'mlp_bc':
        from tgppo.training.pretrain import fit
        try:
            fit(agent.actor_network, cfg, out / 'bc', device=device, resume=(out/'bc/latest.pt').exists())
        finally:
            rt.restore_rng(rng)
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
    init = dict(actor_before_bc=before, actor=state_digest(agent.actor_network.state_dict()),
                critic=state_digest(agent.critic_network.state_dict()),
                cpu_rng=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda_rng=([hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()]
                          if device.startswith('cuda') else []),
                schedule_sha256=fp(schedule(seed, c)), configuration_sha256=fp(cfg),
                hardware=read(out/'environment.json'))
    common = {k: init[k] for k in ('critic', 'cpu_rng', 'cuda_rng', 'schedule_sha256')}
    with lock(RUN/'pairing'/('s'+str(seed)+'.lock')):
        paired = RUN/'pairing'/('s'+str(seed)+'.json')
        record = read(paired) if paired.exists() else dict(common=common, initializations={})
        if record['common'] != common:
            raise ValueError('Actual cross-job Critic/RNG/schedule initialization differs')
        for other, entry in record['initializations'].items():
            if variant != 'b0' and other != 'b0' and entry['actor_before_bc'] != before:
                raise ValueError('Actual MLP/MLP+BC pre-BC initialization differs')
        record['initializations'][variant] = init
        write(paired, record)
    write(out / 'initialization.json', init)
    progress = dict(next_episode=0, phase='ready', experiment_protocol_sha256=fp(c))
    ready_save(out, agent, cfg, progress, c)
    return agent, cfg, progress


def train(c, variant, seed, resume, device=None):
    import torch
    from tgppo import runtime as rt
    device = device or c['learner_device']
    out = RUN / 'training' / ('{}-s{}'.format(variant, seed))
    if (out / 'summary.json').exists() and read(out/'summary.json').get('training_complete'):
        assert rt.read_checkpoint(out/'latest.pt')['progress']['next_episode'] == c['episodes']
        print('Already complete; no repeated training.', flush=True)
        return
    with budget(out, c['job_budget_seconds'], resume) as end:
        input_file = out / 'inputs.json'
        from tgppo.config import fingerprint_inputs
        identity = fingerprint_inputs(c, 'train', include_expert=variant == 'mlp_bc')
        if input_file.exists() and read(input_file) != identity:
            raise ValueError('Source, configuration or training inputs changed; choose a new TGPPO_OUTPUT_DIR')
        write(input_file, identity)
        info = preflight(c, device)
        write(out / ('environment-attempt-'+str(len(read(out/'budget.json')['attempts']))+'.json'), info)
        write(out / 'environment.json', info)
        previous = read(out/'state.json') if (out/'state.json').exists() else {'next_episode': 0}
        write(out/'state.json', dict(previous, status='target_device_checks', learner_device=device))
        # Actual target-device bridge is mandatory, synthetic only, before any solve.
        from tgppo.training.checks import target_checks
        gate = out / 'target-checks' / fp(dict(stack=info['stack'], gpu=info['gpu'], device=device))[:20]
        target_checks(c, seed, gate, device)
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        if (out/'latest.pt').exists():
            if not resume:
                raise ValueError('Existing checkpoint requires --resume')
            agent, payload = rt.load_agent(out/'latest.pt', device=device)
            cfg, progress = payload['config'], payload['progress']
            if cfg != configuration(c, variant, seed, device):
                raise ValueError('Resume config changed')
        else:
            write(out/'state.json', dict(status='initialization_or_bc', next_episode=0))
            agent, cfg, progress = initial(c, variant, seed, out, device)
        pending = out / 'pending.pt'
        plan = schedule(seed, c)
        write(out / 'schedule.json', [dict(episode=i+1, instance=n, solver_seed=s) for i, (n, s) in enumerate(plan)])
        write(out / 'state.json', dict(status='running', **progress))
        for i in range(progress['next_episode'], c['episodes']):
            # Never shorten a solve just to fit the allocation. Keep a PPO reserve.
            if end-time.time() < c['watchdog'] + 900:
                write(out/'state.json', dict(status='stopped_before_next_complete_episode', **progress))
                return
            if pending.exists():
                pp = rt.read_checkpoint(pending)
                if pp['progress']['next_episode'] == i:
                    if pp['config'] != cfg or pp['progress']['phase'] != 'update_pending':
                        raise ValueError('Wrong pending checkpoint')
                    agent, pp = rt.load_agent(pending, device=device)
                    task_result = pp['progress']['task_result']
                    collection_directory = Path(pp['progress']['collection_directory']).resolve()
                    if out.resolve() not in collection_directory.parents:
                        raise ValueError('Pending trajectory must belong to this run')
                else:
                    if pp['progress']['next_episode'] > i:
                        raise ValueError('Pending checkpoint ahead of committed state')
                    pending.unlink()
                    pp = None
            else:
                pp = None
            if pp is None:
                name, ss = plan[i]
                write(out/'state.json', dict(status='normalization', next_episode=i, instance=name, solver_seed=ss))
                base = native(c, name, ss, info, resume)
                if end-time.time() < c['watchdog'] + 900:
                    write(out/'state.json', dict(status='stopped_after_normalization', **progress))
                    return
                jid = 'collect-e{:05d}-{}-s{}'.format(i+1, name, ss)
                job = make_job(c, name, ss, seed, 'collect', jid, out/'latest.pt', base)
                identity = dict(checkpoint_sha256=sha(out/'latest.pt'), task=rt.job_identity(job),
                                stack=info['stack'], sampling_seed=job['sampling_seed'])
                write(out/'state.json', dict(status='collecting', next_episode=i, instance=name, solver_seed=ss))
                task_result, collection_directory = cached_job(job,
                    out/'episodes'/str(i+1)/fp(identity)[:24], identity, c, resume)
                if len(agent.memory):
                    raise ValueError('Cross-episode buffer is forbidden')
                rt.import_trajectories(agent, [task_result], collection_directory)
                if len(agent.memory.episodes) != 1:
                    raise ValueError('Exactly one complete episode required')
                pp = dict(next_episode=i, phase='update_pending', task_result=task_result,
                          collection_directory=str(collection_directory), experiment_protocol_sha256=fp(c))
                rt.save_checkpoint(pending, agent, cfg, pp)
            rt.seed_all(rt.action_seed(seed, 'paper-ppo-update-e{:05d}'.format(i+1)))
            start = time.monotonic()
            def heartbeat(value):
                write(out/'state.json', dict(status='updating', next_episode=i, **value))
                print(json.dumps(dict(episode=i+1, phase='updating', **value)), flush=True)
            update = agent.learn(microbatch_size=c['microbatch_size'], progress_callback=heartbeat)
            assert len(agent.memory) == 0 and agent.memory.active is None
            update.update(episode=i+1, instance=plan[i][0], solver_seed=plan[i][1],
                          learner_wall_seconds=time.monotonic()-start,
                          phase='complete_episode_update', actor_sha256=state_digest(agent.actor_network.state_dict()))
            progress = dict(next_episode=i+1, phase='ready', experiment_protocol_sha256=fp(c))
            # Commit state first. A kill during learning replays pending, a kill
            # after this save resumes at the following episode with no double step.
            write(out/'updates'/('episode-{:04d}.json'.format(i+1)), update)
            ready_save(out, agent, cfg, progress, c)
            pending.unlink()
            for p in collection_directory.glob('*.pkl'):
                p.unlink()  # Only this newly committed trajectory; old runs untouched.
            write(out/'state.json', dict(status='running', **progress))
            print(json.dumps(update, ensure_ascii=False), flush=True)
        summary = dict(training_complete=True, episodes=c['episodes'], next_episode=progress['next_episode'],
                       model=variant, training_seed=seed, selection='fixed_final_episode',
                       checkpoint_sha256=sha(out/('episode-{:04d}.pt'.format(c['episodes']))), actor_sha256=sha(out/('actor-{:04d}.pt'.format(c['episodes']))))
        write(out/'summary.json', summary)
        write(out/'state.json', dict(status='training_complete', **progress))
        print(json.dumps(summary), flush=True)


def normalize(c, shard, shards, resume):
    if not 0 <= shard < shards:
        raise ValueError('Invalid normalization shard')
    out = RUN/'normalization-jobs'/('shard-{}-of-{}'.format(shard, shards))
    with budget(out, c['job_budget_seconds'], resume) as end:
        # Optional prefill is sequential; request one CPU rather than four.
        info = preflight(c, 'cpu', workers=1)
        for i, (name, ss) in enumerate((n, s) for n in c['train'] for s in c['solver_seeds']):
            if i % shards != shard:
                continue
            if end-time.time() < c['watchdog']+60:
                return
            native(c, name, ss, info, resume)
            print(json.dumps(dict(normalized=name, solver_seed=ss)), flush=True)


def status(c):
    result = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            folder = RUN/'training'/('{}-s{}'.format(variant, seed))
            row = read(folder/'state.json') if (folder/'state.json').exists() else dict(status='not_started', next_episode=0)
            if (folder/'summary.json').exists():
                row['summary'] = read(folder/'summary.json')
            if (folder/'budget.json').exists():
                row['budget'] = read(folder/'budget.json')
            result[variant+'-s'+str(seed)] = row
    return result
