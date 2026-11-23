"""Command line for paper-aligned TGPPO, MLP, and MLP+BC."""
import argparse
import json
from pathlib import Path
import tempfile

from tgppo.config import protocol, fingerprint_inputs


VARIANTS = ('b0', 'mlp', 'mlp_bc')


def print_json(value):
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='validate environment; no SCIP solves')
    check.add_argument('--data', action='store_true', help='also verify benchmark and expert inputs')
    check.add_argument('--bridge', action='store_true', help='verify synthetic CPU PPO update and exact replay')
    train = sub.add_parser('train', help='train B0, MLP, or MLP+BC')
    train.add_argument('--variant', choices=VARIANTS, required=True)
    train.add_argument('--seed', type=int, default=11)
    train.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    train.add_argument('--resume', action='store_true')
    norm = sub.add_parser('normalize', help='prefill relpscost reward baselines')
    norm.add_argument('--shard', type=int, default=0)
    norm.add_argument('--shards', type=int, default=1)
    norm.add_argument('--resume', action='store_true')
    sub.add_parser('status', help='show local training progress')
    evaluate = sub.add_parser('evaluate', help='run a CPU shard with fixed Actor checkpoints')
    evaluate.add_argument('--name', default='fixed-final')
    evaluate.add_argument('--seed', type=int, default=11)
    evaluate.add_argument('--episode', type=int, help='fixed checkpoint episode (default: final)')
    evaluate.add_argument('--shard', type=int, required=True)
    evaluate.add_argument('--workers', type=int)
    evaluate.add_argument('--resume', action='store_true')
    for variant in VARIANTS:
        evaluate.add_argument('--' + variant.replace('_', '-') + '-actor', type=Path,
                              help='optional explicit Actor export from a trusted run')
    summary = sub.add_parser('summarize', help='summarize completed evaluation panels')
    summary.add_argument('--name', default='fixed-final')
    args = parser.parse_args()
    c = protocol()

    if args.command == 'status':
        from tgppo.training.runner import status
        print_json(status(c))
    elif args.command == 'summarize':
        from tgppo.evaluation.runner import summarize
        print_json(summarize(args.name))
    elif args.command == 'check':
        from tgppo.solver.stack import stack
        configure_torch()
        result = dict(stack=stack())
        if args.data:
            result['data'] = {split: len(fingerprint_inputs(c, split, split == 'train')['instances'])
                              for split in ('train', 'test')}
        if args.bridge:
            from tgppo.training.checks import target_checks
            with tempfile.TemporaryDirectory(prefix='tgppo-check-') as folder:
                target_checks(c, c['training_seeds'][0], folder, 'cpu')
            result['synthetic_update_and_replay'] = 'passed'
        print_json(result)
    elif args.command == 'train':
        from tgppo.training.runner import train as run
        if args.seed not in c['training_seeds']:
            parser.error('--seed must be listed in configs/protocol.json')
        configure_torch()
        run(c, args.variant, args.seed, args.resume, args.device)
    elif args.command == 'normalize':
        from tgppo.training.runner import normalize
        configure_torch()
        normalize(c, args.shard, args.shards, args.resume)
    elif args.command == 'evaluate':
        from tgppo.evaluation.runner import run
        configure_torch()
        actors = {v: getattr(args, v + '_actor') for v in VARIANTS if getattr(args, v + '_actor')}
        run(c, args.name, args.seed, args.episode if args.episode is not None else c['episodes'],
            actors, args.shard, args.workers if args.workers is not None else c['evaluation_workers'],
            args.resume)


if __name__ == '__main__':
    main()
