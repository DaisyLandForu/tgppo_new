"""Aggregate paired, fixed-horizon results and the paper's easy/hard metrics."""
import math
from collections import defaultdict
from statistics import mean


def shifted_gmean(values, shift=0):
    if any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError('Invalid evaluation metric')
    if shift == 0 and 0 in values:
        return 0.0
    return max(0., math.exp(mean(math.log(v + shift) for v in values)) - shift)


def statistics(rows, horizon):
    proofs = [r['proof_completed'] and r['scip_solve_time'] <= horizon for r in rows]
    calls = sum(r.get('branch_count', 0) for r in rows)
    return dict(tasks=len(rows), proofs=sum(proofs), proof_rate=mean(proofs),
                mean_pdi=mean(r['pdi_horizon_midpoint'] for r in rows),
                mean_pdi_lower=mean(r['pdi_horizon_lower'] for r in rows),
                mean_pdi_upper=mean(r['pdi_horizon_upper'] for r in rows),
                sgm_pdi=shifted_gmean([r['pdi_horizon_midpoint'] for r in rows]),
                sgm_nodes_shift100=shifted_gmean([r['nnodes'] for r in rows], 100),
                mean_optimize_seconds=mean(r['optimize_wall_seconds'] for r in rows),
                par2=mean(r['scip_solve_time'] if ok else 2 * horizon for r, ok in zip(rows, proofs)),
                actor_calls=calls,
                actor_ms=1000 * sum(r.get('callback_seconds', {}).get('actor', 0) for r in rows) / calls
                if calls else None)


def aggregate(panels, suite, subsets):
    grouped = defaultdict(list)
    for panel in panels:
        grouped[panel['instance']].append(panel)
    labels, horizon = suite['labels'], suite['time_limit']
    by_instance = {
        name: {label: statistics([p['rows'][label] for p in group], horizon) for label in labels}
        for name, group in grouped.items()}
    overall = {label: statistics([p['rows'][label] for p in panels], horizon) for label in labels}
    comparisons = [('mlp', 'b0'), ('mlp_bc', 'mlp'), ('mlp_bc', 'b0')]
    comparisons += [(model, 'relpscost') for model in labels if model != 'relpscost']
    effects = {}
    for model, reference in comparisons:
        a, b = overall[model], overall[reference]
        effects[model + '_vs_' + reference] = dict(
            mean_pdi_difference=a['mean_pdi'] - b['mean_pdi'],
            mean_pdi_difference_lower=a['mean_pdi_lower'] - b['mean_pdi_upper'],
            mean_pdi_difference_upper=a['mean_pdi_upper'] - b['mean_pdi_lower'],
            proof_rate_difference=a['proof_rate'] - b['proof_rate'])
    paper = {}
    for split, metric in [('easy', 'sgm_nodes_shift100'), ('hard', 'sgm_pdi')]:
        names = set(subsets[split]) & set(grouped)
        if not names:
            continue
        paper[split] = dict(instances=len(names), metric=metric,
            aggregate={label: statistics([p['rows'][label] for p in panels if p['instance'] in names],
                                         horizon)[metric] for label in labels},
            strict_wins={model + '_vs_' + ref: sum(by_instance[n][model][metric] < by_instance[n][ref][metric]
                                                   for n in names)
                         for model, ref in comparisons})
    return dict(overall=overall, effects=effects, by_instance=by_instance, paper_subsets=paper)
