"""Read the audited expert NPZ and collate candidate sets without truncation."""
import numpy as np


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in
                ('candidates', 'states', 'actions', 'offsets', 'instances', 'split')}


def batch(data, ids, device='cpu'):
    import torch
    ids = np.asarray(ids, dtype=int)
    lengths = data['offsets'][ids + 1] - data['offsets'][ids]
    maximum = int(lengths.max())
    x = np.zeros((len(ids), maximum, 25), np.float32)
    mask = np.arange(maximum)[None, :] >= lengths[:, None]
    for i, j in enumerate(ids):
        x[i, :lengths[i]] = data['candidates'][data['offsets'][j]:data['offsets'][j + 1]]
    state = data['states'][ids]
    return tuple(torch.as_tensor(a, device=device) for a in
                 (x, state[:, :8], state[:, 8:], mask, data['actions'][ids]))
