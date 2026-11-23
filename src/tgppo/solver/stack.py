"""Verify the custom SCIP feature binding and record the actual execution stack."""
import importlib.util
import os
import platform
import socket
import sys
from pathlib import Path

from tgppo.paths import data_paths
from tgppo.solver.protocol import sha256


def stack():
    import torch
    from pyscipopt import Model
    required = ('getCandsState', 'getMIPState', 'getNodeState', 'getIntegralSnapshot')
    missing = [m for m in required if not hasattr(Model, m)]
    if missing:
        raise RuntimeError('Custom PySCIPOpt methods missing: ' + ', '.join(missing))
    with_model = Model()
    try:
        version = with_model.version()
    finally:
        with_model.freeProb()
    if abs(version - 6.01) > 1e-9:
        raise RuntimeError('Expected SCIP 6.0.1, got {}'.format(version))
    prefix = data_paths().get('scip_prefix')
    if prefix is None:
        raise ValueError('Set SCIPOPTDIR or configs/paths.local.json:scip_prefix')
    lib = prefix / 'lib/libscip.so'
    binding = importlib.util.find_spec('pyscipopt.scip').origin
    cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')), platform.machine())
    return dict(cpu_model=cpu, machine=platform.machine(), scip_version=version,
                libscip_sha256=sha256(lib), binding_sha256=sha256(binding),
                torch_version=str(torch.__version__), torch_cuda_runtime=torch.version.cuda,
                python_abi='{}.{}'.format(*sys.version_info[:2]))


def preflight(c, device, workers=None):
    import torch
    torch.set_num_threads(1)
    info = stack()
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    cores = sorted(os.sched_getaffinity(0))
    quotas, memory = [], []
    base = Path('/sys/fs/cgroup')
    if (base / 'cpu.max').exists():
        quota, period = (base / 'cpu.max').read_text().split()
        if quota != 'max':
            quotas.append(int(quota) / int(period))
    for group in ('cpu', 'cpu,cpuacct'):
        quota = base / group / 'cpu.cfs_quota_us'
        if quota.exists() and int(quota.read_text()) > 0:
            quotas.append(int(quota.read_text()) / int((quota.parent / 'cpu.cfs_period_us').read_text()))
    for path in (base / 'memory.max', base / 'memory/memory.limit_in_bytes'):
        if path.exists() and path.read_text().strip() != 'max':
            memory.append(int(path.read_text()))
    effective_cores = min([len(cores)] + quotas)
    required_cores = workers if workers is not None else c['cpu_allocation_per_training_job']
    if effective_cores < required_cores:
        raise RuntimeError('Requested {} CPU cores, available {}'.format(required_cores, effective_cores))
    memory_limit = min(memory) if memory else None
    if device == 'cuda' and memory_limit is not None and memory_limit < c['ram_gib_per_training_job'] * 1024**3:
        raise RuntimeError('Container RAM below configured training allocation')
    return dict(stack=info, host=socket.gethostname(), cores=cores,
                effective_cpu_count=effective_cores, memory_limit_bytes=memory_limit,
                device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)
