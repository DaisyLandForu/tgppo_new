"""One execution lock per condition; retain progress on handled job termination."""
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import time

from tgppo.solver.protocol import write_json


class Stopped(RuntimeError):
    pass


@contextlib.contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


@contextlib.contextmanager
def budget(folder, seconds, resume):
    with lock(folder / 'execution.lock', blocking=False):
        path = folder / 'budget.json'
        record = json.loads(path.read_text()) if path.exists() else {'attempts': []}
        if (record['attempts'] or (folder / 'latest.pt').exists()) and not resume:
            raise ValueError('Existing run requires --resume')
        start = time.time()
        spent = sum(a.get('elapsed_seconds', a.get('allocated_seconds') or 0)
                    for a in record['attempts'])
        remaining = math.inf if seconds is None else seconds - spent
        if remaining <= 60:
            raise Stopped('Cumulative execution budget exhausted')
        attempt = dict(started_epoch=start, host=socket.gethostname(), pid=os.getpid(),
                       allocated_seconds=remaining if seconds is not None else None)
        record['attempts'].append(attempt)
        write_json(path, record)

        def stop(signum, frame):
            raise Stopped('Signal {}; last committed checkpoint retained'.format(signum))

        handlers = {s: signal.signal(s, stop) for s in
                    (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        if seconds is not None:
            signal.setitimer(signal.ITIMER_REAL, remaining - 30)
        try:
            yield start + remaining
        except BaseException as exc:
            state_path = folder / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            write_json(state_path, dict(state, status='stopped' if isinstance(exc, Stopped)
                       else 'failed', error=repr(exc)))
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
            attempt.update(elapsed_seconds=time.time() - start, finished_epoch=time.time())
            write_json(path, record)
