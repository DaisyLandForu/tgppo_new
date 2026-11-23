"""External data and local runtime paths."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def local_config():
    path = Path(os.getenv('TGPPO_PATHS', PROJECT_ROOT / 'configs/paths.local.json')).resolve()
    values = json.loads(path.read_text()) if path.exists() else {}
    return values, path.parent


def data_paths():
    values, base = local_config()
    defaults = {
        'train_instances': 'data/train_instances',
        'test_instances': 'data/test_instances',
        'expert_dataset': 'data/expert.npz',
        'scip_prefix': '',
    }
    paths = {}
    for key, default in defaults.items():
        value = os.getenv('TGPPO_' + key.upper(), values.get(key))
        if key == 'scip_prefix':
            value = value or os.getenv('SCIPOPTDIR')
            if not value:
                continue
        path = Path(value).expanduser() if value else PROJECT_ROOT / default
        paths[key] = (base / path).resolve()
    return paths


def expert_dataset_path(config):
    return Path(config.get('expert_dataset_path', data_paths()['expert_dataset']))


def output_root():
    return Path(os.getenv('TGPPO_OUTPUT_DIR', PROJECT_ROOT / 'runs')).expanduser().resolve()
