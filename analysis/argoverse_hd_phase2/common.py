import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(clean(value), f, indent=2, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def json_gzip(path, value):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as z:
            z.write(json.dumps(clean(value), separators=(',', ':'), allow_nan=False).encode())


def config_from_dict(data):
    from adaptive_pipeline.shared.config import PipelineConfig
    data = dict(data)
    data['class_map'] = {int(k): v for k, v in data['class_map'].items()}
    return PipelineConfig(**data)
