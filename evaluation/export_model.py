"""Convert a FableDan NPZ export to a dependency-free Node inference artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fabledan.model_np import NumpyModel

def export(source, output):
    model = NumpyModel(source)  # Reject incompatible schemas before export.
    arrays = {}
    for name, value in model.w.items():
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError('Non-finite model weight: ' + name)
        arrays[name] = {'shape': list(value.shape), 'data': value.reshape(-1).tolist()}
    result = {'format': 'fabledan-node-v1', 'config': model.cfg,
              'sourceNpzSha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'weights': arrays}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'path': str(target), 'bytes': target.stat().st_size,
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    export(a.source, a.output)
