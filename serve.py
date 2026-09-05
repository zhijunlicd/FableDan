"""Experimental JSONL inference transport; receives encoded PUBLIC observations.

The game adapter must supply validated legal actions and public-only features.
This transport deliberately does not claim to implement the TypeScript adapter.
"""
import argparse
import json
import sys
import numpy as np
from fabledan.encode import FEAT_DIM, MAX_SEQ, SCHEMA_VERSION, VOCAB
from fabledan.model_np import NumpyModel


def decide(model, request):
    if request.get('schema') != SCHEMA_VERSION:
        raise ValueError('schema mismatch')
    if 'observation' in request:
        from app_adapter import encode_app
        request = encode_app(request)
    tokens = np.asarray(request['tokens'])
    features = np.asarray(request['features'], dtype=np.float32)
    if tokens.ndim != 1 or not 1 <= len(tokens) <= MAX_SEQ or tokens.dtype.kind not in 'iu' or np.any(tokens < 0) or np.any(tokens >= VOCAB):
        raise ValueError('invalid tokens')
    if features.ndim != 2 or features.shape[1] != FEAT_DIM or not 1 <= features.shape[0] <= 50000 or not np.isfinite(features).all():
        raise ValueError('invalid action features')
    scores = model.q_values(tokens.tolist(), features)
    if not np.isfinite(scores).all():
        raise ValueError('non-finite model output')
    return {'index': int(np.argmax(scores)), 'candidates': len(scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = NumpyModel(args.model)
    print(json.dumps({'ready': True, 'schema': SCHEMA_VERSION}), flush=True)
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('request must be an object')
            response = {'id': request.get('id'), **decide(model, request)}
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            response = {'id': request.get('id') if isinstance(request, dict) else None, 'error': str(exc)}
        print(json.dumps(response), flush=True)

if __name__ == '__main__':
    main()
