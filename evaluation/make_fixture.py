"""Deterministic synthetic inference weights for CI; never a trained bot."""
import argparse
from pathlib import Path
import numpy as np
from fabledan.encode import SCHEMA_VERSION, FEAT_DIM, VOCAB


def make(path):
    rng = np.random.default_rng(101)
    cfg = dict(schema_version=SCHEMA_VERSION, feat_dim=FEAT_DIM, vocab=VOCAB,
               max_seq=512, d_model=16, n_blocks=1, n_heads=2, qk_dim=8,
               v_dim=8, ffn_hidden=32, hand_hidden=32, n_hand_layers=1,
               q_hidden=32, n_q_layers=1)
    w = {}

    def weight(name, *shape):
        w[name] = rng.normal(0, 0.08, shape).astype(np.float32)

    def norm(name, dim):
        w[name] = np.ones(dim, dtype=np.float32)

    half = cfg['qk_dim'] // 2
    frequency = 1 / (10000 ** (np.arange(half, dtype=np.float32) / half))
    angle = np.arange(512, dtype=np.float32)[:, None] * frequency[None]
    w['rope_cos'], w['rope_sin'] = np.cos(angle), np.sin(angle)
    weight('token_emb.weight', VOCAB, 16)
    norm('blocks.0.attn_norm.weight', 16)
    norm('blocks.0.ffn_norm.weight', 16)
    for name in ('q', 'k', 'v', 'out'):
        weight('blocks.0.attn.' + name + '_proj.weight', 16, 16)
    for name in ('q', 'k'):
        norm('blocks.0.attn.' + name + '_norm.weight', 8)
    for name in ('gate', 'up'):
        weight('blocks.0.ffn.' + name + '_proj.weight', 32, 16)
    weight('blocks.0.ffn.down_proj.weight', 16, 32)
    norm('final_norm.weight', 16)
    for name, input_dim, hidden_dim, output_dim in (
            ('hand_mlp.', FEAT_DIM, 32, 16), ('q_head.', 32, 32, 1)):
        for idx, dims in ((0, (hidden_dim, input_dim)), (2, (output_dim, hidden_dim))):
            weight(name + str(idx) + '.weight', *dims)
            weight(name + str(idx) + '.bias', dims[0])
    w['__config__'] = np.array([f'{k}={v}' for k, v in cfg.items()], dtype=np.str_)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **w)
    print('Synthetic, untrained CI fixture:', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    make(parser.parse_args().out)
