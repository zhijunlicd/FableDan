"""Bounded CPU DMC smoke run. Outputs are NOT playing-strength evidence."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from fabledan.agents import TorchAgent
from fabledan.encode import encode_decision, SCHEMA_VERSION
from fabledan.engine import GuandanRound
from fabledan.model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt
from fabledan.model_np import NumpyModel


def run(out, rounds=2, seed=7):
    if not 1 <= rounds <= 20:
        raise ValueError('smoke rounds must be between 1 and 20')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = FableDanNet(ModelConfig(d_model=16, n_blocks=1, n_heads=2,
        qk_dim=8, v_dim=8, ffn_hidden=32, hand_hidden=32,
        n_hand_layers=1, q_hidden=32, n_q_layers=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    losses, decisions = [], 0
    for _ in range(rounds):
        samples = []
        agent = TorchAgent(model, eps=0.2, top_k=0, rng=rng)
        def collect(player, obs, legal, idx):
            toks, feats = encode_decision(obs)
            samples.append((player, toks, feats[idx:idx+1]))
        game = GuandanRound(rng.randrange(13), rng=rng)
        rewards, ranking = game.play([agent] * 4, sample_cb=collect)
        # Monte Carlo team return; no rank-rule imitation or hidden hand inputs.
        model.train()
        for player, toks, feats in samples:
            q, _ = model(torch.tensor([toks]), torch.tensor([len(toks)]),
                         torch.from_numpy(feats[None]))
            loss = (q[0, 0] - rewards[player] / 3.0).square()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        decisions += len(samples)
    assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
    model.eval()
    export_npz(model, out / 'smoke.npz')
    save_ckpt(model, optimizer, {'seed': seed, 'rounds': rounds, 'smoke_only': True}, out / 'smoke.pt')
    numpy_model = NumpyModel(out / 'smoke.npz')
    with torch.no_grad():
        expected, _ = model(torch.tensor([toks]), torch.tensor([len(toks)]), torch.from_numpy(feats[None]))
    actual = numpy_model.q_values(toks, feats)
    np.testing.assert_allclose(actual, expected[0].numpy(), rtol=2e-4, atol=2e-4)
    report = {'schema': SCHEMA_VERSION, 'seed': seed, 'rounds': rounds,
              'decisions': decisions, 'mean_loss': float(np.mean(losses)),
              'numpy_max_error': float(np.max(np.abs(actual - expected[0].numpy()))),
              'smoke_only': True, 'strength_evaluated': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='training/runs/smoke')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    run(args.out, args.rounds, args.seed)
