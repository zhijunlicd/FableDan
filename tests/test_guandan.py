"""Local regressions: python -m unittest discover -s tests -p test_guandan.py."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fabledan.cards import order_of
from fabledan.combos import gen_moves, Move, SINGLE, PAIR, SFLUSH, claim_ids, classify_claim
from fabledan.encode import hand_action_features, tokenize, SCHEMA_VERSION, FEAT_DIM
from fabledan.engine import default_return_card
from fabledan.model_torch import FableDanNet, ModelConfig, export_npz, save_ckpt, load_ckpt
from fabledan.model_np import NumpyModel


def obs(hand, events=None):
    return dict(player=0, level=1, hand=hand, left=[len(hand),27,27,27], done=[False]*4,
                lead=None, events=events or [])

class ForkTests(unittest.TestCase):
    def test_suit_sensitive_actions(self):
        hand = [10,9,14,18,22,26]
        legal = gen_moves(hand,1)
        self.assertTrue(any(m.type == SINGLE and m.cards == [9] for m in legal))
        self.assertTrue(any(m.type == SINGLE and m.cards == [10] for m in legal))
        self.assertTrue(any(m.type == SFLUSH for m in legal))
        a,b = (Move(SINGLE,order_of(2,1),[c],[2]) for c in [9,10])
        self.assertFalse(np.array_equal(hand_action_features(obs(hand),a),hand_action_features(obs(hand),b)))
        self.assertNotEqual(tokenize([('play',1,a)],0,1),tokenize([('play',1,b)],0,1))

    def test_identical_deck_copies_canonicalized(self):
        singles = [m for m in gen_moves([9,63],1) if m.type == SINGLE]
        self.assertEqual(len(singles),1)
        self.assertTrue(any(m.type == PAIR and len(m.cards) == 2 for m in gen_moves([9,63],1)))

    def test_wild_level_single_remains_available(self):
        singles = [m.cards for m in gen_moves([4,5],1) if m.type == SINGLE]
        self.assertIn([4],singles)
        self.assertIn([5],singles)

    def test_optional_wildcard_consumption(self):
        pairs = {tuple(sorted(m.cards)) for m in gen_moves([4,9,10],1) if m.type == PAIR}
        self.assertEqual(pairs,{(4,9),(4,10),(9,10)})

    def test_joker_pair_cannot_attach_to_full_house(self):
        self.assertFalse(any(m.size == 5 for m in gen_moves([9,10,11,52,106],1)))

    def test_two_wilds_are_level_pair(self):
        pairs = [m for m in gen_moves([4,58,9],1) if sorted(m.cards)==[4,58]]
        self.assertEqual([(m.type,m.key) for m in pairs],[(PAIR,order_of(1,1))])

    def test_pairs_preserve_suit_choices(self):
        pairs = [m for m in gen_moves([9,10,11],1) if m.type == PAIR]
        self.assertEqual({tuple(sorted(m.cards)) for m in pairs},{(9,10),(9,11),(10,11)})

    def test_history_summary_survives_truncation(self):
        a,b = (Move(SINGLE,0,[c],[2]) for c in [9,10])
        tail = [('pass',1)]*300
        self.assertEqual(tokenize([('play',1,a)]+tail,0,1),tokenize([('play',1,b)]+tail,0,1))
        self.assertFalse(np.array_equal(hand_action_features(obs([20],[('play',1,a)]+tail),a),
                                       hand_action_features(obs([20],[('play',1,b)]+tail),a)))

    def test_return_excludes_ace_and_level_when_low_available(self):
        self.assertEqual(default_return_card([0,5,9],1),9)
        self.assertEqual(default_return_card([0,5],1),0)

    def test_physical_choices_still_classify(self):
        rng = random.Random(19)
        for _ in range(40):
            hand = rng.sample(range(108),27)
            level = rng.randrange(13)
            for m in gen_moves(hand,level):
                self.assertEqual(len(m.cards),len(set(m.cards)))
                self.assertTrue(set(m.cards) <= set(hand))
                parsed = classify_claim(m.cards,claim_ids(m),level)
                self.assertEqual((parsed.type,parsed.key),(m.type,m.key))

    def test_export_checkpoint_and_jsonl_transport(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = FableDanNet(ModelConfig(d_model=16,n_blocks=1,n_heads=2,qk_dim=8,v_dim=8,
            ffn_hidden=32,hand_hidden=32,n_hand_layers=1,q_hidden=32,n_q_layers=1)).eval()
        features = np.random.default_rng(4).normal(size=(8,FEAT_DIM)).astype(np.float32)
        tokens = [1,3,15,20,34]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            export_npz(model,path)
            numpy_model = NumpyModel(path)
            with torch.no_grad():
                expected,_ = model(torch.tensor([tokens]),torch.tensor([len(tokens)]),torch.from_numpy(features[None]))
            np.testing.assert_allclose(numpy_model.q_values(tokens,features),expected[0].numpy(),rtol=2e-4,atol=2e-4)
            save_ckpt(model,None,{},Path(tmp)/'model.pt')
            restored,_ = load_ckpt(Path(tmp)/'model.pt')
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
            weights = dict(np.load(path,allow_pickle=False))
            weights['__config__'] = np.array(['feat_dim=80'])
            with self.assertRaisesRegex(ValueError,'schema'):
                NumpyModel(weights)
            request = dict(id=1,schema=SCHEMA_VERSION,tokens=tokens,features=features.tolist())
            lines = '\n'.join([json.dumps(dict(request,schema='old')),json.dumps(request)])+'\n'
            proc = subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'serve.py'),'--model',str(path)],
                input=lines,text=True,capture_output=True,timeout=30,check=True)
            ready,bad,good = map(json.loads,proc.stdout.splitlines())
            self.assertTrue(ready['ready'])
            self.assertIn('error',bad)
            self.assertEqual(good['index'],int(expected[0].argmax()))

if __name__ == '__main__':
    unittest.main()
