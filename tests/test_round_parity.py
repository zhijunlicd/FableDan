import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fabledan.engine import GuandanRound
from app_adapter import move_from_pattern


def physical(cards): return sorted(c%54 for c in cards)

class RoundParityTests(unittest.TestCase):
    def test_round_replays(self):
        cases=json.loads((Path(__file__).resolve().parents[1]/'fixtures/round-traces.json').read_text())['cases']
        for case in cases:
            game=GuandanRound(case['level'],deal=case['initial'],first_player=case['first'])
            stream=game.play_steps(observe_forced=True)
            obs=next(stream)
            result=None
            for number,step in enumerate(case['steps']):
                label=f"level={case['level']} first={case['first']} step={number}"
                self.assertIsNotNone(obs,label)
                self.assertEqual(obs['player'],step['player'],label)
                self.assertEqual(obs['done'],step['done'],label)
                self.assertEqual([physical(h) for h in game.hands],[physical(h) for h in step['hands']],label)
                expected=move_from_pattern(step['lead'],case['level']) if step['lead'] else None
                self.assertEqual((obs['lead'].type,obs['lead'].key) if obs['lead'] else None,
                                 (expected.type,expected.key) if expected else None,label)
                action=move_from_pattern(step['action'],case['level'])
                indices=[i for i,m in enumerate(obs['legal']) if (m.type,m.key,physical(m.cards))==(action.type,action.key,physical(action.cards))]
                self.assertTrue(indices,label)
                try: obs=stream.send(indices[0])
                except StopIteration as stopped:
                    result=stopped.value;obs=None
            self.assertIsNone(obs)
            self.assertEqual(result[1],case['ranking'])
            self.assertEqual([physical(h) for h in game.hands],[physical(h) for h in case['final']])

if __name__=='__main__': unittest.main()
