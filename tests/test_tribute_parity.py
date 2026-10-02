import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fabledan.engine import GuandanRound

class TributeParityTests(unittest.TestCase):
    def test_tribute_replays(self):
        cases=json.loads((Path(__file__).resolve().parents[1]/'fixtures/tribute-traces.json').read_text())['cases']
        for number,case in enumerate(cases):
            game=GuandanRound(case['level'],deal=case['initial'],tribute_mode=case['mode'])
            game._do_tribute()
            label=f'case {number}: {case["mode"]}'
            self.assertEqual(game.resist,case['resisted'],label)
            self.assertEqual(game.lead_player,case['leader'],label)
            canonical=lambda hs:[sorted(c%54 for c in h) for h in hs]
            self.assertEqual(canonical(game.hands),canonical(case['final']),label)
            events=lambda es:[(e[0],e[1],e[2]%54,e[3]) for e in es]
            self.assertEqual(events(game.events),events(case['events']),label)

if __name__=='__main__': unittest.main()
