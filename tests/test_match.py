import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fabledan.match import advance_match, GuandanMatch
from fabledan.agents import RandomAgent

class MatchTests(unittest.TestCase):
    def test_advancement_and_ace_against_app(self):
        for case in json.loads((Path(__file__).resolve().parents[1]/'fixtures/advancement.json').read_text())['cases']:
            self.assertEqual(advance_match(case['levels'],case['ranking']),
                dict(levels=case['newLevels'],winner=case['winner'],game_over=case['gameOver'],score=case['score']))

    def test_multiple_rounds_preserve_state(self):
        game=GuandanMatch(seed=789)
        agents=[RandomAgent(i) for i in range(4)]
        prior=None
        for round_number in range(1,6):
            levels=list(game.levels)
            declaring=game.declaring_team
            rewards,ranking,round_game=game.play_round(agents)
            self.assertEqual(round_game.lv,levels[declaring])
            self.assertEqual(game.round_number,round_number)
            self.assertEqual(game.levels,advance_match(levels,ranking)['levels'])
            self.assertEqual(sum(rewards),0)
            if prior is not None:
                self.assertEqual(round_game.tribute_mode[-2:],[prior[-1],prior[0]] if isinstance(round_game.tribute_mode,list) else (prior[-1],prior[0]))
            prior=ranking
            if game.game_over: break

    def test_finished_match_rejects_new_round(self):
        game=GuandanMatch();game.game_over=True
        with self.assertRaisesRegex(ValueError,'finished'):game.play_round([])

if __name__=='__main__':unittest.main()
