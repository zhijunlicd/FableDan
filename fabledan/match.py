"""Multi-round state for the app's level advancement and passing A rules."""
import random
from .engine import GuandanRound

RANK_ORDER = tuple(range(1,13)) + (0,)


def advance_match(levels, ranking):
    if sorted(ranking) != [0,1,2,3] or len(levels) != 2 or any(l not in RANK_ORDER for l in levels):
        raise ValueError('Invalid match state')
    winner = ranking[0] % 2
    partner_position = ranking.index((ranking[0]+2)%4) + 1
    score = 5 - partner_position
    new_levels = list(levels)
    old = levels[winner]
    new_levels[winner] = RANK_ORDER[min(12,RANK_ORDER.index(old)+score)]
    return dict(levels=new_levels,winner=winner,game_over=old == 0 and partner_position in (2,3),score=score)


class GuandanMatch:
    """Explicit round-boundary state; callers bound the number of rounds.

    Tribute decisions still use the engine's deterministic baseline. This is
    an environment building block, not the final multi-round training objective.
    """
    def __init__(self, level=1, seed=0):
        if level not in RANK_ORDER:
            raise ValueError('Invalid level')
        self.levels = [level,level]
        self.declaring_team = 0
        self.ranking = None
        self.round_number = 0
        self.game_over = False
        self.rng = random.Random(seed)

    def play_round(self, agents, deal=None, sample_cb=None):
        if self.game_over:
            raise ValueError('Match already finished')
        tribute = None
        first = 0
        if self.ranking is not None:
            first = self.ranking[0]
            kind = 'double' if self.ranking[1] == (first+2)%4 else 'single'
            tribute = (kind,self.ranking[-1],first)
        game = GuandanRound(self.levels[self.declaring_team],rng=self.rng,
                            tribute_mode=tribute,deal=deal,first_player=first)
        rewards, ranking = game.play(agents,sample_cb)
        result = advance_match(self.levels,ranking)
        self.levels = result['levels']
        self.declaring_team = result['winner']
        self.game_over = result['game_over']
        self.ranking = ranking
        self.round_number += 1
        return rewards, ranking, game
