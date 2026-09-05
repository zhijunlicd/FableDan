# -*- coding: utf-8 -*-
# Modified by the Guandan_opencode project: suit-aware experimental fork.
# See ../README.md for changes and ../LICENSE for upstream terms.
"""GuanDan single-round engine aligned with Guandan_opencode rules.

Players 0..3, teams (0,2) and (1,3).
Events recorded for tokenization:
  ('tribute', player, card, receiver)   ('return', player, card, receiver)
  ('play', player, Move)      ('pass', player)
Rewards: winning team gets +3 (双下) / +2 (1st+3rd) / +1 (1st+4th), losers
the negative. Per-player reward = team reward.
"""

import random

from .cards import BJ, NUM_CARDS, is_wildcard, order_of, rank_of
from .combos import PASS_MOVE, Move, beats, gen_moves


def partner(p):
    return (p + 2) % 4


def forced_tribute_card(cards, lv):
    """Largest card excluding the wildcard (heart level)."""
    cand = [c for c in cards if not is_wildcard(c, lv)]
    if not cand:
        cand = list(cards)
    return max(cand, key=lambda c: order_of(rank_of(c), lv))


def default_return_card(cards, lv):
    """Heuristic return (还贡): smallest-order card with face value <= 10."""
    cand = [c for c in cards
            if 1 <= rank_of(c) <= 9 and rank_of(c) != lv]  # literal 2..10, excluding level
    if not cand:
        cand = list(cards)
    return min(cand, key=lambda c: order_of(rank_of(c), lv))


class GuandanRound:
    """One round. `agents` is a list of 4 objects with .act(obs) -> move index.

    tribute_mode: None | ('single', last, first) | ('double', last, first)
    """

    def __init__(self, level, rng=None, tribute_mode=None, deal=None, first_player=0):
        self.lv = level
        self.rng = rng or random.Random()
        if deal is None:
            deck = list(range(NUM_CARDS))
            self.rng.shuffle(deck)
            self.hands = [deck[i * 27:(i + 1) * 27] for i in range(4)]
        else:
            self.hands = [list(h) for h in deal]
        self.events = []
        self.tribute_mode = tribute_mode
        self.done_order = []          # players in finish order
        self.lead_player = first_player
        self.first_player = first_player
        self.resist = False

    # ------------------------------------------------------------------
    def _do_tribute(self):
        mode = self.tribute_mode
        if not mode:
            self.lead_player = self.first_player
            return
        kind, last, first = mode
        lv = self.lv
        if kind == 'single':
            payers = [last]
        else:
            payers = [last, partner(last)]
        n_bj = sum(1 for p in payers for c in self.hands[p]
                   if rank_of(c) == BJ)
        if n_bj >= 2:
            self.resist = True
            self.lead_player = first
            return
        if kind == 'single':
            c = forced_tribute_card(self.hands[last], lv)
            self.hands[last].remove(c)
            self.hands[first].append(c)
            self.events.append(('tribute', last, c, first))
            r = default_return_card(self.hands[first], lv)
            self.hands[first].remove(r)
            self.hands[last].append(r)
            self.events.append(('return', first, r, last))
            self.lead_player = last
        else:
            receivers = [first, partner(first)]
            t0 = forced_tribute_card(self.hands[payers[0]], lv)
            t1 = forced_tribute_card(self.hands[payers[1]], lv)
            o0 = order_of(rank_of(t0), lv)
            o1 = order_of(rank_of(t1), lv)
            # Ties go to the giver next in playing order after first.
            tie_top = next((first + d) % 4 for d in range(1,5) if (first+d)%4 in payers)
            if o1 > o0 or (o1 == o0 and tie_top == payers[1]):
                pay_pairs = [(payers[1], t1, first), (payers[0], t0, partner(first))]
                self.lead_player = payers[1]
            else:
                pay_pairs = [(payers[0], t0, first), (payers[1], t1, partner(first))]
                self.lead_player = payers[0]
            for payer, card, recv in pay_pairs:
                self.hands[payer].remove(card)
                self.hands[recv].append(card)
                self.events.append(('tribute', payer, card, recv))
            for payer, card, recv in pay_pairs:
                r = default_return_card(self.hands[recv], lv)
                self.hands[recv].remove(r)
                self.hands[payer].append(r)
                self.events.append(('return', recv, r, payer))

    # ------------------------------------------------------------------
    def play(self, agents, sample_cb=None):
        """Run the round with callback-style agents. Returns (rewards, ranking).

        sample_cb(player, obs, legal, chosen_idx) is called at each decision
        point with >=2 legal moves (for training data collection).
        """
        gen = self.play_steps()
        try:
            obs = next(gen)
            while True:
                idx = agents[obs["player"]].act(obs)
                if sample_cb is not None:
                    sample_cb(obs["player"], obs, obs["legal"], idx)
                obs = gen.send(idx)
        except StopIteration as e:
            return e.value

    # ------------------------------------------------------------------
    def play_steps(self, observe_forced=False):
        """Yield decisions; observe_forced also exposes one-action turns for replay."""
        self._do_tribute()
        cur, lead_move, lead_owner = self.lead_player, None, None
        done, passes = [False] * 4, 0

        def next_active(p):
            for offset in range(1,5):
                candidate = (p + offset) % 4
                if not done[candidate]:
                    return candidate
            return p

        while True:
            legal = gen_moves(self.hands[cur], self.lv, lead_move)
            if len(legal) == 1 and not observe_forced:
                idx = 0
            else:
                idx = yield self._make_obs(cur, legal, lead_move, lead_owner, done)
            if not isinstance(idx, int) or not 0 <= idx < len(legal):
                raise ValueError('Invalid action index')
            move = legal[idx]
            if move.type == PASS_MOVE.type:
                passes += 1
                self.events.append(('pass', cur))
            else:
                for c in move.cards:
                    self.hands[cur].remove(c)
                self.events.append(('play', cur, move))
                lead_move, lead_owner, passes = move, cur, 0
                if not self.hands[cur]:
                    done[cur] = True
                    self.done_order.append(cur)
            if self.done_order and partner(self.done_order[0]) in self.done_order:
                break
            active_count = done.count(False)
            passes_needed = active_count - (0 if done[lead_owner] else 1)
            if passes >= passes_needed:
                if active_count <= 1:
                    self.done_order.extend(p for p in (0,3,2,1) if not done[p])
                    break
                cur = lead_owner
                if done[cur]:
                    cur = partner(cur) if not done[partner(cur)] else next_active(cur)
                lead_move, lead_owner, passes = None, None, 0
            else:
                cur = next_active(cur)
        # Match the app's stable PLAYERS order for unfinished losers.
        ranking = self.done_order + [p for p in (0,3,2,1) if p not in self.done_order]
        return self._rewards(ranking), ranking

    def _rewards(self, ranking):
        first = ranking[0]
        winners = (first, partner(first))
        pos_partner = ranking.index(partner(first))
        score = {1: 3, 2: 2, 3: 1}[pos_partner]
        return [score if p in winners else -score for p in range(4)]

    # ------------------------------------------------------------------
    def _make_obs(self, p, legal, lead_move, lead_owner, done):
        return {
            "player": p,
            "level": self.lv,
            "hand": list(self.hands[p]),
            "legal": legal,
            "lead": lead_move,
            "lead_owner": lead_owner,
            "events": self.events,
            "done": list(done),
            "left": [len(self.hands[i]) for i in range(4)],
        }


def random_tribute_mode(rng):
    """Random previous-round outcome for training diversity."""
    x = rng.random()
    if x < 1 / 3:
        return None
    first = rng.randrange(4)
    last = rng.choice([p for p in range(4) if p != first and p != partner(first)])
    if x < 2 / 3:
        return ('single', last, first)
    return ('double', last, first)


def play_round(agents, rng=None, level=None, tribute_mode='random',
               sample_cb=None):
    rng = rng or random.Random()
    if level is None:
        level = rng.randrange(13)
    if tribute_mode == 'random':
        tribute_mode = random_tribute_mode(rng)
    rnd = GuandanRound(level, rng, tribute_mode)
    rewards, ranking = rnd.play(agents, sample_cb)
    return rewards, ranking, rnd
