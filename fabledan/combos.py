# -*- coding: utf-8 -*-
# Modified by the Guandan_opencode project: suit-aware experimental fork.
# See ../README.md for changes and ../LICENSE for upstream terms.
"""Move (一手牌) representation, enumeration with wildcards (配子), comparison.

Move types and fixed sizes:
  PASS      -
  SINGLE    1
  PAIR      2
  TRIPLE    3
  FULL      5   (三带二)
  STRAIGHT  5   (顺子, A=1 or 14)
  PLATE     6   (三连对/木板)
  TUBE      6   (钢板, two consecutive triples)
  BOMB      4..10
  SFLUSH    5   (同花顺)
  ROCKET    4   (4 jokers)
"""

from .cards import (BJ, SJ, NUM_RANKS, SEQV_TO_RANK, is_wildcard, order_of,
                    rank_of, seq_values, suit_of)

PASS, SINGLE, PAIR, TRIPLE, FULL, STRAIGHT, PLATE, TUBE, BOMB, SFLUSH, ROCKET = range(11)
TYPE_NAMES = ["PASS", "SINGLE", "PAIR", "TRIPLE", "FULL", "STRAIGHT",
              "PLATE", "TUBE", "BOMB", "SFLUSH", "ROCKET"]

# bomb tiers: 4bomb < 5bomb < SF < 6bomb < 7 < 8 < 9 < 10 < rocket
_BOMB_TIER = {4: 0, 5: 1, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7}
_SF_TIER = 2
_ROCKET_TIER = 99


class Move:
    __slots__ = ("type", "key", "cards", "claim_ranks", "size")

    def __init__(self, mtype, key, cards, claim_ranks):
        self.type = mtype
        self.key = key                  # comparison key within same type/size
        self.cards = cards              # actual card ids played
        self.claim_ranks = claim_ranks  # claimed rank indices (len == len(cards))
        self.size = len(cards)

    def is_bombish(self):
        return self.type in (BOMB, SFLUSH, ROCKET)

    def bomb_tier(self):
        if self.type == ROCKET:
            return _ROCKET_TIER
        if self.type == SFLUSH:
            return _SF_TIER
        return _BOMB_TIER[self.size]

    def __repr__(self):
        from .cards import RANK_NAMES
        return "%s[%s]" % (TYPE_NAMES[self.type],
                           ",".join(RANK_NAMES[r] for r in self.claim_ranks))


PASS_MOVE = Move(PASS, 0, [], [])


def beats(move, lead, lv):
    """Does `move` beat `lead`?  lead=None means leading (any non-pass ok)."""
    if move.type == PASS:
        return False
    if lead is None or lead.type == PASS:
        return True
    if move.is_bombish():
        if not lead.is_bombish():
            return True
        if move.bomb_tier() != lead.bomb_tier():
            return move.bomb_tier() > lead.bomb_tier()
        return move.key > lead.key
    if lead.is_bombish():
        return False
    if move.type != lead.type or move.size != lead.size:
        return False
    return move.key > lead.key


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

class HandIndex:
    """Pre-indexed hand for fast enumeration."""

    def __init__(self, cards, lv):
        self.lv = lv
        self.wilds = [c for c in cards if is_wildcard(c, lv)]
        self.naturals = [c for c in cards if not is_wildcard(c, lv)]
        self.cnt = [0] * NUM_RANKS
        self.by_rank = [[] for _ in range(NUM_RANKS)]
        self.by_suit_rank = {}
        for c in self.naturals:
            r = rank_of(c)
            self.cnt[r] += 1
            self.by_rank[r].append(c)
            s = suit_of(c)
            if s >= 0:
                self.by_suit_rank.setdefault((s, r), []).append(c)
        self.w = len(self.wilds)

    def pick(self, r, k, wild_used):
        """Pick k cards of rank r, using wildcards for the shortfall."""
        have = self.by_rank[r]
        n_nat = min(len(have), k)
        need = k - n_nat
        if wild_used + need > self.w:
            return None
        cards = have[:n_nat] + self.wilds[wild_used:wild_used + need]
        return cards, [r] * k, wild_used + need


def _claim_card_for_rank(r, used_ids, force_suit=None, avoid_suit=None):
    """A representative card id for a claimed rank (for botzone claim arrays)."""
    for base in (0, 54):
        if r == SJ:
            cid = 52 + base
        elif r == BJ:
            cid = 53 + base
        else:
            cid = None
        if cid is not None:
            if cid not in used_ids:
                used_ids.add(cid)
                return cid
            continue
        if force_suit is not None:
            suits = [force_suit]
        elif avoid_suit is not None:
            suits = [x for x in range(4) if x != avoid_suit]
        else:
            suits = range(4)
        for s in suits:
            cid2 = r * 4 + s + base
            if cid2 not in used_ids:
                used_ids.add(cid2)
                return cid2
    return r * 4 + (force_suit or 0)  # give up on uniqueness


def claim_ids(move):
    """Build the botzone `claim` array (card ids) for a move."""
    used = set()
    out = []
    force_suit = None
    if move.type == SFLUSH:
        from collections import Counter
        suits = Counter(suit_of(c) for c in move.cards if suit_of(c) >= 0)
        force_suit = suits.most_common(1)[0][0] if suits else 0
    for c, r in zip(move.cards, move.claim_ranks):
        if rank_of(c) == r and (force_suit is None or suit_of(c) == force_suit):
            out.append(c)
            used.add(c)
        else:
            out.append(None)  # wildcard placeholder
    avoid_suit = None
    if move.type == STRAIGHT:
        nat_suits = set(suit_of(c) for c in out if c is not None)
        if len(nat_suits) == 1:
            avoid_suit = nat_suits.pop()
    for i, (c, r) in enumerate(zip(move.cards, move.claim_ranks)):
        if out[i] is None:
            out[i] = _claim_card_for_rank(r, used, force_suit, avoid_suit)
    return out


# ---------------------------------------------------------------------------
# classification of an observed (action, claim) pair  -> Move
# ---------------------------------------------------------------------------

def _find_seq_low(ranks, length, mult):
    """If `ranks` is exactly `length` consecutive rank groups with multiplicity
    `mult`, return lowest seq value; else None."""
    from collections import Counter
    cnt = Counter(ranks)
    if any(v != mult for v in cnt.values()) or len(cnt) != length:
        return None
    for low in range(1, 15 - length + 1):
        need = [SEQV_TO_RANK[v] for v in range(low, low + length)]
        if None in need:
            continue
        need_cnt = Counter()
        for r in need:
            need_cnt[r] += mult
        if need_cnt == cnt:
            return low
    return None


def classify_claim(action_cards, claim_cards, lv):
    """Reconstruct a Move from an observed botzone (action, claim) pair."""
    if not claim_cards:
        return PASS_MOVE
    ranks = sorted(rank_of(c) for c in claim_cards)
    n = len(ranks)
    from collections import Counter
    cnt = Counter(ranks)
    distinct = sorted(cnt)

    if n == 4 and cnt.get(SJ) == 2 and cnt.get(BJ) == 2:
        return Move(ROCKET, 0, list(action_cards), ranks)
    if len(distinct) == 1:
        r = distinct[0]
        if n == 1:
            return Move(SINGLE, order_of(r, lv), list(action_cards), ranks)
        if n == 2:
            return Move(PAIR, order_of(r, lv), list(action_cards), ranks)
        if n == 3:
            return Move(TRIPLE, order_of(r, lv), list(action_cards), ranks)
        if n >= 4:
            return Move(BOMB, order_of(r, lv), list(action_cards), ranks)
    if n == 5:
        if len(distinct) == 2 and sorted(cnt.values()) == [2, 3]:
            trip = [r for r in cnt if cnt[r] == 3][0]
            return Move(FULL, order_of(trip, lv), list(action_cards), ranks)
        low = _find_seq_low(ranks, 5, 1)
        if low is not None:
            suits = set(suit_of(c) for c in claim_cards)
            t = SFLUSH if len(suits) == 1 and -1 not in suits else STRAIGHT
            return Move(t, low, list(action_cards), ranks)
    if n == 6:
        low = _find_seq_low(ranks, 3, 2)
        if low is not None:
            return Move(PLATE, low, list(action_cards), ranks)
        low = _find_seq_low(ranks, 2, 3)
        if low is not None:
            return Move(TUBE, low, list(action_cards), ranks)
    raise ValueError("unclassifiable claim: %r" % (claim_cards,))


def gen_moves(cards, lv, lead=None):
    """Enumerate physical choices and resolve declarations as the app does.

    Natural/wildcard use is a choice, not a greedy minimum. Only identical
    double-deck copies are collapsed. Ambiguous physical plays use the app's
    classification precedence: bombs first, strongest full house, lowest run.
    """
    from itertools import combinations
    from functools import lru_cache
    h = HandIndex(sorted(cards), lv)
    result = {}
    priority = {SINGLE:0, PAIR:1, TRIPLE:2, ROCKET:3, BOMB:4,
                FULL:5, PLATE:6, TUBE:7, SFLUSH:8, STRAIGHT:9}

    @lru_cache(None)
    def natural_choices(rank, count, suit):
        pool = [c for c in h.by_rank[rank] if suit < 0 or suit_of(c) == suit]
        unique = {}
        for choice in combinations(pool, count):
            unique.setdefault(tuple(c % 54 for c in sorted(choice, key=lambda c:c % 54)), choice)
        return tuple(unique.values())

    def select(groups, suit=-1, i=0, used=0, picked=(), claims=()):
        if i == len(groups):
            yield list(picked), list(claims)
            return
        rank, count = groups[i]
        remaining = h.w - used if rank < 13 else 0
        for natural_count in range(max(0, count - remaining), min(count,h.cnt[rank])+1):
            wild_count = count - natural_count
            for natural in natural_choices(rank, natural_count, suit):
                chosen = natural + tuple(h.wilds[used:used+wild_count])
                yield from select(groups,suit,i+1,used+wild_count,picked+chosen,claims+(rank,)*count)

    def add(kind, key, groups, suit=-1):
        for picked, claims in select(groups,suit):
            if kind == PAIR and groups[0][0] != lv and all(is_wildcard(c,lv) for c in picked):
                continue
            move = Move(kind,key,picked,claims)
            signature = tuple(sorted(c % 54 for c in picked))
            # The app chooses a straight flush whenever naturals share a suit.
            if kind == STRAIGHT and len({suit_of(c) for c in picked if not is_wildcard(c,lv)}) == 1:
                continue
            old = result.get(signature)
            score = lambda m:(priority[m.type], -m.key if m.type == FULL else m.key)
            if old is None or score(move) < score(old):
                result[signature] = move

    # Singles cannot declare a wildcard as an arbitrary rank.
    for c in sorted(cards):
        signature = (c % 54,)
        result.setdefault(signature,Move(SINGLE,order_of(rank_of(c),lv),[c],[rank_of(c)]))
    for rank in range(15):
        if h.cnt[rank] or rank == lv:
            add(PAIR,order_of(rank,lv),[(rank,2)])
        if rank >= 13 or not h.cnt[rank]:
            continue
        add(TRIPLE,order_of(rank,lv),[(rank,3)])
        for count in range(4,min(10,h.cnt[rank]+h.w)+1):
            add(BOMB,order_of(rank,lv),[(rank,count)])
        for pair in range(13):
            if pair != rank and (h.cnt[pair] or pair == lv):
                add(FULL,order_of(rank,lv),[(rank,3),(pair,2)])
    for kind, length, count in [(STRAIGHT,5,1),(PLATE,3,2),(TUBE,2,3)]:
        for low in range(1,16-length):
            groups = [(SEQV_TO_RANK[v],count) for v in range(low,low+length)]
            if sum(max(0,count-h.cnt[r]) for r,_ in groups) > h.w:
                continue
            add(kind,low,groups)
            if kind == STRAIGHT:
                for suit in range(4):
                    add(SFLUSH,low,groups,suit)
    if h.cnt[SJ] >= 2 and h.cnt[BJ] >= 2:
        add(ROCKET,0,[(SJ,2),(BJ,2)])
    moves = [result[k] for k in sorted(result)]
    if lead is None or lead.type == PASS:
        return moves
    return [PASS_MOVE] + [m for m in moves if beats(m,lead,lv)]
