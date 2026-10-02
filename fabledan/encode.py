# -*- coding: utf-8 -*-
# Modified by the Guandan_opencode project: suit-aware experimental fork.
# See ../README.md for changes and ../LICENSE for upstream terms.
"""State/action encoding shared by training (torch) and inference (numpy).

Fork schema guandan-suits-v1: original rank/type tokens plus physical card
identities (48..101) and a physical-card marker (102). Tribute events include
actual cards and receiver seats. Feature vectors preserve played-card suit
counts even when old sequence tokens are truncated.
"""

import numpy as np

from .cards import NUM_RANKS, is_wildcard, order_of, rank_of
from .combos import PASS, TYPE_NAMES

SCHEMA_VERSION = "guandan-suits-v1"
CARD_BASE = 48
PHYSICAL_TOK = 102
VOCAB = 103
PAD_TOK, BOS_TOK = 0, 1
LEVEL_BASE = 2
PLAYER_BASE = 15
TYPE_BASE = 19          # + move type (0..10)
TRIBUTE_TOK, RETURN_TOK = 30, 31
RANK_BASE = 32
MAX_SEQ = 512

N_TYPES = 11
# hand/action feature layout
FEAT_DIM = (
    15      # hand rank counts /4
    + 1     # wildcards in hand /2
    + 1     # hand size /27
    + 4     # cards left per relative player /27
    + 4     # done flags per relative player
    + 13    # level one-hot
    + N_TYPES  # action type one-hot
    + 15    # action claim rank counts /4
    + 1     # action size /27
    + 1     # action wildcards used /2
    + 1     # action comparison key /15
    + N_TYPES  # current lead type one-hot (all 0 if leading)
    + 1     # lead key /15
    + 1     # leading flag
    + 54 * 6  # physical hand, action and played cards by relative seat
)


def tokenize(events, viewer, level):
    """events: engine event list; viewer: absolute player id. -> list[int]"""
    toks = [BOS_TOK, LEVEL_BASE + level]
    for ev in events:
        kind = ev[0]
        p = (ev[1] - viewer) % 4
        if kind == 'pass':
            toks.append(PLAYER_BASE + p)
            toks.append(TYPE_BASE + PASS)
        elif kind == 'play':
            mv = ev[2]
            toks.append(PLAYER_BASE + p)
            toks.append(TYPE_BASE + mv.type)
            for r in sorted(mv.claim_ranks):
                toks.append(RANK_BASE + r)
            toks += [PHYSICAL_TOK] + [CARD_BASE + c % 54 for c in sorted(mv.cards, key=lambda c: c % 54)]
        elif kind == 'tribute':
            toks += [PLAYER_BASE + p, TRIBUTE_TOK, CARD_BASE + ev[2] % 54, PLAYER_BASE + (ev[3] - viewer) % 4]
        elif kind == 'return':
            toks += [PLAYER_BASE + p, RETURN_TOK, CARD_BASE + ev[2] % 54, PLAYER_BASE + (ev[3] - viewer) % 4]
    if len(toks) > MAX_SEQ:
        toks = toks[:2] + toks[-(MAX_SEQ - 2):]
    return toks


def hand_action_features(obs, move):
    """Suit-aware float32 features for (state-side hand info, candidate move)."""
    lv = obs["level"]
    me = obs["player"]
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    i = 0
    for c in obs["hand"]:
        f[rank_of(c)] += 0.25
    i += 15
    f[i] = sum(1 for c in obs["hand"] if is_wildcard(c, lv)) / 2.0; i += 1
    f[i] = len(obs["hand"]) / 27.0; i += 1
    for rel in range(4):
        f[i + rel] = obs["left"][(me + rel) % 4] / 27.0
    i += 4
    for rel in range(4):
        f[i + rel] = 1.0 if obs["done"][(me + rel) % 4] else 0.0
    i += 4
    f[i + lv] = 1.0; i += 13
    f[i + move.type] = 1.0; i += N_TYPES
    for r in move.claim_ranks:
        f[i + r] += 0.25
    i += 15
    f[i] = move.size / 27.0; i += 1
    f[i] = sum(1 for c in move.cards if is_wildcard(c, lv)) / 2.0; i += 1
    f[i] = (move.key / 15.0) if move.type != PASS else 0.0; i += 1
    lead = obs["lead"]
    if lead is not None and lead.type != PASS:
        f[i + lead.type] = 1.0
        f[i + N_TYPES] = lead.key / 15.0
        f[i + N_TYPES + 1] = 0.0
    else:
        f[i + N_TYPES + 1] = 1.0  # leading
    i += N_TYPES + 2
    for c in obs["hand"]:
        f[i + c % 54] += 0.5
    i += 54
    for c in move.cards:
        f[i + c % 54] += 0.5
    i += 54
    for ev in obs["events"]:
        if ev[0] == "play":
            rel = (ev[1] - me) % 4
            for c in ev[2].cards:
                f[i + rel * 54 + c % 54] += 0.5
    i += 216
    assert i == FEAT_DIM
    return f


def encode_decision(obs):
    """-> (tokens list[int], feats ndarray [n_legal, FEAT_DIM])"""
    toks = tokenize(obs["events"], obs["player"], obs["level"])
    feats = np.stack([hand_action_features(obs, m) for m in obs["legal"]])
    return toks, feats


def pad_tokens(toks, length=None):
    length = length or MAX_SEQ
    arr = np.zeros(length, dtype=np.int64)
    arr[:len(toks)] = toks[:length]
    return arr, min(len(toks), length)


# ---------------------------------------------------------------------------
# flat encoding (for the numpy MLP demo model / fallback)
# ---------------------------------------------------------------------------

FLAT_DIM = FEAT_DIM + 4 * 15 + N_TYPES + 15 + 1   # suit-aware base plus rank history


def encode_flat(obs, move):
    """Flat features = hand/action features + aggregated history."""
    from .combos import PASS as _PASS
    me = obs["player"]
    base = hand_action_features(obs, move)
    f = np.zeros(FLAT_DIM, dtype=np.float32)
    f[:FEAT_DIM] = base
    i = FEAT_DIM
    # cards played per relative player per rank
    for ev in obs["events"]:
        if ev[0] == 'play':
            rel = (ev[1] - me) % 4
            for r in ev[2].claim_ranks:
                f[i + rel * 15 + r] += 0.125
    i += 60
    # last non-pass move in history
    last = None
    for ev in reversed(obs["events"]):
        if ev[0] == 'play':
            last = ev[2]
            break
        if ev[0] in ('tribute', 'return'):
            break
    if last is not None:
        f[i + last.type] = 1.0
        f[i + N_TYPES + (last.claim_ranks[0] if last.claim_ranks else 0)] = 1.0
        f[i + N_TYPES + 15] = last.size / 27.0
    return f
