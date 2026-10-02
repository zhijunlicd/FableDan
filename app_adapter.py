"""Translate authoritative TypeScript patterns into fork observations.

Unsupported legal declarations fail explicitly; never silently drop candidates.
"""
from fabledan.cards import order_of
from fabledan.combos import gen_moves, PASS_MOVE, TYPE_NAMES
from fabledan.encode import encode_decision, SCHEMA_VERSION

TYPES = {'single':1,'pair':2,'triple':3,'full_house':4,'straight':5,
         'tube':6,'plate':7,'bomb':8,'straight_flush':9,'joker_bomb':10}


def move_from_pattern(pattern, level):
    if pattern is None:
        return PASS_MOVE
    kind = TYPES[pattern['type']]
    primary = pattern['primaryRank']
    if kind in (5,9):
        key = primary - 4
    elif kind == 6:
        key = primary - 2
    elif kind == 7:
        key = primary - 1
    elif kind == 10:
        key = 0
    else:
        rank = level if primary == 15 else (13 if primary == 16 else (14 if primary == 17 else (0 if primary == 14 else primary - 1)))
        key = order_of(rank, level)
    cards = pattern['cards']
    for move in gen_moves(cards, level):
        if move.type == kind and move.key == key and sorted(move.cards) == sorted(cards):
            return move
    raise ValueError(f'unsupported authoritative pattern: {pattern["type"]}/{primary}')


def encode_app(request):
    data = request['observation']
    level = data['level']
    events = []
    for event in data['events']:
        if event[0] == 'play':
            events.append(('play',event[1],move_from_pattern(event[2],level)))
        else:
            events.append(tuple(event))
    obs = dict(data, events=events, lead=move_from_pattern(data['lead'],level) if data['lead'] else None,
               legal=[move_from_pattern(p,level) for p in data['legal']])
    tokens, features = encode_decision(obs)
    return dict(schema=SCHEMA_VERSION,tokens=tokens,features=features)
