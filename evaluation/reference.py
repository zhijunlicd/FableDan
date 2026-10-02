"""Independent Python encoding and NumPy predictions for TS-generated public cases."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from app_adapter import move_from_pattern
from fabledan.encode import encode_decision
from fabledan.model_np import NumpyModel

RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
SUITS = ['H','D','S','C']
def card(c):
    return c['deck']*54+(52 if c.get('jokerType')=='BJ' else 53) if c['kind']=='joker' else c['deck']*54+4*RANKS.index(c['rank'])+SUITS.index(c['suit'])
def pattern(p):
    return None if p is None else {**p, 'cards': [card(c) for c in p['cards']]}
def observation(case):
    v=case['view'];lv=RANKS.index(v['levelRank']);me=int(v['viewFor'][1]);events=[]
    for e in case['events']:
        if e['type']=='cards_played':events.append(('play',int(e['playerId'][1]),move_from_pattern(pattern(e['pattern']),lv)))
        elif e['type']=='player_passed':events.append(('pass',int(e['playerId'][1])))
        elif e['type'] in ('tribute_given','tribute_returned'):events.append(('tribute' if e['type']=='tribute_given' else 'return',int(e['from'][1]),card(e['card']),int(e['to'][1])))
        else:raise ValueError('Unknown input event')
    return dict(player=me,level=lv,hand=[card(c) for c in v['hand']],
                left=[len(v['hand']) if p==me else v['otherHandCounts']['P'+str(p)] for p in range(4)],
                done=['P'+str(p) in v['finishOrder'] for p in range(4)],events=events,
                lead=move_from_pattern(pattern(v['currentTrick']['currentPattern']),lv),
                legal=[move_from_pattern(pattern(p),lv) for p in case['patterns']])
def run(source, model, output):
    data=json.loads(Path(source).read_text());net=NumpyModel(model)
    with Path(output).open('w') as f:
        for i,case in enumerate(data['cases']):
            toks,features=encode_decision(observation(case));q=net.q_values(toks,features)
            if not np.isfinite(q).all():raise ValueError('Non-finite oracle prediction')
            f.write(json.dumps(dict(id=case['id'],tokens=toks,features=features.tolist(),q=q.tolist(),index=int(np.argmax(q))),allow_nan=False)+'\n')
            if (i+1)%100==0:print('reference',i+1,flush=True)
    print(json.dumps({'numpy':np.__version__,'cases':len(data['cases']),'modelSha256':hashlib.sha256(Path(model).read_bytes()).hexdigest()}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.cases,a.model,a.out)
