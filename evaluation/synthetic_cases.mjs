// Independently authored numerical fixtures; no private game code or game-generated snapshots.
// The action subsets exercise encoding, not exhaustive legality or playing strength.
import {mkdirSync,writeFileSync} from 'node:fs';
import {dirname} from 'node:path';
const output=process.argv[2];if(!output)throw Error('Usage: synthetic_cases.mjs OUTPUT');
const ranks=['A','2','3','4','5','6','7','8','9','10','J','Q','K'];
const card=(rank,suit='D',deck=0)=>({id:`${deck}-${suit}-${rank}`,deck,kind:'suited',suit,rank});
const joker=(type,deck)=>({id:`${deck}-${type}`,deck,kind:'joker',jokerType:type});
const cases=[],types=new Set();
for(const [index,levelRank] of ranks.entries()){
 const primary=rank=>rank===levelRank?15:rank==='A'?14:Number(rank);
 const same=(rank,suits)=>suits.map(suit=>card(rank,suit));
 const patterns=[
  {type:'single',primaryRank:primary('3'),cards:[card('3')]},
  {type:'pair',primaryRank:primary('3'),cards:same('3',['D','C'])},
  {type:'triple',primaryRank:primary('5'),cards:same('5',['D','S','C'])},
  {type:'full_house',primaryRank:primary('5'),cards:[...same('5',['D','S','C']),...same('3',['D','C'])]},
  {type:'bomb',primaryRank:primary('8'),cards:same('8',['H','D','S','C'])},
  {type:'straight',primaryRank:7,cards:['3','4','5','6','7'].map((r,i)=>card(r,i%2?'C':'D'))},
  {type:'straight_flush',primaryRank:7,cards:['3','4','5','6','7'].map(r=>card(r,'S'))},
  {type:'tube',primaryRank:5,cards:['3','4','5'].flatMap(r=>same(r,['D','C']))},
  {type:'plate',primaryRank:4,cards:['3','4'].flatMap(r=>same(r,['D','S','C']))},
  {type:'joker_bomb',primaryRank:18,cards:[joker('BJ',0),joker('BJ',1),joker('RJ',0),joker('RJ',1)]},
  {type:'pair',primaryRank:15,cards:[card(levelRank,'H',0),card(levelRank,'H',1)]}
 ];
 for(const [j,p] of patterns.entries()){
  const viewer='P'+(index+j)%4,ace={type:'single',primaryRank:primary('A'),cards:[card('A','C')]};
  const hand=[...p.cards,...ace.cards],view={phase:'PLAYING',viewFor:viewer,levelRank,hand,
   otherHandCounts:Object.fromEntries([0,1,2,3].filter(i=>'P'+i!==viewer).map(i=>['P'+i,27])),
   finishOrder:[],revealedHands:{},currentTrick:{currentPattern:null}};
  const declarations=[p,ace].map(p=>({...p,size:p.cards.length,wildCardsUsed:[]}));
  cases.push({id:`synthetic-${index}-${j}`,view,patterns:declarations,
   legalMoves:declarations.map(p=>({type:'play',playerId:viewer,cardIds:p.cards.map(c=>c.id)})),events:[]});
  types.add(p.type);
 }
}
const base=structuredClone(cases[0]);base.id='synthetic-long-history';
base.events=[{type:'tribute_given',from:'P0',to:'P2',card:joker('RJ',0)},
 {type:'tribute_returned',from:'P2',to:'P0',card:card('3')},
 ...Array.from({length:600},(_,i)=>({type:'player_passed',playerId:'P'+i%4}))];cases.push(base);
const follow=structuredClone(cases[0]);follow.id='synthetic-pass';follow.view.currentTrick.currentPattern={type:'single',primaryRank:13,cards:[card('K')]};follow.legalMoves=[{type:'pass',playerId:follow.view.viewFor}];follow.patterns=[null];cases.push(follow);
mkdirSync(dirname(output),{recursive:true});
writeFileSync(output,JSON.stringify({types:[...types].sort(),scope:'Independent synthetic encoding inputs; not authoritative game-worker validation',cases})+'\n');
console.log(JSON.stringify({cases:cases.length,types:[...types].sort()}));
