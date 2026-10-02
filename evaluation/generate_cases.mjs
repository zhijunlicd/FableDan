// The authoritative TS rules supply actions and classifications, independently of Python.
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
const [gameRoot,out]=process.argv.slice(2);if(!gameRoot||!out)throw Error('Usage: generate_cases.mjs GAME_ROOT OUTPUT');
const rules=await import(pathToFileURL(join(resolve(gameRoot),'packages/rules/dist/index.js')));
const {createDeck,shuffleDeck,createGame,dealCards,getLegalMoves,validateMove,applyMove,getPlayerView,classifyCards}=rules;
const fixture=JSON.parse(readFileSync(new URL('../fixtures/rule-subsets.json',import.meta.url)));
const card=(id)=>{const ranks=['A','2','3','4','5','6','7','8','9','10','J','Q','K'],suits=['H','D','S','C'],b=id%54,deck=Math.floor(id/54);return b>=52?{id:`${deck}-${b===52?'BJ':'RJ'}`,deck,kind:'joker',jokerType:b===52?'BJ':'RJ'}:{id:`${deck}-${suits[b%4]}-${ranks[Math.floor(b/4)]}`,deck,kind:'suited',suit:suits[b%4],rank:ranks[Math.floor(b/4)]};};
const lvRank=i=>['A','2','3','4','5','6','7','8','9','10','J','Q','K'][i];
const cases=[],types=new Set(),players=['P0','P1','P2','P3'];
function add(id,view,moves,events){
 const patterns=moves.map(m=>m.type==='pass'?null:classifyCards(m.cardIds.map(id=>view.hand.find(c=>c.id===id)),view.levelRank));
 if(patterns.some((p,i)=>!p&&moves[i].type==='play'))throw Error('Legal action unclassified');
 patterns.forEach(p=>{if(p)types.add(p.type);});cases.push({id,view,legalMoves:moves,patterns,events});
}
for(const [n,c] of fixture.cases.entries()){
 const hand=c.hand.map(card),levelRank=lvRank(c.level),view={phase:'PLAYING',levelRank,hand,viewFor:'P'+n%4,otherHandCounts:Object.fromEntries(players.filter(p=>p!=='P'+n%4).map(p=>[p,27])),revealedHands:{},finishOrder:[],currentTrick:{currentPattern:null}};
 const expected=new Map(c.expected.map(p=>[JSON.stringify([...p.cards].sort((a,b)=>a-b)),p]));
 const moves=[];
 for(const p of expected.values()){
  const cards=p.cards.map(card),actual=classifyCards(cards,levelRank);
  if(!actual||actual.type!==p.type||actual.primaryRank!==p.primaryRank)throw Error('Historical rule fixture drift');
  moves.push({type:'play',playerId:view.viewFor,cardIds:cards.map(c=>c.id)});
 }add('edge-'+n,view,moves,[]);
}
const jokerHand=[52,53,106,107].map(card);
const jokerView={phase:'PLAYING',levelRank:'2',hand:jokerHand,viewFor:'P3',otherHandCounts:{P0:27,P1:27,P2:27},revealedHands:{},finishOrder:[],currentTrick:{currentPattern:null}};
add('joker-bomb',jokerView,[{type:'play',playerId:'P3',cardIds:jokerHand.map(c=>c.id)}],[]);
const rankings=[];
for(let lv=0;lv<13;lv++){
 let state=dealCards(createGame({players,startingLevel:lvRank(lv)}),shuffleDeck(createDeck(),1032026+lv)).state;
 const events=[];let turns=0;
 while(state.phase==='PLAYING'){
  if(turns>=700)throw Error('Fixture round did not finish');
  const p=state.currentTrick.activePlayer,moves=getLegalMoves(state,p),view=getPlayerView(state,p);
  if(turns%2===0)add(`round-${lv}-${turns}`,view,moves,structuredClone(events));
  const plays=moves.filter(m=>m.type==='play');
  // Independent deterministic physical play, not the learned policy's own choices.
  const chosen=plays.length?plays[(turns*17+lv*31)%plays.length]:moves[0];
  const checked=validateMove(state,chosen);if(!checked.ok)throw Error('Generated illegal action');
  const next=applyMove(state,checked.validatedMove);state=next.state;
  events.push(...next.events.filter(e=>['cards_played','player_passed','tribute_given','tribute_returned'].includes(e.type)));turns++;
 }rankings.push({level:lvRank(lv),turns,finishOrder:state.finishOrder});
}
// Guaranteed public tribute history, both deck copies of wilds and long-token truncation.
const initial=cases[0];
for(let p=0;p<4;p++){
 const events=[{type:'tribute_given',from:'P0',to:'P2',card:card(53)},{type:'tribute_returned',from:'P2',to:'P0',card:card(5)},...Array.from({length:600},(_,i)=>({type:'player_passed',playerId:'P'+i%4}))];
 const base=cases.find(c=>c.view.viewFor==='P'+p);add('long-history-'+p,base.view,base.legalMoves,events);
}
writeFileSync(out,JSON.stringify({seed:1032026,types:[...types].sort(),rankings,cases})+'\n');
console.log(JSON.stringify({cases:cases.length,actions:cases.reduce((n,c)=>n+c.legalMoves.length,0),types:[...types].sort()}));
