// Guandan-suits-v1 encoding, ported from fabledan/encode.py and app_adapter.py.
// FableDan LICENSE applies; training and game runtime remain separate.
export const SCHEMA='guandan-suits-v1',FEAT_DIM=404,VOCAB=103,MAX_SEQ=512;
const ranks=['A','2','3','4','5','6','7','8','9','10','J','Q','K'];
const suits=['H','D','S','C'];
export const seat=p=>{if(!/^P[0-3]$/.test(p))throw Error('Invalid player');return Number(p[1]);};
export const levelIndex=rank=>{const i=ranks.indexOf(rank);if(i<0)throw Error('Invalid level');return i;};
export function cardId(c){
 if(c?.deck!==0&&c?.deck!==1)throw Error('Invalid physical card');
 const rank=c.kind==='joker'?(c.jokerType==='BJ'?52:c.jokerType==='RJ'?53:-1):ranks.indexOf(c.rank)*4+suits.indexOf(c.suit);
 if(rank<0||rank>53||c.kind==='suited'&&(!ranks.includes(c.rank)||!suits.includes(c.suit)))throw Error('Invalid physical card');
 return c.deck*54+rank;
}
const rankOf=c=>c%54>=52?c%54-39:Math.floor(c%54/4);
const wild=(c,lv)=>rankOf(c)===lv&&c%54%4===0&&c%54<52;
export function orderOf(rank,lv){return [...Array.from({length:12},(_,i)=>i+1),0].filter(r=>r!==lv).concat(lv,13,14).indexOf(rank);}
const TYPES={single:1,pair:2,triple:3,full_house:4,straight:5,tube:6,plate:7,bomb:8,straight_flush:9,joker_bomb:10};
const primaryRank=(value,lv)=>value===15?lv:value===16?13:value===17?14:value===14?0:value-1;
const pass=()=>({type:0,key:0,cards:[],claim_ranks:[],size:0});
export function moveFromPattern(pattern,lv){
 if(pattern===null)return pass();
 const type=TYPES[pattern?.type];if(!type)throw Error('Unsupported authoritative pattern');
 const cards=pattern.cards.map(cardId),primary=pattern.primaryRank;
 const key=[5,9].includes(type)?primary-4:type===6?primary-2:type===7?primary-1:type===10?0:orderOf(primaryRank(primary,lv),lv);
 let claims;
 if([1,2,3,8].includes(type))claims=Array(cards.length).fill(primaryRank(primary,lv));
 else if(type===10)claims=[13,13,14,14];
 else if(type===4){
  const triple=primaryRank(primary,lv),counts=Array(15).fill(0);for(const c of cards)if(!wild(c,lv))counts[rankOf(c)]++;
  // Matches gen_moves' ascending pair declaration, constrained to this exact play.
  const pair=Array.from({length:13},(_,r)=>r).find(r=>r!==triple&&(counts[r]||r===lv)&&counts.every((n,i)=>i===triple?n<=3:i===r?n<=2:n===0)&&5-cards.filter(c=>!wild(c,lv)).length===Math.max(0,3-counts[triple])+Math.max(0,2-counts[r]));
  if(pair===undefined)throw Error('Unsupported full-house declaration');
  claims=[triple,triple,triple,pair,pair];
 }else{
  const length=type===6?3:type===7?2:5,mult=type===6?2:type===7?3:1;
  claims=Array.from({length:length},(_,i)=>Array(mult).fill((key+i-1)%13)).flat();
 }
 if(!Number.isInteger(key)||key<0||claims.length!==cards.length||claims.some(r=>!Number.isInteger(r)||r<0||r>=15))throw Error('Invalid authoritative declaration');
 return {type,key,cards,claim_ranks:claims,size:cards.length};
}
export function eventFromApp(e,lv){
 if(e.type==='cards_played')return ['play',seat(e.playerId),moveFromPattern(e.pattern,lv)];
 if(e.type==='player_passed')return ['pass',seat(e.playerId)];
 if(e.type==='tribute_given'||e.type==='tribute_returned')return [e.type==='tribute_given'?'tribute':'return',seat(e.from),cardId(e.card),seat(e.to)];
 return null;
}
export function appObservation(view,legalMoves,events,classifyCards){
 if(view.phase!=='PLAYING')throw Error('Model only supports PLAYING decisions');
 if('hands' in view||Object.keys(view.revealedHands??{}).length)throw Error('Model input contains revealed hands');
 const level=levelIndex(view.levelRank),player=seat(view.viewFor);
 const byId=new Map(view.hand.map(c=>[c.id,c]));
 const legal=legalMoves.map(m=>{
  if(m.playerId!==view.viewFor)throw Error('Wrong legal-action seat');
  if(m.type==='pass')return pass();
  if(m.type!=='play')throw Error('Unsupported model action');
  const cards=m.cardIds.map(id=>{if(!byId.has(id))throw Error('Action card absent from hand');return byId.get(id);});
  const pattern=classifyCards(cards,view.levelRank);if(!pattern)throw Error('Unclassified legal action');
  return moveFromPattern(pattern,level);
 });
 if(!legal.length)throw Error('Empty action set');
 const left=Array.from({length:4},(_,p)=>p===player?view.hand.length:view.otherHandCounts['P'+p]);
 if(left.some(n=>!Number.isInteger(n)||n<0||n>27))throw Error('Invalid public card counts');
 const done=Array.from({length:4},(_,p)=>view.finishOrder.includes('P'+p));
 return {player,level,hand:view.hand.map(cardId),left,done,events,lead:moveFromPattern(view.currentTrick?.currentPattern??null,level),legal};
}
export function encodeObservation(obs){
 const {player:me,level:lv,events}=obs;
 let tokens=[1,2+lv];
 for(const e of events){
  const p=(e[1]-me+4)%4;
  if(e[0]==='pass')tokens.push(15+p,19);
  else if(e[0]==='play')tokens.push(15+p,19+e[2].type,...[...e[2].claim_ranks].sort((a,b)=>a-b).map(r=>32+r),102,...[...e[2].cards].sort((a,b)=>a%54-b%54).map(c=>48+c%54));
  else if(e[0]==='tribute'||e[0]==='return')tokens.push(15+p,e[0]==='tribute'?30:31,48+e[2]%54,15+(e[3]-me+4)%4);
  else throw Error('Unsupported public event');
 }
 if(tokens.length>MAX_SEQ)tokens=tokens.slice(0,2).concat(tokens.slice(-(MAX_SEQ-2)));
 const features=obs.legal.map(move=>{
  const f=new Float32Array(FEAT_DIM);let i=0;
  for(const c of obs.hand)f[rankOf(c)]+=.25;i+=15;
  f[i++]=obs.hand.filter(c=>wild(c,lv)).length/2;f[i++]=obs.hand.length/27;
  for(let r=0;r<4;r++)f[i+r]=obs.left[(me+r)%4]/27;i+=4;
  for(let r=0;r<4;r++)f[i+r]=Number(obs.done[(me+r)%4]);i+=4;
  f[i+lv]=1;i+=13;f[i+move.type]=1;i+=11;
  for(const r of move.claim_ranks)f[i+r]+=.25;i+=15;
  f[i++]=move.size/27;f[i++]=move.cards.filter(c=>wild(c,lv)).length/2;f[i++]=move.type?move.key/15:0;
  if(obs.lead?.type){f[i+obs.lead.type]=1;f[i+11]=obs.lead.key/15;}else f[i+12]=1;i+=13;
  for(const c of obs.hand)f[i+c%54]+=.5;i+=54;
  for(const c of move.cards)f[i+c%54]+=.5;i+=54;
  for(const e of events)if(e[0]==='play')for(const c of e[2].cards)f[i+((e[1]-me+4)%4)*54+c%54]+=.5;
  return f;
 });
 return {schema:SCHEMA,tokens,features};
}
