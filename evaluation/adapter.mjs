// Evaluation-only FableDan policy. No training source is copied into the game.
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {isAbsolute} from 'node:path';
import {pathToFileURL} from 'node:url';
import {FableDanModel} from './inference.mjs';
import {appObservation,encodeObservation,eventFromApp,levelIndex,cardId,orderOf,seat} from './encoding.mjs';
export async function createEvaluationBot({modelFiles,options}){
 if(typeof options?.rulesEntry!=='string'||!isAbsolute(options.rulesEntry))throw Error('Authoritative rulesEntry is required');
 if(!Array.isArray(options.rulesFiles)||!options.rulesFiles.length||!options.rulesFiles.some(f=>f.path===options.rulesEntry))throw Error('Pinned authoritative runtime is required');
 for(const f of options.rulesFiles)if(createHash('sha256').update(readFileSync(f.path)).digest('hex')!==f.sha256)throw Error('Authoritative rules runtime drift');
 const {classifyCards}=await import(pathToFileURL(options.rulesEntry));
 if(typeof classifyCards!=='function')throw Error('Invalid rules classifier');
 const file=modelFiles.find(f=>f.name.endsWith('.node.json'));if(!file)throw Error('Missing Node model artifact');
 const model=new FableDanModel(JSON.parse(readFileSync(file.path,'utf8')));
 let events=[],level=null,viewer=null;
 return {
  name:'fabledan-suits-v1',version:'0.1.0-eval',
  onGameStarted(view){events=[];level=levelIndex(view.levelRank);viewer=view.viewFor;seat(viewer);},
  onGameEvent(event){if(level===null)throw Error('Missing lifecycle start');const e=eventFromApp(event,level);if(e)events.push(e);},
  onGameEnded(){events=[];level=null;viewer=null;},
  async decideMove({view,legalMoves}){
   if(view.viewFor!==viewer||levelIndex(view.levelRank)!==level)throw Error('Lifecycle/view mismatch');
   if(!legalMoves.length)throw Error('No legal move');
   if(view.phase==='TRIBUTE'){
    // Explicit fixed tribute policy; the action-value model has not learned this phase.
    const hand=new Map(view.hand.map(c=>[c.id,c]));
    if(legalMoves.some(m=>!['tribute','returnTribute'].includes(m.type)||m.type!==legalMoves[0].type))throw Error('Invalid tribute actions');
    const sorted=[...legalMoves].sort((a,b)=>{
     if(!['tribute','returnTribute'].includes(a.type)||a.type!==b.type)throw Error('Invalid tribute actions');
     const value=m=>{const c=cardId(hand.get(m.cardId));const r=c%54>=52?c%54-39:Math.floor(c%54/4);return orderOf(r,level);};
     return (a.type==='tribute'?-1:1)*(value(a)-value(b));
    });return {type:'move',move:sorted[0]};
   }
   const encoded=encodeObservation(appObservation(view,legalMoves,events,classifyCards));
   const q=model.qValues(encoded.tokens,encoded.features);let index=0;
   for(let i=1;i<q.length;i++)if(q[i]>q[index])index=i;
   return {type:'move',move:legalMoves[index]};
  }
 };
}
