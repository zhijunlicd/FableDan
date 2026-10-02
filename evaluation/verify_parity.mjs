import {readFileSync,writeFileSync} from 'node:fs';
import {createInterface} from 'node:readline';
import {createReadStream} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';
import {appObservation,encodeObservation,eventFromApp,levelIndex} from './encoding.mjs';
import {FableDanModel} from './inference.mjs';
const [root,casesPath,referencePath,modelPath,out]=process.argv.slice(2);
// Synthetic mode exercises encoding against independently authored declarations, without a game checkout.
const classifyCards=root==='--synthetic'?null:(await import(pathToFileURL(join(resolve(root),'packages/rules/dist/index.js')))).classifyCards;
const data=JSON.parse(readFileSync(casesPath)),model=new FableDanModel(JSON.parse(readFileSync(modelPath)));
const reader=createInterface({input:createReadStream(referencePath),crlfDelay:Infinity});
let i=0,maxScoreError=0,maxFeatureError=0,exactArgmax=0,equivalentArgmax=0,actions=0,truncatedCases=0;const latencies=[];
for await(const line of reader){
 const expected=JSON.parse(line),c=data.cases[i++];assert.equal(expected.id,c.id);
 const before=performance.now(),lv=levelIndex(c.view.levelRank),events=c.events.map(e=>eventFromApp(e,lv)).filter(Boolean);
 const key=cards=>JSON.stringify(cards.map(c=>c.id).sort());
 const classifier=classifyCards??(cards=>c.patterns.find(p=>p&&key(p.cards)===key(cards))??null);
 const encoded=encodeObservation(appObservation(c.view,c.legalMoves,events,classifier));
 assert.deepEqual(encoded.tokens,expected.tokens,`tokens ${c.id}`);assert.equal(encoded.features.length,expected.features.length);
 for(let a=0;a<encoded.features.length;a++)for(let j=0;j<404;j++){
  const diff=Math.abs(encoded.features[a][j]-expected.features[a][j]);maxFeatureError=Math.max(maxFeatureError,diff);assert.equal(diff,0,`features ${c.id} ${a}/${j}`);
 }
 const q=model.qValues(encoded.tokens,encoded.features);assert.equal(q.length,expected.q.length);
 let index=0;for(let a=0;a<q.length;a++){
  if(q[a]>q[index])index=a;
  const error=Math.abs(q[a]-expected.q[a]);maxScoreError=Math.max(maxScoreError,error);
  assert.ok(error<=2e-5+2e-4*Math.abs(expected.q[a]),`score ${c.id}/${a}: ${q[a]} vs ${expected.q[a]}`);
 }
 exactArgmax+=Number(index===expected.index);
 assert.ok(expected.q[expected.index]-expected.q[index]<=2*(2e-5+2e-4*Math.abs(expected.q[expected.index])),`action value differs ${c.id}`);
 equivalentArgmax++;actions+=q.length;truncatedCases+=Number(encoded.tokens.length===512);latencies.push(performance.now()-before);
 if(i%100===0)console.log('parity',i);
}
assert.equal(i,data.cases.length);latencies.sort((a,b)=>a-b);
const sha=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
const report={schema:'fabledan-parity-v1',cases:i,actions,levels:new Set(data.cases.map(c=>c.view.levelRank)).size,types:data.types,classificationMode:classifyCards?'live-authority':'synthetic-fixture',maxFeatureError,maxScoreError,exactArgmax,equivalentArgmax,truncatedCases,
 latencyMs:{p50:latencies[Math.floor(.5*latencies.length)],p95:latencies[Math.floor(.95*latencies.length)],max:latencies.at(-1)},
 inputs:{cases:sha(casesPath),reference:sha(referencePath),model:sha(modelPath)},strengthValidated:false,torchRerun:false};
writeFileSync(out,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
