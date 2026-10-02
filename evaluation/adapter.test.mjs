import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {cardId,moveFromPattern,encodeObservation,SCHEMA} from './encoding.mjs';
import {FableDanModel} from './inference.mjs';
const modelPath=process.env.FABLEDAN_MODEL;
const data=()=>JSON.parse(readFileSync(modelPath,'utf8'));
const card=(deck,suit,rank)=>({id:`${deck}-${suit}-${rank}`,deck,suit,rank,kind:'suited'});

test('both physical heart copies remain wild and suit counts are preserved',()=>{
 const cards=[card(0,'H','2'),card(1,'H','2')];
 assert.deepEqual(cards.map(cardId),[4,58]);
 const move=moveFromPattern({type:'pair',primaryRank:15,cards},1);
 const e=encodeObservation({player:0,level:1,hand:cards.map(cardId),left:[2,27,27,27],done:[false,false,false,false],events:[],lead:null,legal:[move]});
 assert.equal(e.features[0][15],1);assert.equal(e.features[0][65],1);assert.equal(e.features[0][84],1);
 const other=encodeObservation({player:0,level:1,hand:[5,58],left:[2,27,27,27],done:[false,false,false,false],events:[],lead:null,legal:[{...move,cards:[5,58]}]});
 assert.notDeepEqual(Array.from(e.features[0]),Array.from(other.features[0]));
});

test('full house preserves the authoritative triple and both-wild level attachment',()=>{
 const naturals=[card(0,'S','K'),card(0,'D','K'),card(0,'C','K')];
 const move=moveFromPattern({type:'full_house',primaryRank:13,cards:[...naturals,card(0,'H','2'),card(1,'H','2')]},1);
 assert.deepEqual(move.claim_ranks,[12,12,12,1,1]);
 assert.throws(()=>moveFromPattern({type:'unknown',cards:[]},1),/Unsupported/);
});

test('real Transformer checkpoint shape/schema/nonfinite guards',{skip:!modelPath},()=>{
 const net=new FableDanModel(data());assert.equal(net.cfg.schema_version,SCHEMA);
 const bad=data();bad.config.schema_version='old';assert.throws(()=>new FableDanModel(bad),/schema/);
 const shape=data();shape.weights['token_emb.weight'].shape=[1,1];assert.throws(()=>new FableDanModel(shape),/weight/);
 const nan=data();nan.weights['final_norm.weight'].data[0]=Infinity;assert.throws(()=>new FableDanModel(nan),/weight/);
 assert.throws(()=>net.qValues([103],[new Float32Array(404)]),/tokens/);
 assert.throws(()=>net.qValues([1],[[NaN]]),/features/);
});

test('actual weights determine action values and maximum; no rule-score substitute',{skip:!modelPath},()=>{
 const original=new FableDanModel(data()),changed=data();
 const prefix='q_head.'+2*changed.config.n_q_layers;
 changed.weights[prefix+'.weight'].data=changed.weights[prefix+'.weight'].data.map(x=>-x);
 changed.weights[prefix+'.bias'].data=changed.weights[prefix+'.bias'].data.map(x=>-x);
 const inverse=new FableDanModel(changed),features=Array.from({length:6},(_,i)=>Float32Array.from({length:404},(_,j)=>((i*7+j*3)%19)/19));
 const q=original.qValues([1,3],features),r=inverse.qValues([1,3],features);
 for(let i=0;i<q.length;i++)assert.ok(Math.abs(q[i]+r[i])<1e-7);
 assert.notEqual(q.indexOf(Math.max(...q)),r.indexOf(Math.max(...r)));
});
