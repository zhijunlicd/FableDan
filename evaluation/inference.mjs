// CPU Float32 implementation of model_np.py, under the original FableDan LICENSE.
import {SCHEMA,FEAT_DIM,VOCAB,MAX_SEQ} from './encoding.mjs';
const matrix=(n,m)=>Array.from({length:n},()=>new Float32Array(m));
const linear=(x,w,b)=>x.map(row=>{
 const out=new Float32Array(w.length);
 for(let j=0;j<w.length;j++){let sum=b?b[j]:0;for(let k=0;k<row.length;k++)sum+=row[k]*w[j][k];out[j]=sum;}
 return out;
});
const rms=(x,w)=>x.map(row=>{let sum=0;for(const a of row)sum+=a*a;const scale=1/Math.sqrt(sum/row.length+1e-6);return Float32Array.from(row,(a,j)=>a*scale*w[j]);});
const add=(a,b)=>a.map((row,i)=>Float32Array.from(row,(v,j)=>v+b[i][j]));
export class FableDanModel {
 constructor(data){
  if(data?.format!=='fabledan-node-v1')throw Error('Unsupported model format');
  const c=data.config;
  if(c?.schema_version!==SCHEMA||c.feat_dim!==FEAT_DIM||c.vocab!==VOCAB||c.max_seq!==MAX_SEQ)throw Error('Incompatible model schema');
  for(const k of ['d_model','n_blocks','n_heads','qk_dim','v_dim','ffn_hidden','hand_hidden','n_hand_layers','q_hidden','n_q_layers'])if(!Number.isSafeInteger(c[k])||c[k]<1||c[k]>2048)throw Error('Invalid model dimensions');
  if(c.qk_dim%2)throw Error('RoPE dimension must be even');
  this.cfg=c;this.w={};const declared=new Set();
  const take=(name,shape)=>{
   const value=data.weights?.[name];
   if(!value||JSON.stringify(value.shape)!==JSON.stringify(shape)||!Array.isArray(value.data)||value.data.length!==shape.reduce((a,b)=>a*b,1)||value.data.some(v=>typeof v!=='number'||!Number.isFinite(v)||!Number.isFinite(Math.fround(v))))throw Error('Invalid model weight: '+name);
   declared.add(name);this.w[name]=shape.length===1?Float32Array.from(value.data):Array.from({length:shape[0]},(_,i)=>Float32Array.from(value.data.slice(i*shape[1],(i+1)*shape[1])));
  };
  take('rope_cos',[MAX_SEQ,c.qk_dim/2]);take('rope_sin',[MAX_SEQ,c.qk_dim/2]);take('token_emb.weight',[VOCAB,c.d_model]);
  for(let i=0;i<c.n_blocks;i++){
   const p=`blocks.${i}.`;
   take(p+'attn_norm.weight',[c.d_model]);take(p+'ffn_norm.weight',[c.d_model]);
   for(const proj of ['q','k'])take(p+`attn.${proj}_proj.weight`,[c.n_heads*c.qk_dim,c.d_model]);
   take(p+'attn.v_proj.weight',[c.n_heads*c.v_dim,c.d_model]);take(p+'attn.out_proj.weight',[c.d_model,c.n_heads*c.v_dim]);
   for(const n of ['q','k'])take(p+`attn.${n}_norm.weight`,[c.qk_dim]);
   for(const n of ['gate','up'])take(p+`ffn.${n}_proj.weight`,[c.ffn_hidden,c.d_model]);
   take(p+'ffn.down_proj.weight',[c.d_model,c.ffn_hidden]);
  }
  take('final_norm.weight',[c.d_model]);
  const mlp=(prefix,input,hidden,layers,output)=>{
   for(let i=0;i<=layers;i++){const out=i===layers?output:hidden;take(prefix+2*i+'.weight',[out,input]);take(prefix+2*i+'.bias',[out]);input=out;}
  };
  mlp('hand_mlp.',FEAT_DIM,c.hand_hidden,c.n_hand_layers,c.d_model);mlp('q_head.',2*c.d_model,c.q_hidden,c.n_q_layers,1);
  if(Object.keys(data.weights).some(k=>!declared.has(k)))throw Error('Unexpected inference weight');
 }
 mlp(x,prefix,layers){
  for(let i=0;i<=layers;i++){
   x=linear(x,this.w[prefix+2*i+'.weight'],this.w[prefix+2*i+'.bias']);
   if(i<layers)for(const row of x)for(let j=0;j<row.length;j++)row[j]=Math.max(0,row[j]);
  }return x;
 }
 attention(x,i){
  const c=this.cfg,w=this.w,p=`blocks.${i}.attn.`,T=x.length;
  const q=linear(x,w[p+'q_proj.weight']),k=linear(x,w[p+'k_proj.weight']),v=linear(x,w[p+'v_proj.weight']);
  const out=matrix(T,c.n_heads*c.v_dim);
  for(let h=0;h<c.n_heads;h++){
   const rotate=(rows,name)=>rms(rows.map(row=>row.slice(h*c.qk_dim,(h+1)*c.qk_dim)),w[p+name+'_norm.weight']).map((row,t)=>{
    const d=c.qk_dim/2,r=new Float32Array(c.qk_dim);
    for(let j=0;j<d;j++){r[j]=row[j]*w.rope_cos[t][j]-row[j+d]*w.rope_sin[t][j];r[j+d]=row[j]*w.rope_sin[t][j]+row[j+d]*w.rope_cos[t][j];}return r;
   });
   const Q=rotate(q,'q'),K=rotate(k,'k');
   for(let t=0;t<T;t++){
    const scores=new Float32Array(t+1);let max=-Infinity;
    for(let s=0;s<=t;s++){let sum=0;for(let j=0;j<c.qk_dim;j++)sum+=Q[t][j]*K[s][j];scores[s]=sum/Math.sqrt(c.qk_dim);max=Math.max(max,scores[s]);}
    let total=0;for(let s=0;s<=t;s++){scores[s]=Math.exp(scores[s]-max);total+=scores[s];}
    for(let j=0;j<c.v_dim;j++){let sum=0;for(let s=0;s<=t;s++)sum+=scores[s]/total*v[s][h*c.v_dim+j];out[t][h*c.v_dim+j]=sum;}
   }
  }return linear(out,w[p+'out_proj.weight']);
 }
 context(tokens){
  const c=this.cfg,w=this.w;let x=tokens.map(t=>w['token_emb.weight'][t]);
  for(let i=0;i<c.n_blocks;i++){
   const p=`blocks.${i}.`;x=add(x,this.attention(rms(x,w[p+'attn_norm.weight']),i));
   const h=rms(x,w[p+'ffn_norm.weight']),g=linear(h,w[p+'ffn.gate_proj.weight']),u=linear(h,w[p+'ffn.up_proj.weight']);
   const act=g.map((row,t)=>Float32Array.from(row,(v,j)=>v/(1+Math.exp(-v))*u[t][j]));
   x=add(x,linear(act,w[p+'ffn.down_proj.weight']));
  }return rms(x,w['final_norm.weight']).at(-1);
 }
 qValues(tokens,features){
  if(!Array.isArray(tokens)||!tokens.length||tokens.length>MAX_SEQ||tokens.some(t=>!Number.isInteger(t)||t<0||t>=VOCAB))throw Error('Invalid model tokens');
  if(!Array.isArray(features)||!features.length||features.length>50000||features.some(f=>f.length!==FEAT_DIM||Array.from(f).some(v=>!Number.isFinite(v))))throw Error('Invalid action features');
  const ctx=this.context(tokens),h=this.mlp(features,'hand_mlp.',this.cfg.n_hand_layers);
  const q=this.mlp(h.map(row=>Float32Array.from([...ctx,...row])),'q_head.',this.cfg.n_q_layers).map(row=>row[0]);
  if(q.some(v=>!Number.isFinite(v)))throw Error('Non-finite model output');return q;
 }
}
