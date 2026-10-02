// Bounded interface validation, not playing-strength evaluation.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {join,resolve,dirname} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const [gameRoot,model,out]=process.argv.slice(2);
if(!gameRoot||!model||!out)throw Error('Usage: run_validation.mjs GAME_ROOT MODEL OUT');
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..'),game=resolve(gameRoot),output=resolve(out);
mkdirSync(output,{recursive:true});
const {createPlan,verifyRuntime}=await import(pathToFileURL(join(game,'packages/bot-simple/evaluation/plan.mjs')));
const {runDirectory}=await import(pathToFileURL(join(game,'packages/bot-simple/evaluation/cli.mjs')));
const {fileTree,fileHash}=await import(pathToFileURL(join(game,'packages/bot-simple/evaluation/common.mjs')));
const rulesFiles=fileTree(join(game,'packages/rules/dist')).map(path=>({path,sha256:fileHash(path)}));
const options={rulesEntry:join(game,'packages/rules/dist/index.js'),rulesFiles};
const spec={kind:'model',root,entry:'evaluation/adapter.mjs',runtimeFiles:['evaluation/adapter.mjs','evaluation/encoding.mjs','evaluation/inference.mjs'],modelFiles:[model],options};
const summaries=[];
for(const role of ['partner','opponent']){
 const config={id:'fabledan-inference-'+role,groups:1,seed:1032027,levels:['2','A'],opponents:['baseline'],baseline:'baseline',candidateAdapter:spec,partnerProxy:'baseline',role,mode:'round',split:'development',decisionLimitMs:1000,startupLimitMs:10000,maxTurns:700,maxRounds:30};
 const m=createPlan(config),dir=join(output,role);mkdirSync(dir);
 assert.ok(rulesFiles.every(f=>m.runtimeFiles.some(r=>r.path===f.path&&r.sha256===f.sha256)),'Classifier dependencies must be frozen by evaluator');
 verifyRuntime(m);writeFileSync(join(dir,'config.json'),JSON.stringify(config,null,2)+'\n');writeFileSync(join(dir,'manifest.json'),JSON.stringify(m,null,2)+'\n');
 const report=await runDirectory(dir);assert.equal(report.completedRuns,4,JSON.stringify(report));assert.equal(report.releaseApproved,false);
 summaries.push({role,runs:report.scheduledRuns,completed:report.completedRuns,faults:report.faults,verdict:report.verdict});
 // Independent re-aggregation verifies every trace hash and preserves primary attempts.
 const rerun=await runDirectory(dir);assert.deepEqual(rerun,report);
}
const config={id:'fabledan-inference-match-A',groups:1,seed:1032028,levels:['A'],opponents:['baseline'],baseline:'baseline',candidateAdapter:spec,partnerProxy:'baseline',role:'partner',mode:'match',split:'development',decisionLimitMs:1000,startupLimitMs:10000,maxTurns:700,maxRounds:30};
const m=createPlan(config),dir=join(output,'match-A');mkdirSync(dir);verifyRuntime(m);
writeFileSync(join(dir,'config.json'),JSON.stringify(config,null,2)+'\n');writeFileSync(join(dir,'manifest.json'),JSON.stringify(m,null,2)+'\n');
const report=await runDirectory(dir);assert.equal(report.completedRuns,2,JSON.stringify(report));
const records=JSON.parse('['+readFileSync(join(dir,'runs.jsonl'),'utf8').trim().split('\n').join(',')+']');
assert.ok(records.some(r=>r.rounds>1),'Must cover lifecycle restart and tribute across rounds');
summaries.push({role:'partner/match-A',runs:2,completed:2,rounds:records.map(r=>r.rounds),faults:report.faults,verdict:report.verdict});
const result={schema:'fabledan-worker-validation-v1',summaries,trainedStrengthValidated:false,tributePolicy:'fixed; not learned',rulesFiles};
writeFileSync(join(output,'summary.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
