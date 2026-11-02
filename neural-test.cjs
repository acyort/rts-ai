const fs=require('fs'),vm=require('vm'),assert=require('assert');let depth=0;const noop=()=>{};
const ctx=new Proxy({save(){depth++},restore(){assert(depth>0);depth--}},{get:(o,k)=>o[k]||noop,set(o,k,v){o[k]=v;return true}});
const cls={toggle:noop,add:noop,remove:noop};
function el(){return {style:{setProperty:noop},classList:cls,parentElement:{classList:cls,getBoundingClientRect:()=>({width:1600,height:900})},querySelector:()=>el(),setAttribute:noop,append:noop,appendChild:noop,replaceChildren:noop,addEventListener:noop,getContext:()=>ctx,getBoundingClientRect:()=>({left:0,top:0,width:1600,height:900}),matches:()=>false};}
const elems=new Map(),document={querySelector(id){if(!elems.has(id))elems.set(id,el());return elems.get(id)},createElement:el,addEventListener:noop};
const box={performance,localStorage:{data:{},getItem(k){return this.data[k]||null},setItem(k,v){this.data[k]=v}},setTimeout:()=>0,clearTimeout:noop,console,document,ResizeObserver:class{observe(){}},requestAnimationFrame:noop,Math};
let src=fs.readFileSync(require('path').join(__dirname,'game.js'),'utf8').replace('    setupBuildingButtons();',`globalThis.t={recordDamage,checkVictory,neuralForward,neuralGradient,neuralObservation,neuralMask,chooseNeural,settleNeural,validNetwork,makeNetwork,NN_ACTIONS,getModel:()=>neural,getPending:()=>neuralPending,saveNeural,createHelicopter,createFighter,armyStrength,combat,chooseCounter,aiAttack,toggleTraining,finishTrainingRound,learning,createRocket,trainingState:()=>training,units,buildings,mines,cash,buildQueues,BUILD_TIME,powerState,productionRate,enqueueBuilding,cancelBuilding,updateBuildingProduction,construct,hasTech,building,createTank,createMG,createAntiAir,createHarvester,issueFormation,issueMove,movement,rebuildSpatial,updateMining,updateProduction,enqueue,q,update,resetMatch,draw,preview,canTarget,staticFree,center,setTopology:()=>topology++};\n    setupBuildingButtons();`);
vm.runInNewContext(src,box);const t=box.t;

const model=t.getModel();assert(t.validNetwork(model));assert.equal(t.neuralObservation().length,61);
const input=Array.from({length:61},(_,i)=>Math.sin(i)*.2),mask=[true,true,false,true,false,true,true],f=t.neuralForward(input,mask);
assert(Math.abs(f.p.reduce((a,b)=>a+b,0)-1)<1e-12);assert.equal(f.p[2],0);assert.equal(f.p[4],0);
const action=1,adv=.4,target=f.value+adv,g=t.neuralGradient(f,action,adv);
function objective(){const f=t.neuralForward(input,mask);return adv*Math.log(f.p[action])-.01*f.p.reduce((n,p)=>n+(p?p*Math.log(p):0),0)-.25*(target-f.value)**2;}
for(const [l,j,i] of [[0,3,7],[1,4,6],[2,1,4],[2,7,3],[2,2,2]]){const old=model.weights[l][j][i],eps=1e-5;model.weights[l][j][i]=old+eps;const a=objective();model.weights[l][j][i]=old-eps;const b=objective();model.weights[l][j][i]=old;assert(Math.abs((a-b)/(2*eps)-g.dw[l][j][i])<1e-6,'finite-difference backprop check');}
console.log('PASS 2-hidden-layer gradients, masked softmax, schema');
t.toggleTraining();const before=JSON.stringify(model.weights),updates=model.updates;
for(let i=0;i<620*60;i++)t.update(1/60);
assert(model.episodes>=2);assert(model.updates>updates);assert.notEqual(JSON.stringify(model.weights),before);assert(t.validNetwork(model));
const positions=t.units.map(u=>[u.id,u.x,u.y]);t.toggleTraining();assert.equal(t.units.length,0,'stop starts fresh match');assert.equal(t.buildings.length,2);assert.equal(t.cash[0],1600);const saved=JSON.parse(box.localStorage.data['rts-neural-actor-critic-v1']);assert(t.validNetwork(saved));assert.deepEqual(saved.weights,JSON.parse(JSON.stringify(model.weights)));const result=elems.get('#trainingResults');assert(!result.hidden);assert(result.textContent.includes('深度AI'));assert(result.textContent.includes('規則AI'));assert.equal(t.getPending(),null);
console.log(JSON.stringify({passed:true,episodes:model.episodes,updates:model.updates,summary:result.textContent}));
// Mask permits only wait with no cash; no unaffordable production.
t.cash[1]=0;const m=t.neuralMask();assert(m.slice(0,6).every(v=>!v));assert(m[6]);assert.equal(t.chooseNeural(),null);
console.log('PASS real matches update weights, stop/save summary, invalid action masking');

const oldWeights=JSON.stringify(model.weights),oldUpdates=model.updates;t.cash[1]=10000;
for(let i=0;i<12;i++)t.chooseNeural();assert(model.updates>oldUpdates);assert.notEqual(JSON.stringify(model.weights),oldWeights);assert(t.getPending());
assert(JSON.parse(box.localStorage.data['rts-neural-actor-critic-v1']).updates>=oldUpdates);
const episodes=model.episodes;t.buildings.find(b=>b.team===0).hp=0;t.checkVictory();assert.equal(model.episodes,episodes+1);assert.equal(t.getPending(),null);
const endedUpdates=model.updates;t.checkVictory();assert.equal(model.updates,endedUpdates);
t.resetMatch();t.chooseNeural();const beforeRestart=model.updates;t.resetMatch();assert.equal(model.updates,beforeRestart+1);assert.equal(t.getPending(),null);
console.log('PASS player-match learning, periodic saves, terminal settlement and restart');

const restored=JSON.parse(box.localStorage.data['rts-neural-actor-critic-v1']);vm.runInNewContext(src,box);assert.deepEqual(JSON.parse(JSON.stringify(box.t.getModel().weights)),restored.weights);console.log('PASS checkpoint reload and live takeover');
