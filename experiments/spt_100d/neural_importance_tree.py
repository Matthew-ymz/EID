"""Audit all six-source Syn splits before building a shared-channel neural TM tree."""
import importlib.util,json,math,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[2]
spec=importlib.util.spec_from_file_location('imp',P.with_name('neural_importance.py'));imp=importlib.util.module_from_spec(spec);spec.loader.exec_module(imp)
import numpy as np
import torch
from eid import UniformBox

def run(M):
 start=time.monotonic();p={'samples':4096,'n':6,'draws':M,'evaluation_count':128,'syn_tolerance_nats':.06931471805599453,'method':'shared neural conditional TM, full-support importance marginalization','uses_planted_groups':False}
 try:
  torch.set_num_threads(1);cfg=imp.pilot.base.mk_system(6,101,0);x,y=imp.pilot.base.data(cfg,4096,10101)
  o=imp.ImportanceOracle(x[:3276],y[:3276],UniformBox([[-math.pi,math.pi]]*6),[[i] for i in range(6)],evaluation_count=128,marginal_samples=M,integration_batch_size=8192,seed=101,context_transform=imp.pilot.context,epochs=100,hidden_features=64,layers=2,batch_size=256,threads=1,context_selection=False)
  A=imp.pilot.Adapter(o);table={0:0.};table.update({i:A.ei(i) for i in range(1,64)});values=[]
  for parent in range(1,64):
   if parent.bit_count()<2:continue
   for s in range(1,parent):
    if s&parent==s and s&(parent&-parent) and s!=parent:values.append(table[parent]-table[s]-table[parent^s])
  v=np.array(values);tol=p['syn_tolerance_nats'];p['audit']={'minimum_syn_nats':float(v.min()),'negative_within_tol':int(np.sum((v<0)&(v>=-tol))),'negative_below_tol':int(np.sum(v< -tol)),'count':len(v),'threshold':-tol}
  p['ei_nats']={str(k):v for k,v in table.items()}
  if p['audit']['negative_below_tol']:p['status']='invalid_syn';p['error']=f"minimum={v.min()}, threshold={-tol}, affected={p['audit']['negative_below_tol']}"
  else:
   decisions=[]
   def tree(m):
    if m.bit_count()==1:return {'mask':m,'children':[],'syn':0.}
    l,r,val,_=imp.pilot.base.pilot.queyranne_split(A,m);best,_,_=imp.pilot.base.exhaustive(A,m);decisions.append({'p':m,'l':l,'r':r,'syn':val,'optimality_gap':val-best});return {'mask':m,'children':[tree(l),tree(r)],'syn':val}
   t=tree(63);p['scores']=imp.pilot.base.modules(t,cfg);p['max_optimality_gap']=max(z['optimalimality_gap'] if 'optimalimality_gap' in z else z['optimality_gap'] for z in decisions);p['tree']=imp.pilot.base.serial_tree(t);p['status']='complete'
 except Exception as e:p.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 p['elapsed_seconds']=time.monotonic()-start;return p

if __name__=='__main__':
 out=ROOT/'docs/log/spt_100d_stress/results/neural_importance_tree.jsonl'
 with out.open('w') as f:
  for M in (2048,8192):
   v=run(M);f.write(json.dumps(v,allow_nan=False)+'\n');f.flush();print(json.dumps({k:z for k,z in v.items() if k not in ('ei_nats','tree','traceback')}),flush=True)
   if v['status']=='complete':break
