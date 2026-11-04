"""Six-node nonlinear shared-channel TM diagnostic; no planted source groups supplied."""
import importlib.util,itertools,json,math,os,signal,sys,time,traceback
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
P=Path(__file__).resolve();ROOT=P.parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('stress_impl',P.with_name('run.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import numpy as np
import torch
from eid import UniformBox
from eid.estimators.neural import NeuralEIOracle

def context(x):
 x=np.asarray(x);cols=[x,np.sin(x),np.cos(x),np.sin(2*x),np.cos(2*x)]
 # Complete 3-body and 6-body Fourier dictionary; no planted partition.
 for order in (3,6):
  for group in itertools.combinations(range(x.shape[1]),order):
   total=x[:,group].sum(axis=1)
   for i in group:
    v=total-order*x[:,i];cols.extend([np.sin(v)[:,None],np.cos(v)[:,None]])
 return np.concatenate(cols,axis=1)

class Adapter:
 def __init__(self,oracle):self.o=oracle;self.calls=0
 def ei(self,m):return self.o.ei(tuple(i for i in range(6) if m>>i&1)) if m else 0.
 def gain(self,parts):self.calls+=1;return self.ei(sum(parts))-sum(self.ei(m) for m in parts)

def main():
 started=time.monotonic();out=ROOT/'docs/log/spt_100d_stress/results/nonlinear_pilot.json';p={'status':'running','method':'shared conditional neural TM with complete 3- and 6-body Fourier dictionary','n':6,'syn_tolerance_nats':.06931471805599453,'seed':101,'channel_factorization':False,'uses_planted_groups':False}
 try:
  signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('nonlinear pilot 600-second limit')));signal.alarm(600)
  cfg=base.mk_system(6,101,0);x,y=base.data(cfg,4096,10101);torch.set_num_threads(1)
  t=time.monotonic();o=NeuralEIOracle(x[:3276],y[:3276],UniformBox([[-math.pi,math.pi]]*6),[[i] for i in range(6)],evaluation_count=256,marginal_samples=128,seed=101,context_transform=context,epochs=100,hidden_features=64,layers=2,batch_size=256,threads=1,context_selection=False)
  p['fit_seconds']=time.monotonic()-t;p['fit_diagnostics']=o.metadata['fits'];p['samples']=4096
  A=Adapter(o);t=time.monotonic();table={i:A.ei(i) for i in range(1,64)};table[0]=0.;p['ei_all63_seconds']=time.monotonic()-t
  checks=[]
  for parent in range(1,64):
   if parent.bit_count()<2:continue
   for s in range(1,parent):
    if s&parent==s and s&(parent&-parent) and s!=parent:checks.append(table[parent]-table[s]-table[parent^s])
  v=np.array(checks);tol=p['syn_tolerance_nats'];p['audit']={'minimum_syn_nats':float(v.min()),'negative_within_tol':int(np.sum((v<0)&(v>=-tol))),'negative_below_tol':int(np.sum(v< -tol)),'count':len(v),'threshold':-tol}
  if p['audit']['negative_below_tol']:
   p['status']='invalid_syn';p['error']=f"minimum={v.min()}, threshold={-tol}, affected={p['audit']['negative_below_tol']}";return
  decisions=[]
  def tree(m):
   if m.bit_count()==1:return {'mask':m,'children':[],'syn':0.}
   l,r,val,_=base.pilot.queyranne_split(A,m);best,_,_=base.exhaustive(A,m);decisions.append({'p':m,'l':l,'r':r,'syn':val,'optimality_gap':val-best});return {'mask':m,'children':[tree(l),tree(r)],'syn':val}
  t=tree(63);p['scores']=base.modules(t,cfg);p['max_optimality_gap']=max(z['optimality_gap'] for z in decisions);p['tree']=base.serial_tree(t);p['status']='complete'
 except Exception as e:p.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 finally:
  signal.alarm(0);p['elapsed_seconds']=time.monotonic()-started;out.write_text(json.dumps(p,indent=2,ensure_ascii=False,allow_nan=False));print(json.dumps({k:v for k,v in p.items() if k not in ('tree','traceback','fit_diagnostics')}),flush=True)
if __name__=='__main__':main()
