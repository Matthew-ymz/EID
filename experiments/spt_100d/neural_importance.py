"""6D neural TM with explicit full-support importance sampling for marginal channels.

Proposal uses same-coordinate Y as a proxy for phase X at a short time horizon;
no hidden group identities enter training, proposals, or optimization.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,sys,time
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('pilot2',P.with_name('nonlinear_pilot.py'));pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
import numpy as np
import torch
from scipy.special import ndtr,ndtri,logsumexp
from eid import UniformBox
from eid.estimators.neural import NeuralEIOracle

class ImportanceOracle(NeuralEIOracle):
 def _marginal(self,part,group_ids):
  channel,x,y=part['channel'],part['x'],part['y'];D=x.shape[1]
  cols=[i for g in group_ids for i in part['positions'][g]]
  missing=[i for i in range(D) if i not in cols]
  if not missing:return channel.log_prob(y,x)
  lo=-math.pi;hi=math.pi; width=hi-lo;sd=.40;weight=.85;M=self.marginal_samples;chunk=max(1,self.integration_batch_size//M);allrows=[];rng=np.random.default_rng(6060)
  for start in range(0,len(x),chunk):
   stop=min(start+chunk,len(x));batch=stop-start;loc=np.clip(y[start:stop],lo,hi);l=ndtr((lo-loc)/sd);r=ndtr((hi-loc)/sd);z=r-l
   u=rng.random((batch,M,D));u2=rng.random((batch,M,D));gauss=loc[:,None,:]+sd*ndtri(l[:,None,:]+u*(z[:,None,:]));uni=lo+width*u;draws=np.where(u2<weight,gauss,uni)
   loggauss=-.5*((draws-loc[:,None,:])/sd)**2-np.log(sd*math.sqrt(2*math.pi))-np.log(z[:,None,:]); q=weight*np.exp(loggauss)+(1-weight)/width;logweights=np.sum(-math.log(width)-np.log(q[:,:,missing]),axis=2)
   if cols:draws[:,:,cols]=x[start:stop,None,cols]
   tiled_y=np.repeat(y[start:stop],M,axis=0);logs=channel.log_prob(tiled_y,draws.reshape(-1,D)).reshape(batch,M)+logweights
   allrows.append(logsumexp(logs,axis=1)-math.log(M))
  return np.concatenate(allrows)

def run(draws):
 torch.set_num_threads(1);start=time.monotonic();cfg=pilot.base.mk_system(6,101,0);x,y=pilot.base.data(cfg,4096,10101)
 o=ImportanceOracle(x[:3276],y[:3276],UniformBox([[-math.pi,math.pi]]*6),[[i] for i in range(6)],evaluation_count=128,marginal_samples=draws,integration_batch_size=8192,seed=101,context_transform=pilot.context,epochs=100,hidden_features=64,layers=2,batch_size=256,threads=1,context_selection=False)
 a=o.ei(tuple(range(6)));left=o.ei((0,1,2));right=o.ei((3,4,5));syn=a-left-right
 return {'draws':draws,'evaluation_count':128,'proposal':'per-coordinate 85% truncated normal centered at same-coordinate Y (sd=0.4) + 15% uniform, exact importance weights','all_nats':a,'left_nats':left,'right_nats':right,'syn_nats':syn,'tol_nats':.06931471805599453,'valid_split':syn>=-.06931471805599453,'elapsed_seconds':time.monotonic()-start}

if __name__=='__main__':
 with (ROOT/'docs/log/spt_100d_stress/results/neural_importance.jsonl').open('w') as f:
  for M in (128,1024,8192):
   v=run(M);f.write(json.dumps(v)+'\n');f.flush();print(json.dumps(v),flush=True)
