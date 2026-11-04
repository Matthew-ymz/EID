"""Analytic 100D independent uniform-to-Gaussian channel control for importance integration."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri,logsumexp
R=Path(__file__).resolve().parents[2];out=R/'docs/log/spt_100d_stress/results/importance_gaussian_replicates.jsonl';rng=np.random.default_rng(808);n=32;dim=100;sigma=.086;width=2*math.pi;x=rng.uniform(-math.pi,math.pi,(n,dim));y=x+rng.normal(size=(n,dim))*sigma
normal=-.5*((y-x)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi));mass=ndtr((math.pi-y)/sigma)-ndtr((-math.pi-y)/sigma);analytic=np.log(np.maximum(mass,1e-300))-math.log(width);true_logpy=analytic.sum(1)
loc=np.clip(y,-math.pi,math.pi);sd=1.25*sigma;low=ndtr((-math.pi-loc)/sd);high=ndtr((math.pi-loc)/sd);Z=high-low

def marginal(active,M):
 r=np.random.default_rng(proposal_seed);missing=np.array([i for i in range(dim) if i not in active]);active=np.array(active,dtype=int);rows=[]
 for i in range(n):
  u=r.random((M,dim));choose=r.random((M,dim));a=loc[i]+sd*ndtri(low[i]+u*Z[i]);b=-math.pi+width*u;z=np.where(choose<.999,a,b);lg=-.5*((z-loc[i])/sd)**2-math.log(sd*math.sqrt(2*math.pi))-np.log(Z[i]);q=.999*np.exp(lg)+.001/width;weights=np.sum(-math.log(width)-np.log(q[:,missing]),axis=1)
  if len(active):z[:,active]=x[i,active]
  loglike=np.sum(-.5*((y[i]-z)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi)),axis=1);rows.append(logsumexp(loglike+weights)-math.log(M))
 return np.array(rows)

with out.open('w') as f:
 for M in (4096,16384):
  for proposal_seed in (6060,6061,6062,6063,6064):
   t=time.monotonic();p=marginal([],M);l=marginal(list(range(50)),M);r=marginal(list(range(50,100)),M);syn=float(np.mean(normal.sum(1)+p-l-r));row={'draws':M,'proposal_seed':proposal_seed,'evaluation_count':n,'n':dim,'true_syn_nats':0.,'estimated_syn_nats':syn,'tol_nats':0.06931471805599453,'below_tolerance':syn < -0.06931471805599453,'elapsed_seconds':time.monotonic()-t};f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
