"""Analytic 100D independent uniform-to-Gaussian channel control for importance integration."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri,logsumexp
R=Path(__file__).resolve().parents[2];out=R/'docs/log/spt_100d_stress/results/importance_gaussian_control.jsonl';rng=np.random.default_rng(808);n=32;dim=100;sigma=.086;width=2*math.pi;x=rng.uniform(-math.pi,math.pi,(n,dim));y=x+rng.normal(size=(n,dim))*sigma
normal=-.5*((y-x)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi));mass=ndtr((math.pi-y)/sigma)-ndtr((-math.pi-y)/sigma);analytic=np.log(np.maximum(mass,1e-300))-math.log(width);true_logpy=analytic.sum(1)
loc=np.clip(y,-math.pi,math.pi);sd=1.25*sigma;low=ndtr((-math.pi-loc)/sd);high=ndtr((math.pi-loc)/sd);Z=high-low

def marginal(active,M):
 r=np.random.default_rng(6060);missing=np.array([i for i in range(dim) if i not in active]);active=np.array(active,dtype=int);rows=[]
 for i in range(n):
  u=r.random((M,dim));choose=r.random((M,dim));a=loc[i]+sd*ndtri(low[i]+u*Z[i]);b=-math.pi+width*u;z=np.where(choose<.999,a,b);lg=-.5*((z-loc[i])/sd)**2-math.log(sd*math.sqrt(2*math.pi))-np.log(Z[i]);q=.999*np.exp(lg)+.001/width;weights=np.sum(-math.log(width)-np.log(q[:,missing]),axis=1)
  if len(active):z[:,active]=x[i,active]
  loglike=np.sum(-.5*((y[i]-z)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi)),axis=1);rows.append(logsumexp(loglike+weights)-math.log(M))
 return np.array(rows)

with out.open('w') as f:
 for M in (128,1024,4096,16384):
  t=time.monotonic();p=marginal([],M);l=marginal(list(range(50)),M);r=marginal(list(range(50,100)),M);actual_l=normal[:,:50].sum(1)+analytic[:,50:].sum(1);actual_r=normal[:,50:].sum(1)+analytic[:,:50].sum(1);syn=float(np.mean(normal.sum(1)+p-l-r));row={'draws':M,'evaluation_count':n,'n':dim,'channel':'independent uniform X and Gaussian Y=X+noise, sigma 0.086','true_syn_nats':0.,'estimated_syn_nats':syn,'mean_log_p_y_error_nats':float(np.mean(p-true_logpy)),'mean_log_p_y_given_left_error_nats':float(np.mean(l-actual_l)),'mean_log_p_y_given_right_error_nats':float(np.mean(r-actual_r)),'elapsed_seconds':time.monotonic()-t};f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
