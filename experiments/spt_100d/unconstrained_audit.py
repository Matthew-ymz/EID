"""Diagnose empirical-source-covariance affine Gaussian TM; no tree after invalid Syn."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,time
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky,cho_solve
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

cfg=b.mk_system(100,101,0);full=(1<<100)-1;out=b.OUT/'unconstrained_audit.jsonl'
with out.open('w') as f:
 for N in (4096,8192,16384,32768):
  t=time.monotonic();x,y=b.data(cfg,N,10101);ix=int(N*.6);iv=int(N*.8);X=b.phi(x[:ix]);V=b.phi(x[ix:iv]);xm=X.mean(0);ym=y[:ix].mean(0);ys=y[:ix].std(0);X-=xm;V-=xm;Y=(y[:ix]-ym)/ys;vy=(y[ix:iv]-ym)/ys;Cx=X.T@X/(len(X)-1);B=X.T@Y/(len(X)-1);Cy=Y.T@Y/(len(Y)-1);lx=cholesky(Cx,lower=True,check_finite=False);reg=cho_solve((lx,True),B,check_finite=False);best=None;cands=[]
  for lam in (.01,.1,1.):
   C=Cy+lam*np.eye(100);cond=C-B.T@reg;lc=cholesky(cond,lower=True,check_finite=False);res=vy-V@reg;nll=.5*(b.logdet(cond)+np.einsum('ij,ji->',res,cho_solve((lc,True),res.T,check_finite=False))/len(res))/100;entry={'ridge':lam,'validation_nll_per_target':float(nll)};cands.append(entry)
   if best is None or nll<best[0]:best=(nll,C,entry)
  ly=cholesky(best[1],lower=True,check_finite=False);S=Cx-B@cho_solve((ly,True),B.T,check_finite=False);cache={0:0.}
  def ei(mask):
   if mask not in cache:
    ii=[2*i+j for i in range(100) if mask>>i&1 for j in (0,1)];cache[mask]=.5*(b.logdet(Cx[np.ix_(ii,ii)])-b.logdet(S[np.ix_(ii,ii)]))
   return cache[mask]
  def gain(mask):return ei(full)-ei(mask)-ei(full^mask)
  masks=[sum(1<<v for v in g) for g in cfg['outer']];parent=[gain(m) for m in masks];singles=[gain(1<<i) for i in range(100)];vals=np.array(parent+singles);tol=1e-6;bad=vals < -tol
  row={'n':100,'seed':101,'bridge':0,'samples':N,'estimator':'unconstrained empirical-covariance Gaussian surrogate; intervention independence not enforced','chosen':best[2],'candidates':cands,'true_parent_cut_nats':parent,'singleton_cut_min_nats':float(min(singles)),'minimum_syn_nats':float(vals.min()),'negative_within_tolerance':int(((vals<0)&~bad).sum()),'negative_below_tolerance':int(bad.sum()),'checked_splits':len(vals),'tol_nats':tol,'status':'invalid_syn' if bad.any() else 'diagnostic_only','elapsed_seconds':time.monotonic()-t}
  f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
