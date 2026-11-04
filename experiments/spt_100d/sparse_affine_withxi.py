"""Exploratory sparsity-regularized affine TM, chosen only by held-out NLL."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,time
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky,cho_solve
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);out=b.OUT/'sparse_affine_withxi.jsonl'
with out.open('w') as f:
 for N in (4096,8192,32768):
  st=time.monotonic();cfg=b.mk_system(100,101,0);x,y=b.data(cfg,N,10101);i=int(N*.6);j=int(N*.8);X=b.phi(x[:i]);V=b.phi(x[i:j]);ym=y[:i].mean(0);ys=np.maximum(y[:i].std(0),1e-10);Y=(y[:i]-ym)/ys;vy=(y[i:j]-ym)/ys;B=X.T@Y/(i-1);C=Y.T@Y/(i-1);candidates=[];best=None
  for tau in (0.,.01,.03,.05,.1,.2):
   Bb=np.where(np.abs(B)>=tau,B,0.);C0=np.where(np.abs(C)>=tau,C,0.);np.fill_diagonal(C0,np.diag(C));eigmin=float(np.linalg.eigvalsh(C0)[0]);
   for ridge in (.01,.1,1.):
    jitter=max(0.,1e-8-(eigmin+ridge));Cy=C0+(ridge+jitter)*np.eye(100);L=cholesky(Cy,lower=True,check_finite=False);K=Bb@cho_solve((L,True),Bb.T,check_finite=False);ev=float(np.linalg.eigvalsh(K)[-1]);factor=min(1.,.99/math.sqrt(ev)) if ev>0 else 1.;Bc=Bb*factor;cond=Cy-Bc.T@Bc;LC=cholesky(cond,lower=True,check_finite=False);err=vy-V@Bc;nll=.5*(b.logdet(cond)+np.einsum('ij,ji->',err,cho_solve((LC,True),err.T,check_finite=False))/len(err))/100;rec={'threshold':tau,'ridge':ridge,'target_diagonal_jitter':jitter,'cross_covariance_scale':factor,'validation_nll_per_target':float(nll),'nonzero_cross_covariance_count':int(np.count_nonzero(Bb))};candidates.append(rec)
    if best is None or nll<best[0]:best=(nll,np.eye(200)-factor*factor*K,rec)
  o=b.Oracle(best[1],time.monotonic()+60);full=(1<<100)-1;parent=[]
  for g in cfg['outer']:
   m=sum(1<<v for v in g);parent.append(o.gain((m,full^m)))
  single=[o.gain((1<<i,full^(1<<i))) for i in range(100)];xi=o.xi(tuple(range(100)));off=best[1].copy()
  for k in range(100):off[2*k:2*k+2,2*k:2*k+2]=0.
  row={'seed':101,'bridge':0,'samples':N,'method':'elementwise sparse covariance, no known groups, threshold/ridge selected by heldout conditional Gaussian NLL','chosen':best[2],'candidates':candidates,'parent_cut_nats':parent,'true_parent_cut_nats':0.,'smallest_singleton_cut_nats':min(single),'root_xi_nats':xi,'max_between_source_conditional_covariance':float(np.max(np.abs(off))),'minimum_syn_nats':o.min_syn,'tol_nats':b.TOL,'negative_within_tol':o.near,'elapsed_seconds':time.monotonic()-st};f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
