"""Paired held-out likelihood differences for preselected polynomial sketches."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,time
from pathlib import Path
import numpy as np
from scipy.fft import rfft,irfft
from scipy.linalg import eigh
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);cfg=b.mk_system(100,101,0);x,y=b.data(cfg,8192,10101);n=4915;iv=6553;dx=y-x;mu=dx[:n].mean(0);scale=np.maximum(dx[:n].std(0),1e-6);Y=(dx-mu)/scale;Z=np.concatenate((x,np.sin(x),np.cos(x)),axis=1);Z=(Z-Z[:n].mean(0))/np.maximum(Z[:n].std(0),1e-6);aug=np.concatenate((Z/np.sqrt(300),np.ones((len(Z),1))),axis=1)
def sketch(degree):
 rng=np.random.default_rng(101);product=None
 for _ in range(degree):
  hashes=rng.integers(0,1024,size=aug.shape[1]);signs=rng.choice([-1.,1.],size=aug.shape[1]);tmp=np.zeros((len(Z),1024))
  for j in range(aug.shape[1]):tmp[:,hashes[j]]+=signs[j]*aug[:,j]
  f=rfft(tmp,axis=1);product=f if product is None else product*f
 return irfft(product,n=1024,axis=1)
def individual(pred):
 residual=Y-pred;var=np.maximum(np.mean(residual[n:iv]**2,axis=0),1e-10);return np.sum(.5*(residual[iv:]**2/var)+.5*np.log(2*np.pi*var)+np.log(scale),axis=1)
baseline=individual(np.zeros_like(Y));result=[]
for degree in (2,3):
 F=sketch(degree);F-=F[:n].mean(0);D,V=eigh(F[:n].T@F[:n],check_finite=False,driver='evd');D=np.maximum(D,0);T=V.T@(F[:n].T@Y[:n]);pred=(F@V)@(T/(D[:,None]+10000));diff=individual(pred)-baseline;sem=float(np.std(diff,ddof=1)/np.sqrt(len(diff)));result.append({'degree':degree,'test_n':len(diff),'alpha':10000,'mean_test_nll_difference_nats':float(diff.mean()),'sem_nats':sem,'approx_95pct_interval_nats':[float(diff.mean()-1.96*sem),float(diff.mean()+1.96*sem)],'note':'paired test-sample uncertainty conditional on the fitted models; not a confidence interval across data or sketch seeds'})
(b.OUT/'polynomial_paired_uncertainty.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
