"""Generic quadratic/cubic context for a 100D diagonal conditional transport.

TensorSketch compresses all polynomial interactions without using
planted labels; results assess conditional prediction, not subset EI or SPT.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.fft import rfft, irfft
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
start=time.monotonic();cfg=b.mk_system(100,101,0);x,y=b.data(cfg,8192,10101);n=4915;iv=6553;dx=y-x;mu=dx[:n].mean(0);scale=np.maximum(dx[:n].std(0),1e-6);Y=(dx-mu)/scale
Z=np.concatenate((x,np.sin(x),np.cos(x)),axis=1);zm=Z[:n].mean(0);zs=np.maximum(Z[:n].std(0),1e-6);Z=(Z-zm)/zs

def metrics(pred):
 residual=Y-pred;var=np.maximum(np.mean(residual[n:iv]**2,axis=0),1e-10);val=float(.5*np.sum(np.log(var)));test=float(np.mean(np.sum(.5*(residual[iv:]**2/var)+.5*np.log(2*np.pi*var)+np.log(scale),axis=1)));mse=float(np.mean((residual[iv:]*scale)**2));return val,test,mse
base=metrics(np.zeros_like(Y));out={'seed':101,'n':100,'samples':8192,'train_n':n,'validation_n':iv-n,'test_n':8192-iv,'target':'full 100-dimensional future state','conditional_model':'Gaussian diagonal transport, polynomial location model','features':'random sketch of complete degree <= d polynomial kernel on standardized phase, sin, cos features','uses_planted_groups':False,'sketch_components':1024,'alphas':[.01,.1,1.,10.,100.,1000.,10000.,100000.,1000000.],'baseline':{'validation_nll_up_to_constant':base[0],'test_nll_nats':base[1],'test_mse':base[2]},'models':[]}
def tensor_sketch(Z,degree,width=1024):
 rng=np.random.default_rng(101);aug=np.concatenate((Z/np.sqrt(300),np.ones((len(Z),1))),axis=1);product=None
 for k in range(degree):
  hashes=rng.integers(0,width,size=aug.shape[1]);signs=rng.choice([-1.,1.],size=aug.shape[1]);tmp=np.zeros((len(Z),width))
  for j in range(aug.shape[1]):tmp[:,hashes[j]]+=signs[j]*aug[:,j]
  f=rfft(tmp,axis=1);product=f if product is None else product*f
 return irfft(product,n=width,axis=1)
for degree in (1,2,3):
 t=time.monotonic();features=Z if degree==1 else tensor_sketch(Z,degree)
 xm=features[:n].mean(0);features=features-xm;G=features[:n].T@features[:n];D,V=eigh(G,check_finite=False,driver='evd');D=np.maximum(D,0);T=V.T@(features[:n].T@Y[:n]);ZV=features@V;best=None;entries=[]
 for a in out['alphas']:
  pred=ZV@(T/(D[:,None]+a));val,test,mse=metrics(pred);entries.append({'alpha':a,'validation_nll_up_to_constant':val})
  if best is None or val<best[0]:best=(val,a,test,mse)
 out['models'].append({'degree':degree,'candidate_alphas':entries,'selected_alpha':best[1],'validation_nll_up_to_constant':best[0],'test_nll_nats':best[2],'test_mse':best[3],'elapsed_seconds':time.monotonic()-t});print(json.dumps(out['models'][-1]),flush=True)
out['elapsed_seconds']=time.monotonic()-start;(b.OUT/'polynomial_context.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'baseline':out['baseline'],'elapsed_seconds':out['elapsed_seconds']}),flush=True)
