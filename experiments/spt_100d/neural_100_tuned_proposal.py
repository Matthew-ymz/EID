"""100D diagonal conditional transport: held-out fit and marginal-integration feasibility.

Target ordering is full 100D, transform is monotone affine in each target but the
conditional location is an MLP of all source phases. No structural labels are used.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr,ndtri,logsumexp
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def main():
 start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(101);rng=np.random.default_rng(808);cfg=b.mk_system(100,101,0);x,y=b.data(cfg,8192,10101);n=4915;m=6553;delta=y-x;mu=delta[:n].mean(0);scale=delta[:n].std(0);scale=np.maximum(scale,1e-6)
 def feat(z):return np.concatenate((z,np.sin(z),np.cos(z)),axis=1).astype('float32')
 X=torch.tensor(feat(x));Y=torch.tensor(((delta-mu)/scale).astype('float32'));net=torch.nn.Sequential(torch.nn.Linear(300,128),torch.nn.Tanh(),torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,100));opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001);gen=torch.Generator().manual_seed(101);best=None;patience=0
 for epoch in range(80):
  net.train()
  for batch in torch.randperm(n,generator=gen).split(256):
   opt.zero_grad();loss=((net(X[batch])-Y[batch])**2).mean();loss.backward();opt.step()
  net.eval()
  with torch.no_grad():val=float(((net(X[n:m])-Y[n:m])**2).mean())
  if best is None or val<best[0]-1e-5:best=(val,{k:v.detach().clone() for k,v in net.state_dict().items()},epoch);patience=0
  else:patience+=1
  if patience>=8:break
 net.load_state_dict(best[1]);net.eval()
 with torch.no_grad():pred=net(X).numpy()*scale+mu
 sigma=np.maximum(np.sqrt(np.mean((delta[n:m]-pred[n:m])**2,axis=0)),1e-5);base_sigma=np.maximum(np.sqrt(np.mean((delta[n:m]-mu)**2,axis=0)),1e-5)
 def nll(res,sd):return float(np.mean(np.sum(.5*(res/sd)**2+np.log(sd*math.sqrt(2*math.pi)),axis=1)))
 p={'n':100,'samples':8192,'train_n':n,'validation_n':m-n,'test_n':8192-m,'architecture':'diagonal Gaussian conditional transport with nonlinear source-conditioned location','target':'all 100 future coordinates','source_inputs':'raw phases and sin/cos','uses_planted_groups':False,'best_epoch':best[2],'epochs_run':epoch+1,'test_nll_nats_per_100d_target':nll(delta[m:]-pred[m:],sigma),'baseline_test_nll_nats_per_100d_target':nll(delta[m:]-mu,base_sigma),'test_mse':float(np.mean((delta[m:]-pred[m:])**2)),'baseline_test_mse':float(np.mean((delta[m:]-mu)**2)),'fit_seconds':time.monotonic()-start,'syn_tolerance_nats':.06931471805599453,'integration_proposal':'per-coordinate 99.9% truncated Gaussian centered at Y minus fitted mean increment, sd = 1.25 times held-out calibrated conditional sd; 0.1% uniform', 'integration':[]}
 def mean_of(z):
  out=[]
  with torch.no_grad():
   for start2 in range(0,len(z),4096):out.append(net(torch.tensor(feat(z[start2:start2+4096]))).numpy()*scale+mu)
  return np.concatenate(out)
 def logp(y0,z):
  residual=(y0-z-mean_of(z))/sigma
  return -.5*np.sum(residual**2+np.log(2*math.pi*sigma**2),axis=1)
 N=32;ex=rng.uniform(-math.pi,math.pi,(N,100));ey=ex+mean_of(ex)+rng.normal(size=(N,100))*sigma;full_log=logp(ey,ex);width=2*math.pi;loc=np.clip(ey-mu,-math.pi,math.pi);sd=1.25*sigma;L=ndtr((-math.pi-loc)/sd);U=ndtr((math.pi-loc)/sd);Z=U-L
 def marginal(active,M):
  out=[];r=np.random.default_rng(6060)
  miss=np.array([i for i in range(100) if i not in active]);active=np.array(active,dtype=int)
  for i in range(N):
   u=r.random((M,100));u2=r.random((M,100));gauss=loc[i]+sd*ndtri(L[i]+u*Z[i]);uniform=-math.pi+width*u;z=np.where(u2<.999,gauss,uniform);lg=-.5*((z-loc[i])/sd)**2-np.log(sd*math.sqrt(2*math.pi))-np.log(Z[i]);q=.999*np.exp(lg)+.001/width;weights=np.sum(-math.log(width)-np.log(q[:,miss]),axis=1)
   if len(active):z[:,active]=ex[i,active]
   values=logp(np.repeat(ey[i:i+1],M,axis=0),z)+weights;out.append(logsumexp(values)-math.log(M))
  return np.array(out)
 for M in (128,1024,4096,16384):
  t=time.monotonic();allpy=marginal([],M);left=marginal(list(range(50)),M);right=marginal(list(range(50,100)),M);allmi=float(np.mean(full_log-allpy));lmi=float(np.mean(left-allpy));rmi=float(np.mean(right-allpy));syn=allmi-lmi-rmi
  p['integration'].append({'draws':M,'evaluation_count':N,'all_nats':allmi,'left_nats':lmi,'right_nats':rmi,'syn_nats':syn,'passes_tolerance':syn>=-p['syn_tolerance_nats'],'seconds':time.monotonic()-t})
  print(json.dumps(p['integration'][-1]),flush=True)
 p['elapsed_seconds']=time.monotonic()-start;(b.OUT/'neural_100_tuned_proposal.json').write_text(json.dumps(p,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in p.items() if k!='integration'}),flush=True)
if __name__=='__main__':main()
