"""100-dimensional mixed-order Kuramoto pressure test. See approved protocol."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,importlib.util,json,math,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.linalg import cholesky,cho_solve
from eid.spt import build_spt, SPTConfig, spectral_candidate_selector,pairwise_syn_affinity,flatten_nodes
spec=importlib.util.spec_from_file_location('pilot',ROOT/'experiments/spt_agglomerative/pilot.py');pilot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pilot;spec.loader.exec_module(pilot)
OUT=ROOT/'docs/log/spt_100d_stress/results';TOL=1e-6;clock_end=0.

def save(row):
 with (OUT/'results.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ('tree','truth','permutation','diagnostics','cache')}),flush=True)

def mk_system(n,seed,bridge):
 if n==6:parts=[3,3];group_sizes=[6]
 elif n==20:parts=[3,4,6,7];group_sizes=[20]
 elif n in (50,100):parts=[3,4,8,10];group_sizes=[25]*(n//25)
 else:raise ValueError(n)
 rng=np.random.default_rng(seed);perm=rng.permutation(n);edges=[];truth=[];outer=[];offset=0
 for size in group_sizes:
  current=perm[offset:offset+size];outer.append(tuple(int(x) for x in current));a=0
  for k in parts:
   ids=tuple(int(x) for x in current[a:a+k]);a+=k;edges.append((ids,.57));truth.append(ids)
  if len(group_sizes)>1:truth.append(tuple(int(x) for x in current))
  edges.append((tuple(int(x) for x in current),1.13));offset+=size
 if bridge:
  for a,b in zip(outer[:-1],outer[1:]):edges.append(((a[-1],b[0]),.02))
 omega=np.linspace(-.4,.4,n);omega=omega[rng.permutation(n)]
 return {'n':n,'perm':perm.tolist(),'edges':edges,'truth':truth,'outer':outer,'omega':omega,'seed':seed,'bridge':bridge,'sigma':.08,'tau':.2,'dt':.01}

def drift(x,cfg):
 y=np.broadcast_to(cfg['omega'],x.shape).copy()
 for ids,w in cfg['edges']:
  idx=np.asarray(ids);z=x[:,idx];s=z.sum(axis=1);y[:,idx]+=w*np.sin(s[:,None]-len(ids)*z)
 return y

def data(cfg,samples,seed):
 rng=np.random.default_rng(seed);x=rng.uniform(-np.pi,np.pi,(samples,cfg['n']));y=x.copy();dt=cfg['dt']
 for _ in range(round(cfg['tau']/dt)):
  k1=drift(y,cfg);k2=drift(y+.5*dt*k1,cfg);k3=drift(y+.5*dt*k2,cfg);k4=drift(y+dt*k3,cfg)
  y+=dt*(k1+2*k2+2*k3+k4)/6+cfg['sigma']*np.sqrt(dt)*rng.normal(size=y.shape)
 return x,y

def phi(x):
 a=np.empty((len(x),2*x.shape[1]));a[:,0::2]=np.sqrt(2)*np.sin(x);a[:,1::2]=np.sqrt(2)*np.cos(x);return a

def logdet(a):
 L=cholesky(a,lower=True,check_finite=False);return float(2*np.log(np.diag(L)).sum())

def fit_constrained(x,y,vx,vy,fixed_lambda=None):
 # Known joint independent-input Gaussian surrogate. Candidate lambda is
 # selected by validation conditional likelihood, never by module labels.
 x=phi(x);vx=phi(vx);ym=y.mean(0);ys=y.std(0);ys=np.where(ys>1e-10,ys,1.); y=(y-ym)/ys;vy=(vy-ym)/ys
 B=x.T@y/(len(x)-1);C=y.T@y/(len(x)-1);d=x.shape[1];candidates=[];best=None
 for lam in ((.01,.1,1.) if fixed_lambda is None else (fixed_lambda,)):
  Cy=C+lam*np.eye(C.shape[0]);L=cholesky(Cy,lower=True,check_finite=False)
  K=B@cho_solve((L,True),B.T,check_finite=False);largest=float(np.linalg.eigvalsh(K)[-1]);factor=min(1.,.99/math.sqrt(largest)) if largest>0 else 1.
  Bb=B*factor;cond=Cy-Bb.T@Bb;LC=cholesky(cond,lower=True,check_finite=False);res=vy-vx@Bb
  nll=float(.5*(logdet(cond)+np.einsum('ij,ji->',res,cho_solve((LC,True),res.T,check_finite=False))/len(res))/y.shape[1])
  record={'ridge':lam,'validation_nll_per_target':nll,'cross_covariance_scale':factor,'max_generalized_correlation_before':math.sqrt(max(largest,0))};candidates.append(record)
  if best is None or nll<best[0]:best=(nll,np.eye(d)-factor*factor*K,record)
 return best[1],{'selection':'lowest validation Gaussian conditional NLL' if fixed_lambda is None else 'fixed from independent 100-dimensional development seed','candidates':candidates,'chosen':best[2],'source_covariance':'known block-independent identity for sqrt(2) sin/cos moments','source_model':'Gaussian surrogate with rank-2 phase features; not true circle distribution','target':'complete future state'}

class Oracle:
 def __init__(self,S,deadline):self.S=S;self.n=S.shape[0]//2;self.cache={0:0.};self.ei_queries=0;self.gain_queries=0;self.near=0;self.min_syn=float('inf');self.deadline=deadline;self.time_ei=0.
 def ei(self,m):
  if m not in self.cache:
   if time.monotonic()>self.deadline:raise TimeoutError('per-condition/global deadline reached')
   a=time.perf_counter();ix=[2*i+j for i in range(self.n) if m>>i&1 for j in (0,1)];self.cache[m]=-.5*logdet(self.S[np.ix_(ix,ix)]);self.ei_queries+=1;self.time_ei+=time.perf_counter()-a
  return self.cache[m]
 def gain(self,parts):
  self.gain_queries+=1;v=self.ei(sum(parts))-sum(self.ei(m) for m in parts);self.min_syn=min(self.min_syn,v)
  if v < -TOL:raise ArithmeticError(f'Syn violation minimum={v:.12g}, threshold={-TOL}, affected_count=1')
  if v<0:self.near+=1
  return v
 def xi(self,sources):
  s=tuple(sources);return self.ei(sum(1<<i for i in s))-sum(self.ei(1<<i) for i in s)

def exhaustive(o,p):
 anchor=p&-p;res=[];s=p
 while s:
  s=(s-1)&p
  if s and s!=p and s&anchor:res.append((o.gain((s,p^s)),s,p^s))
 return min(res)

def qtree(o):
 decisions=[];t=time.perf_counter()
 def visit(p):
  if p.bit_count()==1:return {'mask':p,'children':[],'syn':0.}
  l,r,val,_=pilot.queyranne_split(o,p);decisions.append((p,l,r,val));return {'mask':p,'children':[visit(l),visit(r)],'syn':val}
 tree=visit((1<<o.n)-1);singles=sum(o.ei(1<<i) for i in range(o.n));xi=o.ei((1<<o.n)-1)-singles
 return tree,decisions,{'seconds':time.perf_counter()-t,'ei_seconds':o.time_ei,'unique_ei_queries':o.ei_queries,'gain_calls':o.gain_queries,'tol_nats':TOL,'min_candidate_syn_nats':o.min_syn,'near_zero_negative_count':o.near,'closure_error':float(sum(z[3] for z in decisions)-xi),'root_xi_nats':xi}

def modules(tree,cfg):
 def walk(t):return [t['mask']]+[v for child in t['children'] for v in walk(child)]
 nodes=set(walk(tree));truth={sum(1<<i for i in g) for g in cfg['truth']};outer={sum(1<<i for i in g) for g in cfg['outer']};primitive={t for t in truth if not any(q!=t and q&t==q for q in truth)}
 def unscored(s):
  if any(s!=p and s&p==s for p in primitive):return True
  for blocks in (outer,primitive):
   contained=[p for p in blocks if s&p==p]
   if len(contained)>=2 and sum(contained)==s and s not in truth:return True
  return False
 pred={s for s in nodes if 1<s.bit_count()<cfg['n'] and not unscored(s)}; true={s for s in truth if s.bit_count()<cfg['n']}
 return {'truth_count':len(true),'matched_count':len(pred&true),'predicted_count':len(pred),'recall':len(pred&true)/len(true) if true else None,'precision':len(pred&true)/len(pred) if pred else 0.,'f1':2*len(pred&true)/(len(pred)+len(true)) if pred or true else 1.,'by_order':{str(k):{'found':sum(s in pred for s in true if s.bit_count()==k),'total':sum(s.bit_count()==k for s in true)} for k in sorted({s.bit_count() for s in true})},'missed':[[i for i in range(cfg['n']) if s>>i&1] for s in sorted(true-pred)]}

def serial_tree(t):return {'sources':[i for i in range(t['mask'].bit_length()) if t['mask']>>i&1],'syn_nats':t['syn'],'children':[serial_tree(c) for c in t['children']]}

def eval_case(n,seed,bridge,samples,case_limit,fixed_lambda=None):
 cfg=mk_system(n,seed,bridge);start=time.monotonic(); x,y=data(cfg,samples,seed+10000);data_sec=time.monotonic()-start;ix=int(samples*.6);iv=int(samples*.8)
 fit=time.monotonic();S,meta=fit_constrained(x[:ix],y[:ix],x[ix:iv],y[ix:iv],fixed_lambda);fit_sec=time.monotonic()-fit
 deadline=min(clock_end,time.monotonic()+case_limit)
 o=Oracle(S,deadline);t,dec,search=qtree(o);qscore=modules(t,cfg)
 checks=[]
 for p,l,r,v in dec:
  if p.bit_count()<=8:
   exact,_,_=exhaustive(o,p);checks.append(float(v-exact))
 # Compare current spectral candidate method using same oracle and fixed model.
 spectime=time.monotonic();oo=Oracle(S,deadline);W,_=pairwise_syn_affinity(oo,n,tolerance=TOL)
 class Adapter:
  def xi(self,s):return oo.xi(s)
 cfgs=SPTConfig(syn_tolerance=TOL,eps=1e-10,complete_to_singletons=True)
 tree2=build_spt(tuple(range(n)),Adapter(),config=cfgs,candidate_selector=spectral_candidate_selector(W,exact_max_size=8))
 def convert(z):return {'mask':sum(1<<int(i) for i in z.sources),'syn':z.syn_value,'children':[convert(c) for c in z.children]}
 spectral=modules(convert(tree2.root),cfg)
 # The held-out data were not used by fit or hyperparameter selection.
 return {'event':'case','n':n,'seed':seed,'bridge':bridge,'samples':samples,'train_n':ix,'validation_n':iv-ix,'test_n':samples-iv,'data_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),'truth':cfg['truth'],'permutation':cfg['perm'],'estimator':meta,'dataset_seconds':data_sec,'fit_seconds':fit_sec,'queyranne':{'scores':qscore,'search':search,'tree':serial_tree(t),'small_nodes_compared':len(checks),'small_nodes_max_optimality_gap':max(checks) if checks else None},'spectral':{'scores':spectral,'seconds_including_pairwise_affinity':time.monotonic()-spectime,'unique_ei_queries':oo.ei_queries,'ei_seconds':oo.time_ei,'closure_error':tree2.closure_error},'elapsed_seconds':time.monotonic()-start}

def main():
 global clock_end
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--budget',type=int,default=5400);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);p=OUT/('smoke.jsonl' if a.smoke else 'results.jsonl')
 if p.exists():raise FileExistsError(p)
 origsave=globals()['save']
 def write(row):
  with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print(json.dumps({k:v for k,v in row.items() if k not in ('tree','truth','permutation','queyranne','spectral')}),flush=True)
 globals()['save']=write
 started=time.monotonic();clock_end=started+a.budget; signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('total authorized budget reached')));signal.alarm(a.budget)
 save({'event':'start','smoke':a.smoke,'budget_seconds':a.budget,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tol_nats':TOL})
 tasks=[(6,101,0,512)] if a.smoke else [(6,101,0,4096),(20,101,0,4096),(50,101,0,4096),(100,101,0,4096)]+[(100,seed,bridge,8192) for bridge in (0,1) for seed in (201,202,203)]
 selected_lambda=None
 for n,seed,bridge,N in tasks:
  if time.monotonic()>clock_end:break
  print(f'case n={n} seed={seed} bridge={bridge} samples={N}',flush=True)
  try:
   fixed=selected_lambda if n==100 and seed!=101 else None
   if n==100 and seed!=101 and fixed is None:raise RuntimeError('100-dimensional development calibration did not finish; cannot select configuration without independent development result')
   row=eval_case(n,seed,bridge,N,case_limit=720 if not a.smoke else 100,fixed_lambda=fixed);save(row)
   if n==100 and seed==101:selected_lambda=row['estimator']['chosen']['ridge']
  except Exception as e:
   save({'event':'failure','n':n,'seed':seed,'bridge':bridge,'samples':N,'error':repr(e),'traceback':traceback.format_exc(),'elapsed':time.monotonic()-started})
   if time.monotonic()>clock_end:break
 signal.alarm(0);save({'event':'end','elapsed_seconds':time.monotonic()-started,'budget_hit':time.monotonic()>=clock_end})
if __name__=='__main__':main()
