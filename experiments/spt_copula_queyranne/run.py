"""Original six-source cases: discrete exact and copula/TM SPT. Values nats."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse, hashlib, importlib.util, itertools, json, math, signal, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT.parent/'EISyn'
sys.path.insert(0,str(SRC));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri
from scripts import run_mixed_order_kuramoto_kout_main as mixed
from scripts import validate_mixed_order_kuramoto_hierarchy as orig
from scripts import kuramoto_six_topology_core as core
from scripts.classic_network_dynamics_benchmark import fit_mlp
from scripts.search_six_node_mixed_dynamics import spin_kernel, exact_ei
spec=importlib.util.spec_from_file_location('spt_pilot',ROOT/'experiments/spt_agglomerative/pilot.py');pilot=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pilot;spec.loader.exec_module(pilot)
OUT=Path(__file__).parent/'results'; LOG2=math.log(2); FULL=63

def save(row):
 with (OUT/'results.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in row.items() if k not in ('table','trees','slack','sample_hash')}),flush=True)

def names(mask):return [i+1 for i in range(6) if (mask>>i)&1]

def table_bits(t):return {sum(1<<(int(i)-1) for i in k):float(v)*LOG2 for k,v in t.items()}

def gcmi_table(x,y,ridge=1e-6):
 z=np.column_stack((x,y));z=ndtri(rankdata(z,axis=0,method='average')/(len(z)+1));cov=np.cov(z,rowvar=False);cov+=ridge*np.eye(cov.shape[0]); target=list(range(6,cov.shape[0]));
 def ld(ix):
  s,l=np.linalg.slogdet(cov[np.ix_(ix,ix)]);
  if s<=0:raise ValueError('non-positive copula covariance')
  return l
 hy=ld(target);t={0:0.}
 def query(mask):
  xx=[i for i in range(6) if mask>>i&1];t[mask]=float(.5*(ld(xx)+hy-ld(xx+target)))
  return t[mask]
 for mask in range(1,64):query(mask)
 return t,query

class TableOracle:
 def __init__(self,t,query=None):self.table=t;self.query=query;self.cache={0:0.};self.calls=0
 def ei(self,m):
  if m not in self.cache:
   value=self.table[m] if self.query is None else self.query(m)
   if abs(value-self.table[m])>1e-9:raise ValueError('oracle mismatch')
   self.cache[m]=value
  return self.cache[m]
 def gain(self,parts):
  self.calls+=1; return self.ei(sum(parts))-sum(self.ei(i) for i in parts)

def split_exact(o,parent):
 anchor=parent & -parent; r=parent; options=[]
 while r:
  r=(r-1)&parent
  if r and r!=parent and r&anchor:
   other=parent^r;options.append((o.gain((r,other)),r,other))
 return min(options)

def tree_solve(table,algorithm,query=None):
 o=TableOracle(table,query);started=time.perf_counter(); decisions=[]
 def build(m):
  if m.bit_count()==1:return {'mask':m,'children':[],'syn':0.}
  if algorithm=='exact':val,l,r=split_exact(o,m)
  else:
   l,r,val,_=pilot.queyranne_split(o,m)
  decisions.append({'parent':m,'left':l,'right':r,'syn':val})
  return {'mask':m,'children':[build(l),build(r)],'syn':val}
 tree=build(FULL);elapsed=time.perf_counter()-started
 return tree,decisions,{'seconds':elapsed,'ei_unique_queries':len(o.cache)-1,'gain_calls':o.calls}

def clades(t):return {t['mask']}|set().union(*(clades(c) for c in t['children']))

def f1(t,truth):
 C=clades(t);primitive={p for p in truth if not any(q!=p and (q&p)==q for q in truth)}
 pred={x for x in C if 1<x.bit_count()<6 and not any(x!=p and x&p==x for p in primitive)}
 return 2*len(pred&truth)/(len(pred)+len(truth)) if pred or truth else 1.

def diagnostics(table,tol):
 negatives=[];slacks=[]
 for p in range(1,64):
  bits=[1<<i for i in range(6) if p>>i&1]
  for r in range(1,1<<len(bits)):
   l=sum(bits[i] for i in range(len(bits)) if r>>i&1)
   if l==p or not(l&(p&-p)):continue
   negatives.append(table[p]-table[l]-table[p^l])
  def g(l):return 0. if l in (0,p) else table[p]-table[l]-table[p^l]
  for r in range(1<<len(bits)):
   s=sum(bits[i] for i in range(len(bits)) if r>>i&1);left=[i for i in bits if not s&i]
   for a,b in itertools.combinations(left,2):slacks.append(g(s|a)+g(s|b)-g(s)-g(s|a|b))
 v=np.asarray(negatives);w=np.asarray(slacks)
 return {'tol_nats':tol,'minimum_syn_nats':float(v.min()),'negative_within_tol':int(np.sum((v<0)&(v>=-tol))),'negative_below_tol':int(np.sum(v < -tol)),'syn_count':len(v),'submod_min_slack_nats':float(w.min()),'submod_below_minus_1e-9':int(np.sum(w < -1e-9)),'submod_below_minus_tol':int(np.sum(w < -tol)),'submod_total':len(w)}

def evaluate(case,seed,method,table,tol,truth,metadata,query=None):
 table={int(k):float(v) for k,v in table.items()};table[0]=0.; audit=diagnostics(table,tol)
 row={'event':'evaluation','case':case,'seed':seed,'estimator':method,'audit':audit,'metadata':metadata,'table':table}
 if audit['negative_below_tol']:
  row['status']='invalid_syn';row['failure']=f"min={audit['minimum_syn_nats']}; threshold={-tol}; count={audit['negative_below_tol']}";save(row);return
 trees={};dec={}
 for a in ('exact','queyranne'):
  t,d,s=tree_solve(table,a,query); trees[a]=t;dec[a]=d;row[a]={'f1':f1(t,truth),'search':s,'clades':[names(m) for m in sorted(clades(t)) if 1<m.bit_count()<6], 'closure':sum(v['syn'] for v in d)-(table[FULL]-sum(table[1<<j] for j in range(6)))}
 for a in ('queyranne',):
  gaps=[]
  for d in dec[a]:
   oracle=TableOracle(table);opt,_,_=split_exact(oracle,d['parent']);gaps.append(d['syn']-opt)
  row[a]['node_max_suboptimality_nats']=max(gaps); row[a]['different_from_exhaustive_clades']=clades(trees[a])!=clades(trees['exact'])
 row['status']='valid';row['trees']=trees;save(row)

def discrete():
 t0=time.perf_counter();k=spin_kernel(beta=1.,triad=1.,cross=.04,family='glauber_ctmc',tau=2.);t=table_bits(exact_ei(k));evaluate('II','oracle','exact_discrete',t,1e-10*LOG2,{7,56},{'type':'oracle_matrix_exponential','generation_seconds':time.perf_counter()-t0})
 for seed in (1,2,3):
  p=SRC/f'results/six_node_data_driven_tpm/confirmation/n48000_seed{seed}.json';d=json.loads(p.read_text());q=d['methods']['mlp']['spt']['ei_bits'];t={sum(1<<(int(i)-1) for i in k):v*LOG2 for k,v in q.items()};evaluate('II',seed,'exact_discrete_fitted_MLP',t,float(d['syn_tolerance_bits'])*LOG2,{7,56},{'source':str(p),'channel':'cached 48000-pair learned MLP','n':48000})

def nested():
 base=SRC/'results/spt_third_case_weak_coupling';cfg=json.loads((base/'configuration.json').read_text());model=core.MLPChannel(json.loads((base/'model.json').read_text()))
 for seed,scramble in enumerate((880,1880,2880)):
  p=base/f'ei_N4096_P512_R{scramble}.json';d=json.loads(p.read_text());t={int(k):v for k,v in d['ei_nats'].items()};evaluate('III',seed,'original_shared_channel_density',t,.01,{7,15,48},{'cached':str(p),'fitted_channel_seed':0,'integration_scramble':scramble,'estimator_seconds_cached':d.get('seconds')})
  rng=np.random.default_rng(32000+seed);x=rng.uniform(-np.pi,np.pi,(4096,6));y=model.sample(x,rng.normal(size=x.shape));tt=time.perf_counter();t,query=gcmi_table(x,y);sec=time.perf_counter()-tt
  evaluate('III',seed,'gaussian_copula',t,.01,{7,15,48},{'n':4096,'seconds_for_all_63_ei':sec,'channel':'same cached fitted MLP, new independent intervention draws','target':'unwrapped future 6 phases','source_coordinates':'angles, no phase lift','fitted_channel_seed':0,'sample_hash':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),'search_timing':'cold subset queries; excludes rank/covariance construction and full validity audit'},query)

def mixed_case(seed,smoke=False):
 mixed._configure_coherent_tm();count=480 if smoke else 4800; readout=1024 if smoke else 4000; mt=10 if smoke else 800;ft=3 if smoke else 100; ev=64 if smoke else 512; marginal=32 if smoke else 256
 t0=time.perf_counter();x0,y0=orig.generated_transition_data(sample_count=count,seed=1000+seed,pairwise_coupling=mixed.WITHIN_COUPLING,triadic_coupling=mixed.WITHIN_COUPLING/math.sqrt(2),cross_coupling=.04,process_noise=.08,tau=.2,dt=.01);inc=np.angle(np.exp(1j*(y0-x0)));features=mixed._mlp_phase_features(x0);model=fit_mlp(features,inc,seed=2000+seed,epochs=mt);split=int(.8*len(x0));res=inc[split:]-model.predict(features[split:]);cov=np.cov(res,rowvar=False)+1e-6*np.eye(6); comps,effects=mixed.learned_dependency_components(model,seed=2500+seed);fitsecs=time.perf_counter()-t0
 x=np.random.default_rng(seed+3000).uniform(-np.pi,np.pi,(readout,6));y=model.predict(mixed._mlp_phase_features(x))+np.random.default_rng(seed+4000).multivariate_normal(np.zeros(6),cov,size=readout)
 t0=time.perf_counter();table,query=gcmi_table(x,y);calc=time.perf_counter()-t0
 evaluate('I',seed,'gaussian_copula',table,.1*LOG2,{7,56},{'n':readout,'seconds_for_all_63_ei':calc,'fit_seconds':fitsecs,'target':'wrapped increment fitted model + Gaussian residual','source_coordinates':'angles','components':comps,'sample_hash':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),'smoke':smoke,'search_timing':'cold subset queries; excludes rank/covariance construction and full validity audit'},query)
 tables=[];t0=time.perf_counter()
 for cidx,component in enumerate(comps):
  cn=tuple(mixed.NAMES[i] for i in component);orig.phase_transport_context=lambda values,names=cn:mixed._numpy_context_for_names(values,names)
  flow,details=orig.fit_joint_conditional_transport_map(x[:,component],y[:,component],train_fraction=.8,epochs=ft,seed=seed+5000+cidx,target_scaling='per_dimension')
  try:local,diagn=mixed.coherent_ei_table_without_resampling(flow,source_names=cn,seed=seed+6000+cidx,evaluation_count=ev,marginal_samples=marginal,context_active=bool(details['context_active']))
  except mixed.JointTMNonnegativityError as exc:
   save({'event':'failure','case':'I','seed':seed,'estimator':'original_TM','error':str(exc),'diagnostics':exc.diagnostics,'raw_table':{str(k):v for k,v in exc.table.items()},'smoke':smoke});return
  tables.append((cn,local))
 raw=mixed.combine_independent_component_ei_tables(tables);table=table_bits({tuple(int(s[5:]) for s in k):v for k,v in raw.items()})
 evaluate('I',seed,'original_component_neural_TM',table,.1*LOG2,{7,56},{'n':readout,'seconds_estimator':time.perf_counter()-t0,'fit_seconds':fitsecs,'components':comps,'integration':{'outer':ev,'marginal':marginal},'smoke':smoke})

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--budget',type=int,default=3600);a=p.parse_args();OUT.mkdir(exist_ok=True)
 if (OUT/'results.jsonl').exists():raise FileExistsError(OUT/'results.jsonl')
 signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('declared total runtime budget reached')));signal.alarm(a.budget)
 started=time.monotonic();save({'event':'start','smoke':a.smoke,'budget_seconds':a.budget,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Glauber exact discrete; both Kuramoto original and copula'})
 discrete();nested()
 for seed in ([0] if a.smoke else [0,1,2]):
  if time.monotonic()-started>a.budget:break
  try:mixed_case(seed,a.smoke)
  except Exception as e:save({'event':'failure','case':'I','seed':seed,'error':repr(e),'traceback':traceback.format_exc(),'smoke':a.smoke})
 signal.alarm(0);save({'event':'end','smoke':a.smoke,'elapsed_seconds':time.monotonic()-started,'budget_hit':time.monotonic()-started>=a.budget})
if __name__=='__main__':main()
