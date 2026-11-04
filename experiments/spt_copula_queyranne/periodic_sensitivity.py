"""Gaussian-copula source representation sensitivity on exact same saved-model draws."""
import importlib.util, json, math, time, hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('comparison_main',Path(__file__).with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
np=b.np
OUT=Path(__file__).parent/'results/periodic';OUT.mkdir(exist_ok=True);b.OUT=OUT
if (OUT/'results.jsonl').exists():raise FileExistsError(OUT/'results.jsonl')
previous=[r for r in map(json.loads,(OUT.parent/'results.jsonl').read_text().splitlines()) if r.get('estimator')=='gaussian_copula']
def lifted(x,y):
 xs=np.empty((len(x),12));xs[:,0::2]=np.sin(x);xs[:,1::2]=np.cos(x)
 z=np.column_stack((xs,y));z=b.ndtri(b.rankdata(z,axis=0,method='average')/(len(z)+1));C=np.cov(z,rowvar=False)+1e-6*np.eye(z.shape[1]);Y=list(range(12,C.shape[0]))
 def ld(ids):
  sign,v=np.linalg.slogdet(C[np.ix_(ids,ids)])
  if sign<=0:raise ValueError('nonpositive covariance')
  return v
 target=ld(Y);tab={0:0.}
 def query(m):
  inds=[2*i+j for i in range(6) if m>>i&1 for j in (0,1)];return float(.5*(ld(inds)+target-ld(inds+Y)))
 for m in range(1,64):tab[m]=query(m)
 return tab,query

def go(case,seed,x,y):
 old=next(r for r in previous if r['case']==case and r['seed']==seed)
 sha=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
 if sha!=old['metadata']['sample_hash']:raise ValueError(f'sample hash mismatch {case} {seed}')
 t=time.perf_counter();table,query=lifted(x,y);sec=time.perf_counter()-t
 b.evaluate(case,seed,'gaussian_copula_sincos',table,.1*b.LOG2 if case=='I' else .01,{7,56} if case=='I' else {7,15,48},{'n':len(x),'seconds_for_all_63_ei':sec,'sample_hash':sha,'target':'unchanged; same paired samples as raw angle','source_coordinates':'sin/cos, 2 dimensions retained per source','covariance_ridge':1e-6,'search_timing':'cold subset queries; excludes rank/covariance construction and full validity audit'},query)

base=b.SRC/'results/spt_third_case_weak_coupling';model=b.core.MLPChannel(json.loads((base/'model.json').read_text()))
for seed in (0,1,2):
 rng=np.random.default_rng(32000+seed);x=rng.uniform(-np.pi,np.pi,(4096,6));y=model.sample(x,rng.normal(size=x.shape));go('III',seed,x,y)
b.mixed._configure_coherent_tm()
for seed in (0,1,2):
 x0,y0=b.orig.generated_transition_data(sample_count=4800,seed=1000+seed,pairwise_coupling=b.mixed.WITHIN_COUPLING,triadic_coupling=b.mixed.WITHIN_COUPLING/math.sqrt(2),cross_coupling=.04,process_noise=.08,tau=.2,dt=.01)
 inc=np.angle(np.exp(1j*(y0-x0)));f=b.mixed._mlp_phase_features(x0);model=b.fit_mlp(f,inc,seed=2000+seed,epochs=800);res=inc[3840:]-model.predict(f[3840:]);cov=np.cov(res,rowvar=False)+1e-6*np.eye(6)
 x=np.random.default_rng(seed+3000).uniform(-np.pi,np.pi,(4000,6));y=model.predict(b.mixed._mlp_phase_features(x))+np.random.default_rng(seed+4000).multivariate_normal(np.zeros(6),cov,size=len(x));go('I',seed,x,y)
