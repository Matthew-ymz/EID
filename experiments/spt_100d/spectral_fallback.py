"""Supplementary 100-source spectral search if full Queyranne is too slow.
No truth labels enter fitting or search; fixed validation-selected ridge.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from eid.spt import build_spt,SPTConfig,spectral_candidate_selector,pairwise_syn_affinity

def main():
 out=base.OUT/'spectral_100.jsonl';started=time.monotonic();selected=None
 with out.open('w') as f:
  for seed,bridge,N in [(101,0,4096)]+[(s,b,8192) for b in (0,1) for s in (201,202,203)]:
   ts=time.monotonic();rec={'n':100,'seed':seed,'bridge':bridge,'samples':N,'method':'constrained affine Gaussian moment surrogate + spectral SPT','tol_nats':base.TOL}
   try:
    cfg=base.mk_system(100,seed,bridge);x,y=base.data(cfg,N,seed+10000);ix=int(N*.6);iv=int(N*.8)
    S,meta=base.fit_constrained(x[:ix],y[:ix],x[ix:iv],y[ix:iv],None if seed==101 else selected)
    if seed==101:selected=meta['chosen']['ridge']
    rec['estimator']=meta;rec['preparation_seconds']=time.monotonic()-ts
    o=base.Oracle(S,time.monotonic()+240);start=time.monotonic();W,pair_near=pairwise_syn_affinity(o,100,tolerance=base.TOL)
    class Adapter:
     def xi(self,s):return o.xi(s)
    config=SPTConfig(syn_tolerance=base.TOL,eps=1e-10,complete_to_singletons=True)
    result=build_spt(tuple(range(100)),Adapter(),config=config,candidate_selector=spectral_candidate_selector(W,exact_max_size=8))
    def conv(z):return {'mask':sum(1<<int(i) for i in z.sources),'syn':z.syn_value,'children':[conv(c) for c in z.children]}
    tree=conv(result.root);rec.update(status='complete',scores=base.modules(tree,cfg),tree=base.serial_tree(tree),search_seconds=time.monotonic()-start,queries=o.ei_queries,near_zero_negative_count=pair_near+result.audit.tolerance_zero_count,min_candidate_syn_nats=result.audit.minimum_candidate_syn,closure_error=result.closure_error)
   except Exception as e:rec.update(status='failure',error=repr(e),traceback=traceback.format_exc())
   rec['elapsed_seconds']=time.monotonic()-ts
   f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in rec.items() if k not in ('tree','traceback')}),flush=True)
   if seed==101 and selected is None:break
if __name__=='__main__':main()
