"""Independent root-only Queyranne check at 100 sources, under the same affine channel."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def main():
 out=b.OUT/'root_100.jsonl';t=time.monotonic();cfg=b.mk_system(100,101,0);x,y=b.data(cfg,4096,10101);i=int(len(x)*.6);j=int(len(x)*.8);S,meta=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j]);o=b.Oracle(S,time.monotonic()+300);rec={'seed':101,'n':100,'bridge':0,'estimator':meta,'tol_nats':b.TOL}
 try:
  st=time.monotonic();left,right,val,ncuts=b.pilot.queyranne_split(o,(1<<100)-1);cross=[g for g in cfg['truth'] if any(left>>v&1 for v in g) and any(right>>v&1 for v in g)];rec.update(status='complete',left=[v for v in range(100) if left>>v&1],right=[v for v in range(100) if right>>v&1],split_syn_nats=val,truth_hyperedges_crossed=len(cross),truth_hyperedges_total=len(cfg['truth']),unique_ei_queries=o.ei_queries,gain_calls=o.gain_queries,cut_cache_entries=ncuts,min_syn_nats=o.min_syn,near_zero_negative_count=o.near,search_seconds=time.monotonic()-st)
 except Exception as e:rec.update(status='failure',error=repr(e),traceback=traceback.format_exc(),unique_ei_queries=o.ei_queries,gain_calls=o.gain_queries,near_zero_negative_count=o.near,min_syn_nats=o.min_syn if o.gain_queries else None)
 rec['elapsed_seconds']=time.monotonic()-t;out.write_text(json.dumps(rec,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in rec.items() if k not in ('left','right','traceback')}),flush=True)
if __name__=='__main__':main()
