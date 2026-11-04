"""Independent 100D root-only Queyranne replicates; no labels enter search."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
dev=json.loads((b.OUT/'root_100.jsonl').read_text());lam=dev['estimator']['chosen']['ridge'];out=b.OUT/'root_replicates.jsonl'
with out.open('w') as f:
 for bridge in (0,1):
  for seed in (201,202,203):
   start=time.monotonic();rec={'seed':seed,'bridge':bridge,'n':100,'samples':8192,'ridge_from_development':lam,'tol_nats':b.TOL}
   try:
    cfg=b.mk_system(100,seed,bridge);x,y=b.data(cfg,8192,seed+10000);i=int(len(x)*.6);j=int(len(x)*.8);S,_=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j],lam);o=b.Oracle(S,time.monotonic()+120);search=time.monotonic();left,right,val,ncuts=b.pilot.queyranne_split(o,(1<<100)-1);cross=[g for g in cfg['truth'] if any(left>>v&1 for v in g) and any(right>>v&1 for v in g)]
    rec.update(status='complete',left=[v for v in range(100) if left>>v&1],right=[v for v in range(100) if right>>v&1],split_syn_nats=val,truth_hyperedges_crossed=len(cross),truth_hyperedges_total=len(cfg['truth']),unique_ei_queries=o.ei_queries,gain_calls=o.gain_queries,near_zero_negative_count=o.near,min_syn_nats=o.min_syn,search_seconds=time.monotonic()-search)
   except Exception as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc())
   rec['elapsed_seconds']=time.monotonic()-start;f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in rec.items() if k not in ('left','right','traceback')}),flush=True)
