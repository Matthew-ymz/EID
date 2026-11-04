"""Development-only sample sensitivity of the 100D root with validation-selected ridge."""
import importlib.util,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);cfg=b.mk_system(100,101,0)
with (b.OUT/'root_more_samples.jsonl').open('w') as f:
 for N in (8192,32768):
  start=time.monotonic();rec={'n':100,'seed':101,'bridge':0,'samples':N,'tol_nats':b.TOL}
  try:
   x,y=b.data(cfg,N,10101);i=int(N*.6);j=int(N*.8);S,meta=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j]);o=b.Oracle(S,time.monotonic()+120);t=time.monotonic();l,r,v,_=b.pilot.queyranne_split(o,(1<<100)-1);cross=[g for g in cfg['truth'] if any(l>>z&1 for z in g) and any(r>>z&1 for z in g)];rec.update(status='complete',ridge=meta['chosen']['ridge'],left=[z for z in range(100) if l>>z&1],right=[z for z in range(100) if r>>z&1],syn_nats=v,truth_hyperedges_crossed=len(cross),search_seconds=time.monotonic()-t,queries=o.ei_queries,min_syn_nats=o.min_syn,negative_within_tol=o.near)
  except Exception as e:rec.update(status='failed',error=repr(e),traceback=traceback.format_exc())
  rec['elapsed_seconds']=time.monotonic()-start;f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in rec.items() if k not in ('left','right','traceback')}),flush=True)
