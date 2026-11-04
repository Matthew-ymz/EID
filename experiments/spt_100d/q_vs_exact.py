"""Paired root timings for Queyranne vs exhaustive on identical fitted EI oracle."""
import importlib.util,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);cfg=b.mk_system(20,101,0);x,y=b.data(cfg,4096,10101);i=int(len(x)*.6);j=int(len(x)*.8);S,_=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j]);out=b.OUT/'q_vs_exact.jsonl'
with out.open('w') as f:
 for n in (6,8,10,12,14,16,18,20):
  ss=S[:2*n,:2*n];row={'n':n,'tol_nats':b.TOL}
  try:
   q=b.Oracle(ss,time.monotonic()+120);t=time.monotonic();l,r,qv,c=b.pilot.queyranne_split(q,(1<<n)-1);row.update(q_seconds=time.monotonic()-t,q_value_nats=qv,q_unique_ei_queries=q.ei_queries,q_min_syn_nats=q.min_syn,q_near_negative=q.near)
   e=b.Oracle(ss,time.monotonic()+120);t=time.monotonic();ev,el,er=b.exhaustive(e,(1<<n)-1);row.update(exact_seconds=time.monotonic()-t,exact_value_nats=ev,exact_unique_ei_queries=e.ei_queries,exact_min_syn_nats=e.min_syn,exact_near_negative=e.near,gap_nats=qv-ev,status='complete')
  except Exception as z:row.update(status='failure',error=repr(z),traceback=traceback.format_exc())
  f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k!='traceback'}),flush=True)
