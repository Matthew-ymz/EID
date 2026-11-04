"""Independent test-seed zero conditional-MI references, fixed development lambda."""
import importlib.util,json,time
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
dev=json.loads((b.OUT/'root_100.jsonl').read_text());lam=dev['estimator']['chosen']['ridge'];out=b.OUT/'parent_cut_replicates.jsonl'
with out.open('w') as f:
 for seed in (201,202,203):
  cfg=b.mk_system(100,seed,0);x,y=b.data(cfg,8192,seed+10000);i=int(len(x)*.6);j=int(len(x)*.8);S,_=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j],lam);o=b.Oracle(S,time.monotonic()+30);full=(1<<100)-1;vals=[]
  for g in cfg['outer']:
   mask=sum(1<<v for v in g);vals.append(o.gain((mask,full^mask)))
  rec={'seed':seed,'samples':8192,'bridge':0,'lambda':lam,'true_parent_cut_nats':[0.,0.,0.,0.],'estimated_parent_cut_nats':vals,'tol_nats':b.TOL,'negative_within_tol':o.near}
  f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
