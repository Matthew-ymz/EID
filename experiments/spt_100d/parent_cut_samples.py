"""Development-only zero-CMI benchmark; labels used solely for post-fit diagnosis."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
cfg=b.mk_system(100,101,0);full=(1<<100)-1;out=b.OUT/'parent_cut_samples.jsonl'
with out.open('w') as f:
 for N in (4096,8192,16384,32768):
  t=time.monotonic();x,y=b.data(cfg,N,10101);i=int(N*.6);j=int(N*.8);S,meta=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j]);o=b.Oracle(S,time.monotonic()+60);vals=[]
  for group in cfg['outer']:
   mask=sum(1<<v for v in group);vals.append(o.gain((mask,full^mask)))
  singles=[o.gain((1<<i,full^(1<<i))) for i in range(100)]
  rec={'samples':N,'training_samples':i,'validation_samples':j-i,'lambda':meta['chosen']['ridge'],'validation_nll_per_target':meta['chosen']['validation_nll_per_target'],'planted_parent_cut_nats':vals,'smallest_singleton_cut_nats':min(singles),'expected_true_parent_cut_nats':0.,'tol_nats':b.TOL,'negative_within_tol':o.near,'elapsed_seconds':time.monotonic()-t}
  f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
