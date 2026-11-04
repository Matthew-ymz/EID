"""Post-search diagnostic: score planted parent boundaries only after fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,time
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
cfg=b.mk_system(100,101,0);x,y=b.data(cfg,4096,10101);i=int(len(x)*.6);j=int(len(x)*.8);S,meta=b.fit_constrained(x[:i],y[:i],x[i:j],y[i:j]);o=b.Oracle(S,time.monotonic()+30);full=(1<<100)-1
vals=[]
for group in cfg['outer']:
 mask=sum(1<<v for v in group);vals.append(o.gain((mask,full^mask)))
singles=[o.gain((1<<i,full^(1<<i))) for i in range(100)]
root=json.loads((b.OUT/'root_100.jsonl').read_text());out={'scope':'post-hoc structural diagnosis, no truth used by optimizer','true_parent_cut_syn_nats':vals,'smallest_singleton_cut_nats':min(singles),'largest_singleton_cut_nats':max(singles),'selected_root_cut_nats':root['split_syn_nats'],'tol_nats':b.TOL,'min_syn_nats':o.min_syn,'near_zero_negative_count':o.near,'note':'For bridge=0, the true channel factorizes across parents under independent input and noise, so planted 25-vs-75 boundaries have exact conditional mutual information zero.'};(b.OUT/'root_cut_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
