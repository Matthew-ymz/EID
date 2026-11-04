"""Post-hoc oracle-frequency diagnostic; labels used only to assess planted response."""
import importlib.util,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);cfg=b.mk_system(100,101,0);N=8192;x,y=b.data(cfg,N,10101);delta=y-x-cfg['tau']*cfg['omega'];res=[]
for ids,weight in cfg['edges']:
 z=x[:,ids];phi=np.sin(z.sum(axis=1)[:,None]-len(ids)*z);a=np.mean(delta[:,ids]*phi,axis=1);v=np.mean(phi**2,axis=1);slope=float(a.mean()/v.mean());sem=float(np.std(a-slope*v,ddof=1)/np.sqrt(N)/v.mean());res.append({'order':len(ids),'weight':weight,'coefficient':slope,'sem':sem,'approx_95pct_interval':[slope-1.96*sem,slope+1.96*sem]})
out={'n':100,'seed':101,'bridge':0,'samples':N,'role':'post-hoc diagnostic using planted hyperedge labels; not used for estimation or search','quantity':'projection of finite-horizon phase increment on each planted initial-phase Fourier term; not EI or Syn','edges':res};(b.OUT/'planted_drift_audit.json').write_text(json.dumps(out,indent=2));
for k in sorted(set(x['order'] for x in res)):
 q=[x for x in res if x['order']==k];print('order',k,'coefficients',[round(x['coefficient'],5) for x in q],'SE',[round(x['sem'],5) for x in q])
