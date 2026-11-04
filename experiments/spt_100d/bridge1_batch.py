"""Parallelize the predeclared bridge=1 cases; preserve original per-case limit."""
import importlib.util,json,time,traceback
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('stress',P.with_name('run.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
dev=[json.loads(s) for s in (b.OUT/'results.jsonl').read_text().splitlines() if json.loads(s).get('event')=='case' and json.loads(s).get('n')==100 and json.loads(s).get('seed')==101][0];lam=dev['estimator']['chosen']['ridge'];start=time.monotonic();b.clock_end=start+4500
with (b.OUT/'bridge1_parallel.jsonl').open('w') as f:
 for seed in (201,202,203):
  try:row=b.eval_case(100,seed,1,8192,case_limit=720,fixed_lambda=lam)
  except Exception as e:row={'event':'failure','n':100,'seed':seed,'bridge':1,'error':repr(e),'traceback':traceback.format_exc()}
  f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k not in ('truth','permutation','queyranne','spectral','traceback')}),flush=True)
