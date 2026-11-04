"""Increase marginal importance draws after the exploratory 2048-draw audit."""
import importlib.util,json
from pathlib import Path
P=Path(__file__).resolve();spec=importlib.util.spec_from_file_location('tree',P.with_name('neural_importance_tree.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
v=m.run(8192)
with (P.parents[2]/'docs/log/spt_100d_stress/results/neural_importance_tree.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
print(json.dumps({k:z for k,z in v.items() if k not in ('ei_nats','tree','traceback')}),flush=True)
