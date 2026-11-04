"""Post-hoc best-overlap vs node-permutation null; no labels enter fitting/search."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2];D=R/'docs/log/spt_100d_stress/results'
def lines(name):
 p=D/name;return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def tree_masks(node):
 out=[];a=node['sources']
 if 1<len(a)<100:out.append(sum(1<<i for i in a))
 for child in node['children']:out+=tree_masks(child)
 return out
def score(groups,tree):
 res={}
 for ids in groups:
  mask=sum(1<<int(i) for i in ids);best=max((mask&t).bit_count()/(mask|t).bit_count() for t in tree);res.setdefault(len(ids),[]).append(best)
 return {k:float(np.mean(v)) for k,v in res.items()}
rows=[r for r in lines('results.jsonl')+lines('bridge1_parallel.jsonl') if r.get('event')=='case' and r.get('n')==100];out=[]
for d in rows:
 tree=tree_masks(d['queyranne']['tree']);truth=d['truth'];obs=score(truth,tree);rng=np.random.default_rng(2077);null={k:[] for k in obs}
 for _ in range(200):
  p=rng.permutation(100);z=score([[int(p[i]) for i in g] for g in truth],tree)
  for k,v in z.items():null[k].append(v)
 result={'seed':d['seed'],'bridge':d['bridge'],'method':'post-hoc max Jaccard of each planted module with any nontrivial clade; 200 global label permutations','orders':{str(k):{'observed_mean_best_jaccard':obs[k],'null_mean':float(np.mean(null[k])),'null_95pct_range':[float(v) for v in np.percentile(null[k],[2.5,97.5])]} for k in sorted(obs)}};out.append(result)
(D/'partial_overlap.json').write_text(json.dumps(out,indent=2));
for r in out:print(r['seed'],r['bridge'],r['orders'])
