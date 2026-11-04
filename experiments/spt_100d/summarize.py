"""Aggregate only completed runs, and explicitly list missing/failed 100D runs."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2];D=R/'docs/log/spt_100d_stress/results'
def records(name):
 p=D/name;return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
runs=records('results.jsonl')+records('bridge1_parallel.jsonl');seen={}
for row in runs:
 if row.get('n')==100 and row.get('seed') in (201,202,203) and row.get('event') in ('case','failure'):
  key=(row['seed'],row['bridge'])
  if key in seen:raise ValueError(f'duplicate test condition: {key}')
  seen[key]=row
out={'conditions':[],'paired_clade_jaccard':[]}
for bridge in (0,1):
 for seed in (201,202,203):
  d=seen.get((seed,bridge));r={'seed':seed,'bridge':bridge,'status':'missing' if d is None else d['event']}
  if d and d['event']=='case':
   q=d['queyranne'];ss=q['search'];r.update(scores=q['scores'],search_seconds=ss['seconds'],unique_ei_queries=ss['unique_ei_queries'],min_syn_nats=ss['min_candidate_syn_nats'],near_zero_negative_count=ss['near_zero_negative_count'],tol_nats=ss['tol_nats'],closure_error_nats=ss['closure_error'],spectral_scores=d['spectral']['scores'],spectral_search_seconds=d['spectral']['seconds_including_pairwise_affinity'])
   if ss['min_candidate_syn_nats'] < -ss['tol_nats']:raise ArithmeticError(f'invalid Syn for {(seed,bridge)}')
  elif d:r['error']=d.get('error')
  out['conditions'].append(r)
def clades(node):
 x=set();s=frozenset(node['sources'])
 if 1<len(s)<100:x.add(s)
 for child in node['children']:x |= clades(child)
 return x
for seed in (201,202,203):
 a=seen.get((seed,0));b=seen.get((seed,1))
 if a and b and a['event']==b['event']=='case':
  l=clades(a['queyranne']['tree']);r=clades(b['queyranne']['tree']);out['paired_clade_jaccard'].append({'seed':seed,'no_bridge_clades':len(l),'bridge_clades':len(r),'shared':len(l&r),'jaccard':len(l&r)/len(l|r)})
(D/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False))
for c in out['conditions']:
 print(c['seed'],c['bridge'],c['status'],c.get('scores',{}).get('matched_count'),c.get('search_seconds'),c.get('error'))
for c in out['paired_clade_jaccard']:print('paired',c)
