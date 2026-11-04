"""Exhaustive diminishing-return audit of the stored small neural-TM EI table."""
import itertools,json
from pathlib import Path
P=Path(__file__).resolve().parents[2]/'docs/log/spt_100d_stress/results';rows=[json.loads(s) for s in (P/'neural_importance_tree.jsonl').read_text().splitlines()]
with (P/'neural_submod_audit.jsonl').open('w') as f:
 for row in rows:
  if 'ei_nats' not in row:continue
  E={int(k):v for k,v in row['ei_nats'].items()};slack=[]
  for parent in range(1,64):
   def cut(s):return E[parent]-E[s]-E[parent^s]
   s=parent
   while True:
    absent=[1<<i for i in range(6) if parent>>i&1 and not s>>i&1]
    for a,b in itertools.combinations(absent,2):slack.append(cut(s|a)+cut(s|b)-cut(s)-cut(s|a|b))
    if s==0:break
    s=(s-1)&parent
  tol=row['syn_tolerance_nats'];ans={'draws':row['draws'],'count':len(slack),'minimum_slack_nats':min(slack),'below_zero':sum(v<0 for v in slack),'below_negative_tolerance':sum(v< -tol for v in slack),'tolerance_nats':tol,'note':'Tolerance is an exploratory numerical scale; nonnegativity tolerance does not by itself establish submodularity.'};f.write(json.dumps(ans)+'\n');print(json.dumps(ans))
