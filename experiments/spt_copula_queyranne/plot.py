from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
P=Path(__file__).parent/'results'
a=[r for r in map(json.loads,(P/'results.jsonl').read_text().splitlines()) if r['event']=='evaluation' and r['status']=='valid']
b=[r for r in map(json.loads,(P/'periodic/results.jsonl').read_text().splitlines()) if r['event']=='evaluation' and r['status']=='valid']
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained',sharex=True)
configs={
'I':[('Original TM · exhaustive','original_component_neural_TM','exact'),('Original TM · Queyranne','original_component_neural_TM','queyranne'),('Copula angle · exhaustive','gaussian_copula','exact'),('Copula angle · Queyranne','gaussian_copula','queyranne'),('Copula sin/cos · exhaustive','gaussian_copula_sincos','exact'),('Copula sin/cos · Queyranne','gaussian_copula_sincos','queyranne')],
'II':[('Discrete EI · exhaustive','exact_discrete_fitted_MLP','exact'),('Discrete EI · Queyranne','exact_discrete_fitted_MLP','queyranne')],
'III':[('Original density · exhaustive','original_shared_channel_density','exact'),('Original density · Queyranne','original_shared_channel_density','queyranne'),('Copula angle · exhaustive','gaussian_copula','exact'),('Copula angle · Queyranne','gaussian_copula','queyranne'),('Copula sin/cos · exhaustive','gaussian_copula_sincos','exact'),('Copula sin/cos · Queyranne','gaussian_copula_sincos','queyranne')]}
for ax,(case,title) in zip(axes,[('I','I  Mixed Kuramoto'),('II','II  Glauber'),('III','III  Nested Kuramoto')]):
 rows=configs[case]
 for j,(label,method,algo) in enumerate(rows):
  data=b if method.endswith('sincos') else a
  records=sorted([r for r in data if r['case']==case and r['estimator']==method],key=lambda r:r['seed'])
  for i,r in enumerate(records):
   val=r[algo]['f1'];ax.scatter(val,j+(i-1)*.13,s=24,color='#286f88' if algo=='exact' else '#c27540',alpha=.8,zorder=3)
 ax.set_yticks(range(len(rows)),[q[0] for q in rows],fontsize=8)
 ax.invert_yaxis();ax.set_xlim(-.07,1.07);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Planted-module F1')
 ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.16);ax.spines[['top','right']].set_visible(False)
fig.savefig(P/'module_recovery.png',dpi=180)
