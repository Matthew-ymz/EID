from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
P=Path(__file__).parent
cases=[v for v in map(json.loads,(P/'results/results.jsonl').read_text().splitlines()) if v['event']=='case']
methods=[('InfoNCE / train','infonce','train'),('InfoNCE / test','infonce','heldout'),('Copula / model','copula_0.1','model'),('Copula / test','copula_0.1','heldout'),('Affine TM / model','affine_0.1','model'),('Affine TM / test','affine_0.1','heldout'),('Sparse quadratic / train','sparse_tm_d2','train'),('Sparse quadratic / test','sparse_tm_d2','heldout'),('Sparse cubic / train','sparse_tm_d3','train'),('Sparse cubic / test','sparse_tm_d3','heldout')]
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained',sharex=True)
colors={0:'#2f6a89',4:'#b4583f'}
for i,kind in enumerate(('teacher','gaussian')):
 for j,n in enumerate((256,1199)):
    ax=axes[i,j]
    for t in (0,4):
        selected=[r for r in cases if r['kind']==kind and r['n']==n and r['truth_bits']==t]
        for k,(_,m,which) in enumerate(methods):
            vals=np.array([abs(r['methods'][m]['result'][which]-t) for r in selected])
            if np.any(vals<=0): raise ValueError('Zero errors cannot be represented on log axes')
            ypos=k + (-.13 if t==0 else .13)
            ax.scatter(vals, np.full_like(vals,ypos),color=colors[t],s=14,alpha=.55,linewidths=0)
            ax.scatter([np.median(vals)],[ypos],color=colors[t],s=45,marker='|',linewidths=2,label=f'True MI: {t} bits' if k==0 and i==0 and j==0 else None)
    ax.set_yticks(range(len(methods)),[m[0] for m in methods],fontsize=8)
    ax.invert_yaxis(); ax.set_xscale('log'); ax.grid(axis='x',which='major',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(('Low-rank nonlinear' if kind=='teacher' else 'Full-rank Gaussian')+f'  ·  N = {n}',fontsize=10)
    if i==1: ax.set_xlabel('Absolute error (bits, log scale)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,title='Data conditions',loc='outside right center',frameon=False)
fig.savefig(P/'results/benchmark.png',dpi=180)
