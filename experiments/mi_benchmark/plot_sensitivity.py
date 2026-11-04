from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
P=Path(__file__).parent/'results'
a=[r for r in map(json.loads,(P/'results.jsonl').read_text().splitlines()) if r['event']=='case']
b=[r for r in map(json.loads,(P/'sensitivity.jsonl').read_text().splitlines()) if r['event']=='sensitivity']
fig,axes=plt.subplots(2,2,figsize=(10.5,7),layout='constrained',sharey='row')
colors={'copula':'#2e6f8c','affine':'#b4533e'}
for i,kind in enumerate(('teacher','gaussian')):
 for j,truth in enumerate((0,4)):
  ax=axes[i,j]
  for method in ('copula','affine'):
   for mode,style in (('model','-'),('heldout','--')):
    med=[];err=[]
    for ridge in (.01,.1,1.,10.):
     data=a if ridge<1 else b
     vals=[r['methods'][f'{method}_{ridge}']['result'][mode] for r in data if r['kind']==kind and r['n']==1199 and r['truth_bits']==truth]
     med.append(np.mean(vals));err.append(np.std(vals,ddof=1))
    ax.errorbar([.01,.1,1.,10.],med,yerr=err,color=colors[method],linestyle=style,marker='o',markersize=3,capsize=2,lw=1,label=f'{method.title()} / {"model" if mode=="model" else "test"}')
  ax.axhline(truth,color='#444',lw=.9)
  ax.axhline(0,color='#999',lw=.5)
  ax.set_xscale('log');ax.set_yscale('symlog',linthresh=2);ax.set_title(('Low-rank nonlinear' if i==0 else 'Full-rank Gaussian')+f' · truth {truth} bits',fontsize=10)
  ax.set_xticks([.01,.1,1,10],['0.01','0.1','1','10']);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
  if j==0: ax.set_ylabel('MI estimate (bits; symlog)')
  if i==1: ax.set_xlabel('Covariance ridge')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside right center',frameon=False)
fig.savefig(P/'sensitivity.png',dpi=180)
