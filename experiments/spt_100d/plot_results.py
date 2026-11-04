"""Plot recorded 100D stress outcomes; uncompleted search results remain absent."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];D=R/'docs/log/spt_100d_stress/results';O=R/'fig/spt_100d_stress.png';O.parent.mkdir(exist_ok=True)
def lines(path):
 return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
cases=[d for d in lines(D/'results.jsonl') if d.get('event')=='case' and d.get('seed')==101]; tests=[d for d in lines(D/'results.jsonl')+lines(D/'bridge1_parallel.jsonl') if d.get('event')=='case' and d.get('n')==100 and d.get('seed') in (201,202,203)]; spec=[d for d in lines(D/'spectral_100.jsonl') if d.get('status')=='complete']; noise=lines(D/'integration_noise_diagnostic.jsonl'); orig=lines(D/'integration_diagnostic.jsonl'); samples=lines(D/'parent_cut_samples.jsonl')
plt.rcParams.update({'font.family':'sans-serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(10.2,6.8),layout='constrained');a,b,c,d=axs.flatten()
navy='#1F3B73';teal='#2A9D8F';ochre='#C56B3C';red='#A23E48'
for v in cases:
 n=v['n'];a.scatter(n-.7,v['queyranne']['scores']['recall']*100,marker='o',color=navy,s=25,zorder=3);a.scatter(n+.7,v['spectral']['scores']['recall']*100,marker='s',facecolors='none',edgecolors=teal,s=25,zorder=3)
 b.scatter(n-.7,v['queyranne']['search']['seconds'],marker='o',color=navy,s=25,zorder=3);b.scatter(n+.7,v['spectral']['seconds_including_pairwise_affinity'],marker='s',facecolors='none',edgecolors=teal,s=25,zorder=3)
for v in spec:
 if v['seed']==101:
  if not any(x['n']==100 for x in cases):
   a.scatter(100.7,v['scores']['recall']*100,marker='s',facecolors='none',edgecolors=teal,s=25,zorder=3);b.scatter(100.7,v['search_seconds'],marker='s',facecolors='none',edgecolors=teal,s=25,zorder=3)
 elif v['seed'] in (201,202,203):
  pos=112+(v['seed']-202)*1.3+(-.35 if v['bridge']==0 else .35);a.scatter(pos,v['scores']['recall']*100,marker='s',facecolors='none',edgecolors=teal,s=25,zorder=3)
for v in tests:
 pos=112+(v['seed']-202)*1.3+(-.35 if v['bridge']==0 else .35);a.scatter(pos-.3,v['queyranne']['scores']['recall']*100,marker='o',color=navy,s=19,zorder=2)
a.set_xticks([6,20,50,100,112],['6','20','50','100','100\ntest']);a.set_xlim(-2,119);a.set_ylim(-4,105);a.set_ylabel('Exact planted-module recall (%)');a.set_xlabel('Source dimension');a.text(.03,.93,'a',transform=a.transAxes,fontweight='bold',va='top');a.text(.96,.9,'100D test: 3 seeds × 2 conditions\nQueyranne: 1 timeout',transform=a.transAxes,ha='right',va='top',fontsize=8)
b.set_yscale('log');b.set_xticks([6,20,50,100]);b.set_xlim(-2,107);b.set_xlabel('Source dimension (development)');b.set_ylabel('Full-tree search time (s)');b.text(.03,.93,'b',transform=b.transAxes,fontweight='bold',va='top');b.scatter([],[],marker='o',color=navy,s=25,label='Queyranne');b.scatter([],[],marker='s',facecolors='none',edgecolors=teal,s=25,label='Spectral');b.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
if not any(x['n']==100 for x in cases):b.text(.96,.08,'100D Queyranne\nnot completed',transform=b.transAxes,ha='right',va='bottom',fontsize=8,color='0.35')
for v in samples:
 for value in v['planted_parent_cut_nats']:c.scatter(v['samples'],value,marker='o',color=navy,s=18,alpha=.75)
 c.scatter(v['samples'],v['smallest_singleton_cut_nats'],marker='s',facecolors='none',edgecolors=teal,s=26)
c.scatter([],[],marker='o',color=navy,s=18,label='Parent cut, true 0');c.scatter([],[],marker='s',facecolors='none',edgecolors=teal,s=26,label='Best singleton cut');c.set_xscale('log',base=2);c.set_yscale('log');c.set_xticks([4096,8192,16384,32768],['4k','8k','16k','32k']);c.set_xlabel('Total development samples');c.set_ylabel('Estimated cut Syn (nats)');c.text(.03,.93,'c',transform=c.transAxes,fontweight='bold',va='top');c.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
if orig:d.plot([x['draws'] for x in orig],[x['syn_nats'] for x in orig],'-o',markersize=3,color=ochre,label='6D, naive, σ = 0.08')
if noise:d.plot([x['draws'] for x in noise],[x['syn_nats'] for x in noise],'-s',markersize=3,color=red,label='6D, naive, σ = 0.8')
new=D/'neural_100_feasibility.json'
if new.exists():
 z=json.loads(new.read_text())['integration'];d.plot([x['draws'] for x in z],[x['syn_nats'] for x in z],'--^',markersize=3,color=navy,label='100D, broad proposal')
new=D/'neural_100_tuned_proposal.json'
if new.exists():
 z=json.loads(new.read_text())['integration'];d.plot([x['draws'] for x in z],[x['syn_nats'] for x in z],'--D',markersize=3,color=teal,label='100D, tuned proposal')
d.axhline(-.06931471805599453,color='0.4',lw=.8,linestyle='--',label='−0.069 nats tolerance');d.axhline(0,color='0.75',lw=.7);d.set_xscale('log');d.set_yscale('symlog',linthresh=.05);d.set_ylim(-2000,.1);d.set_yticks([-1000,-100,-10,-1,-.1,0],['−1000','−100','−10','−1','−0.1','0']);d.set_xlabel('Marginal integration draws');d.set_ylabel('Cut Syn (nats)');d.text(.03,.93,'d',transform=d.transAxes,fontweight='bold',va='top');d.legend(loc='center left',bbox_to_anchor=(1.02,.5),frameon=False,fontsize=8)
fig.savefig(O,dpi=220);print(O)
