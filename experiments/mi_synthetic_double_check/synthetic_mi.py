"""Budgeted paired MI pilot; run with --smoke first. See README for contract."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import torch
from torch import nn
from scipy.ndimage import median_filter, gaussian_filter1d
from eid import estimate_mi

torch.set_num_threads(1)
OUT=Path(__file__).parent/'results'
LOG2=math.log(2)

def teacher(seed,n,truth):
    rng=np.random.default_rng(seed)
    rho=math.sqrt(1-2**(-.8)) if truth else 0.0
    zx=rng.normal(size=(n,10)); zy=rho*zx+math.sqrt(1-rho*rho)*rng.normal(size=(n,10))
    # These fixed maps do not depend on the sample seed.
    def embed(z,mapseed):
        mr=np.random.default_rng(mapseed)
        w=mr.normal(size=(10,480))/math.sqrt(10); b=mr.normal(size=480)*.2
        return np.concatenate((np.logaddexp(0,z), np.logaddexp(0,-z), np.logaddexp(0,z@w+b)),axis=1)
    return embed(zx,100),embed(zy,200)

def split_outer(x,y):
    n=len(x)
    return (x[:int(.8*n)],y[:int(.8*n)]),(x[int(.8*n):],y[int(.8*n):])

def normalize(trainx,trainy,*pairs):
    mx=trainx.mean(0); my=trainy.mean(0); sx=trainx.std(0); sy=trainy.std(0)
    sx=np.where(sx>1e-8,sx,1); sy=np.where(sy>1e-8,sy,1)
    return [((a-mx)/sx,(b-my)/sy) for a,b in pairs]

def tm_record(x,y,tx,ty,jitter):
    (x,y),(tx,ty)=normalize(x,y,(x,y),(tx,ty))
    result={}
    for mode in ('train','heldout'):
        try:
            options={'jitter':jitter,'bias_correction':False}
            kw={} if mode=='train' else {'evaluation_x':tx,'evaluation_y':ty}
            v=estimate_mi(x,y,estimator='tm_affine',estimator_options=options,unit='bits',**kw)
            result[mode]=float(v.value)
        except Exception as exc:
            result[mode]=None; result[mode+'_error']=repr(exc)
    return result

class Critic(nn.Module):
    def __init__(self,k):
        super().__init__()
        def net(): return nn.Sequential(nn.Linear(500,256),nn.LeakyReLU(.01),nn.Linear(256,256),nn.LeakyReLU(.01),nn.Linear(256,k))
        self.x=net(); self.y=net()
    def forward(self,x,y): return self.x(x)@self.y(y).T / math.sqrt(self.x[-1].out_features)

def score(model,x,y):
    s=model(x,y)
    return (torch.diagonal(s)-torch.logsumexp(s,dim=1)+math.log(len(x))).mean()/LOG2

def neural_fit(x,y,outerx,outery,k,seed,epochs=100):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); n=len(x)
    order=rng.permutation(n); m=int(n*.75); ix=order[:m];iv=order[m:]
    if len(ix)<16 or len(iv)<8: return {'error':'insufficient disjoint train/validation','train_n':len(ix),'validation_n':len(iv)}
    (a,b),(c,d),(e,f)=normalize(x[ix],y[ix],(x[ix],y[ix]),(x[iv],y[iv]),(outerx,outery))
    a,b,c,d,e,f=[torch.as_tensor(z,dtype=torch.float32) for z in (a,b,c,d,e,f)]
    model=Critic(k); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    batch=min(128,len(ix)); bestval=-float('inf'); beststate=None; bestepoch=0; values=[]
    for epoch in range(epochs):
        model.train(); perm=torch.randperm(len(a))
        for ind in perm.split(batch):
            if len(ind)<2: continue
            opt.zero_grad(); loss=-score(model,a[ind],b[ind]); loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            # fixed training and validation samples; cap 128 as in paper
            tr=score(model,a[:batch],b[:batch]).item(); v=score(model,c[:128],d[:128]).item()
        values.append([tr,v])
    v=np.array(values); window=min(5,len(v) if len(v)%2 else len(v)-1)
    smooth=gaussian_filter1d(median_filter(v[:,1],size=max(1,window),mode='nearest'),1)
    chosen=int(np.argmax(smooth))
    # Refit deterministically to selected epoch. This also preserves a final untouched test.
    torch.manual_seed(seed); model=Critic(k); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    for epoch in range(chosen+1):
        model.train(); perm=torch.randperm(len(a))
        for ind in perm.split(batch):
            if len(ind)<2: continue
            opt.zero_grad(); loss=-score(model,a[ind],b[ind]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        train=score(model,a[:batch],b[:batch]).item()
        val=score(model,c[:128],d[:128]).item()
        test=score(model,e[:128],f[:128]).item()
    return {'train':train,'validation':val,'heldout':test,'best_epoch':chosen+1,'k':k,'batch':batch,
            'train_n':len(ix),'validation_n':len(iv),'heldout_n_used':min(128,len(e)),
            'cap_bits':math.log2(batch),'raw_train_curve':v[:,0].tolist(),'raw_validation_curve':v[:,1].tolist()}

def extrapolate(records):
    ok=[r for r in records if 'train' in r]; gs=sorted(set(r['gamma'] for r in ok))
    if len(gs)<6: return {'reliable':False,'reason':'fewer than six partition levels','gamma_levels':gs}
    p=np.array([r['gamma'] for r in ok],float); z=np.array([r['train'] for r in ok]); weights=1/p
    co=np.polyfit(p,z,2,w=np.sqrt(weights)); delta=abs(co[0]/co[1]) if abs(co[1])>1e-12 else None
    fit=np.polyfit(p,z,1,w=np.sqrt(weights)); means={str(g):float(np.mean([r['train'] for r in ok if r['gamma']==g])) for g in gs}
    # k-saturation, residual symmetry and true coverage are not established.
    return {'candidate_bits':float(fit[1]),'quadratic_delta':delta,'gamma_levels':gs,'means':means,
            'reliable':False,'reason':'pilot: only six levels, finite-sample and k-saturation checks incomplete'}

def append(row,path):
    with path.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('fits','curve')}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--budget',type=int,default=1800); args=p.parse_args()
    OUT.mkdir(exist_ok=True); path=OUT/('smoke.jsonl' if args.smoke else 'results.jsonl')
    if path.exists(): raise FileExistsError(path)
    start=time.monotonic(); end=start+args.budget
    meta={'event':'start','seed_set':[0,1,2],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'budget_seconds':args.budget,'smoke':args.smoke}
    append(meta,path)
    datasets=[]
    for n in ([256] if args.smoke else [256,1199]):
        for seed in ([0] if args.smoke else [0,1,2]):
            for truth in ([4] if args.smoke else [4,0]):
                x,y=teacher(seed+1000,n,truth)
                datasets.append(('synthetic',f'n{n}_seed{seed}_truth{truth}',x,y,truth))
    # First obtain directly comparable TM and one full-data neural fit for every dataset.
    for kind,name,x,y,truth in datasets:
        if time.monotonic()>end: break
        (x,y),(tx,ty)=split_outer(x,y)
        row={'event':'primary','kind':kind,'name':name,'truth_bits':truth,'total_n':len(x)+len(tx),'train_pool_n':len(x),'outer_n':len(tx),'tm':{}}
        for ridge in (.01,.1): row['tm'][str(ridge)]=tm_record(x,y,tx,ty,ridge)
        row['neural']=neural_fit(x,y,tx,ty,32,0,epochs=8 if args.smoke else 100)
        row['elapsed']=time.monotonic()-start;append(row,path)
    # Subsampling protocol on the synthetic 4-bit controls.
    if not args.smoke:
        for kind,name,x,y,truth in datasets:
            if 'seed0' not in name or truth!=4: continue
            (x,y),(tx,ty)=split_outer(x,y)
            for seed in range(3):
                for k in (8,32):
                    fits=[]; rng=np.random.default_rng(seed)
                    for gamma in range(1,7):
                        inds=rng.permutation(len(x))
                        for idx in np.array_split(inds,gamma):
                            if time.monotonic()>end: break
                            a=neural_fit(x[idx],y[idx],tx,ty,k,seed+gamma*100)
                            a.update(gamma=gamma,n=len(idx));fits.append(a)
                        if time.monotonic()>end: break
                    row={'event':'subsample','kind':kind,'name':name,'seed':seed,'k':k,'fits':fits,'diagnostic':extrapolate(fits),'elapsed':time.monotonic()-start}
                    append(row,path)
                    if time.monotonic()>end: break
                if time.monotonic()>end: break
            if time.monotonic()>end: break
    append({'event':'end','elapsed_seconds':time.monotonic()-start,'budget_hit':time.monotonic()>end},path)
if __name__=='__main__': main()
