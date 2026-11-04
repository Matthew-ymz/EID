"""Pre-registered 500-by-500 observational MI benchmark, in bits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, itertools, json, math, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/mi_synthetic_double_check'))
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from scipy.linalg import cho_factor, cho_solve
from eid import estimate_mi
import synthetic_mi as prev

OUT=Path(__file__).parent/'results'
LOG2=np.log(2.)

def gaussian(seed,n,truth):
    rng=np.random.default_rng(2000+seed)
    rho=np.sqrt(1-2**(-2*truth/500)) if truth else 0.
    x=rng.standard_normal((n,500)); y=rho*x+np.sqrt(1-rho*rho)*rng.standard_normal((n,500))
    return x,y

def split(x,y):
    n=len(x); cutoff=int(.8*n); a=np.arange(cutoff); rng=np.random.default_rng(0); a=rng.permutation(a); m=int(.75*len(a));
    return (x[a[:m]],y[a[:m]]),(x[a[m:]],y[a[m:]]),(x[cutoff:],y[cutoff:])

def copula(train,evals):
    n,d=train.shape
    tr=ndtri(rankdata(train,axis=0,method='average')/(n+1))
    out=[np.empty_like(a) for a in evals]
    for j in range(d):
        order=np.argsort(train[:,j],kind='stable'); values=train[order,j]; scores=tr[order,j]
        for a,z in zip(evals,out): z[:,j]=np.interp(a[:,j],values,scores,left=scores[0],right=scores[-1])
    return tr,out

def gaussian_ratio(x,y,evals,jitter):
    def fit(data):
        mu=data.mean(0); cv=np.atleast_2d(np.cov(data,rowvar=False))+jitter*np.eye(data.shape[1]); c=cho_factor(cv,lower=True,check_finite=False); ld=2*np.log(np.diag(c[0])).sum();return mu,c,ld
    fx,fy,fj=fit(x),fit(y),fit(np.column_stack((x,y)))
    model=.5*(fx[2]+fy[2]-fj[2])/LOG2
    def logp(z,f):
        a=z-f[0]; q=np.einsum('ij,ji->i',a,cho_solve(f[1],a.T,check_finite=False));return -.5*(q+f[2]+z.shape[1]*np.log(2*np.pi))
    def ratio(a,b): return float(np.mean(logp(np.column_stack((a,b)),fj)-logp(a,fx)-logp(b,fy))/LOG2)
    return {'model':float(model),'train':ratio(x,y),'validation':ratio(*evals[0]),'heldout':ratio(*evals[1])}

def monomials(p,d):
    return [()] + [c for order in range(1,d+1) for c in itertools.combinations_with_replacement(range(p),order)]

def design(a,terms):
    out=np.ones((len(a),len(terms)))
    for k,t in enumerate(terms[1:],1):
        for idx in t: out[:,k]*=a[:,idx]
    return out

def sparse_poly(x,y,evals,degree,ridge=1.,floor=.1,px=2,py=2):
    # x and y are standardized with training means/scales before invocation.
    ycorr=np.abs(y.T@y); xcorr=np.abs(x.T@y); totals=[np.zeros(len(y))]+[np.zeros(len(t[0])) for t in evals]
    condition=[np.zeros_like(a) for a in totals]; marginal=[np.zeros_like(a) for a in totals]
    rows=[(x,y)]+evals
    for j in range(y.shape[1]):
        iy=np.argsort(ycorr[:j,j],kind='stable')[-min(py,j):] if j and py else np.array([],dtype=int)
        ix=np.argsort(xcorr[:,j],kind='stable')[-min(px,x.shape[1]):] if px else np.array([],dtype=int)
        for conditional,subset,out in ((False,(np.array([],dtype=int),iy),marginal),(True,(ix,iy),condition)):
            xi,yi=subset
            features=np.concatenate([x[:,xi],y[:,yi]],axis=1)
            means=features.mean(axis=0); scales=features.std(axis=0,ddof=1)
            scales=np.where(scales>1e-8,scales,1.)
            features=(features-means)/scales
            terms=monomials(features.shape[1],degree); A=design(features,terms)
            gram=A.T@A; penalty=np.eye(gram.shape[0])*ridge;penalty[0,0]=0
            beta=cho_solve(cho_factor(gram+penalty,lower=True,check_finite=False),A.T@y[:,j],check_finite=False)
            residual=y[:,j]-A@beta; scale=max(float(np.std(residual,ddof=1)),floor)
            for z,(xx,yy) in enumerate(rows):
                ae=design((np.concatenate([xx[:,xi],yy[:,yi]],axis=1)-means)/scales,terms)
                error=(yy[:,j]-ae@beta)/scale
                out[z]+=-.5*error**2-np.log(scale)-.5*np.log(2*np.pi)
    return {'train':float(np.mean(condition[0]-marginal[0])/LOG2), 'validation':float(np.mean(condition[1]-marginal[1])/LOG2), 'heldout':float(np.mean(condition[2]-marginal[2])/LOG2), 'degree':degree,'ridge':ridge,'scale_floor':floor,'source_predictors':px,'past_target_predictors':py}

def pca5(train,others):
    mu=train.mean(0); _,_,v=np.linalg.svd(train-mu,full_matrices=False); v=v[:5].T
    return (train-mu)@v,[(z-mu)@v for z in others]

def projected(x,y,evals,degree):
    x,xs=pca5(x,[v[0] for v in evals]); y,ys=pca5(y,[v[1] for v in evals]);
    opts={'degree':degree,'ridge':1.,'min_scale':.1,'joint_order':'source_first'}
    result={}
    for label,ex,ey in [('train',None,None),('validation',xs[0],ys[0]),('heldout',xs[1],ys[1])]:
        kw={} if ex is None else {'evaluation_x':ex,'evaluation_y':ey}
        v=estimate_mi(x,y,estimator='tm_polynomial',estimator_options=opts,unit='bits',**kw)
        result[label]=v.value
    return result

def append(path,row):
    with path.open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('data_sha256','details')}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--budget',type=int,default=1800);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);path=OUT/('smoke.jsonl' if a.smoke else 'results.jsonl');
    if path.exists():raise FileExistsError(path)
    start=time.monotonic(); deadline=start+a.budget
    append(path,{'event':'start','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'n_seeds':1 if a.smoke else 5,'budget':a.budget})
    dataset_types=['teacher','gaussian']; ns=[256] if a.smoke else [256,1199]; truths=[4] if a.smoke else [0,4]; seeds=range(1 if a.smoke else 5)
    cases=[(kind,n,truth,seed) for kind in dataset_types for n in ns for truth in truths for seed in seeds]
    # complete every method on a case before moving on; record immediately.
    for kind,n,truth,seed in cases:
        if time.monotonic()>=deadline: break
        x,y=prev.teacher(1000+seed,n,truth) if kind=='teacher' else gaussian(seed,n,truth)
        datahash=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        (fx,fy),(vx,vy),(tx,ty)=split(x,y)
        (fxs,fys),(vxs,vys),(txs,tys)=prev.normalize(fx,fy,(fx,fy),(vx,vy),(tx,ty))
        result={'event':'case','kind':kind,'n':n,'truth_bits':truth,'seed':seed,'data_sha256':datahash,'fit_n':len(fx),'validation_n':len(vx),'test_n':len(tx),'methods':{}}
        def run(name,func):
            t=time.monotonic()
            try:
                v=func(); result['methods'][name]={'result':v,'seconds':time.monotonic()-t}
            except Exception as e: result['methods'][name]={'error':repr(e),'seconds':time.monotonic()-t}
        for ridge in (.01,.1):
            run(f'affine_{ridge}',lambda ridge=ridge: gaussian_ratio(fxs,fys,[(vxs,vys),(txs,tys)],ridge))
        cx,cxs=copula(fx,[vx,tx]);cy,cys=copula(fy,[vy,ty])
        for ridge in (.01,.1):
            run(f'copula_{ridge}',lambda ridge=ridge: gaussian_ratio(cx,cy,[(cxs[0],cys[0]),(cxs[1],cys[1])],ridge))
        run('infonce',lambda: prev.neural_fit(x[:int(.8*n)],y[:int(.8*n)],tx,ty,32,0,epochs=8 if a.smoke else 100))
        for degree in (2,3):
            run(f'sparse_tm_d{degree}',lambda degree=degree:sparse_poly(fxs,fys,[(vxs,vys),(txs,tys)],degree))
            run(f'pca5_full_tm_d{degree}',lambda degree=degree:projected(fxs,fys,[(vxs,vys),(txs,tys)],degree))
        result['elapsed']=time.monotonic()-start;append(path,result)
    append(path,{'event':'end','elapsed_seconds':time.monotonic()-start,'budget_hit':time.monotonic()>=deadline,'cases_planned':len(cases)})
if __name__=='__main__':main()
