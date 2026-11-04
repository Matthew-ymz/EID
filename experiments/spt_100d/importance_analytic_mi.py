"""Population MI and fixed-evaluation Monte Carlo error for the independent 100D control."""
import json,math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
R=Path(__file__).resolve().parents[2];D=R/'docs/log/spt_100d_stress/results';sigma=.086;bound=math.pi
p=lambda y:(ndtr((bound-y)/sigma)-ndtr((-bound-y)/sigma))/(2*bound)
def f(y):
 v=p(y);return -v*math.log(v) if v>0 else 0.
h,err=quad(f,-bound-1,bound+1,epsabs=1e-11,limit=200);population=100*(h-.5*math.log(2*math.pi*math.e*sigma*sigma));rng=np.random.default_rng(808);x=rng.uniform(-bound,bound,(32,100));y=x+rng.normal(size=(32,100))*sigma
cond=np.sum(-.5*((y-x)/sigma)**2-math.log(sigma*math.sqrt(2*math.pi)),axis=1);denom=np.sum(np.log(np.maximum(ndtr((bound-y)/sigma)-ndtr((-bound-y)/sigma),1e-300))-math.log(2*bound),axis=1);v=cond-denom
out={'n':100,'sigma':sigma,'population_mi_nats':population,'quadrature_error_bound_nats':100*err,'evaluation_count':32,'evaluation_mean_exact_pointwise_mi_nats':float(v.mean()),'evaluation_sem_nats':float(v.std(ddof=1)/np.sqrt(len(v))),'note':'quadrature omits Gaussian tails outside [-pi-1,pi+1], negligible at sigma .086; sample SEM is conditional on 32 independent outer draws'}
(D/'importance_analytic_mi.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
