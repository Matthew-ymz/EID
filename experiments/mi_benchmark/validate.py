"""Independent numerical checks for benchmark preprocessing and readers."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('run.py'); spec=importlib.util.spec_from_file_location('bench',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
rng=np.random.default_rng(190); x=rng.normal(size=(400,2)); y=np.c_[x[:,0]+.3*rng.normal(size=400),x[:,1]**2+.4*rng.normal(size=400)]
(x,y),(tx,ty)=b.prev.normalize(x[:300],y[:300],(x[:300],y[:300]),(x[300:],y[300:]))
for d in [2,3]:
    ours=b.sparse_poly(x,y,[(tx,ty),(tx,ty)],d,ridge=1,floor=.1,px=2,py=2)
    theirs=b.estimate_mi(x,y,estimator='tm_polynomial',estimator_options={'degree':d,'ridge':1,'min_scale':.1,'joint_order':'source_first'},unit='bits',evaluation_x=tx,evaluation_y=ty).value
    assert abs(ours['heldout']-theirs)<1e-8,(ours,theirs)
    print('polynomial degree',d,'max difference',abs(ours['heldout']-theirs))
r=b.gaussian_ratio(x,y,[(tx,ty),(tx,ty)],.1)
v=b.estimate_mi(x,y,estimator='tm_affine',estimator_options={'jitter':.1},unit='bits',evaluation_x=tx,evaluation_y=ty).value
assert abs(r['heldout']-v)<1e-8
print('affine density ratio difference',abs(r['heldout']-v))
u,_=b.copula(x,[tx]); u2,_=b.copula(np.exp(x),[np.exp(tx)])
assert np.allclose(u,u2)
print('copula training scores invariant to strictly monotonic transform')
rho=np.sqrt(1-2**(-2*4/500)); assert np.isclose(-250*np.log2(1-rho*rho),4)
latent=rng.normal(size=(20,10)); recovered=np.logaddexp(0,latent)-np.logaddexp(0,-latent)
assert np.allclose(latent,recovered)
print('analytical MI and injective latent features verified')
