"""Additional post-main regularization sensitivity; no test-based selection."""
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
spec=importlib.util.spec_from_file_location('mi_benchmark_impl',Path(__file__).with_name('run.py'))
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
P=Path(__file__).parent/'results'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--budget',type=int,default=1200);a=ap.parse_args()
    path=P/'sensitivity.jsonl'
    if path.exists():raise FileExistsError(path)
    start=time.monotonic();end=start+a.budget
    cases=[r for r in map(json.loads,(P/'results.jsonl').read_text().splitlines()) if r['event']=='case']
    b.append(path,{'event':'start','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_cases':len(cases),'budget':a.budget})
    for case in cases:
        if time.monotonic()>=end: break
        kind,n,truth,seed=(case[k] for k in ('kind','n','truth_bits','seed'))
        x,y=b.prev.teacher(1000+seed,n,truth) if kind=='teacher' else b.gaussian(seed,n,truth)
        datahash=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        if datahash!=case['data_sha256']:raise ValueError('data hash mismatch')
        (fx,fy),(vx,vy),(tx,ty)=b.split(x,y)
        (fxs,fys),(vxs,vys),(txs,tys)=b.prev.normalize(fx,fy,(fx,fy),(vx,vy),(tx,ty))
        cx,cxs=b.copula(fx,[vx,tx]);cy,cys=b.copula(fy,[vy,ty])
        result={'event':'sensitivity','kind':kind,'n':n,'truth_bits':truth,'seed':seed,'data_sha256':datahash,'methods':{}}
        def call(name,fun):
            t=time.monotonic()
            try: result['methods'][name]={'result':fun(),'seconds':time.monotonic()-t}
            except Exception as e:result['methods'][name]={'error':repr(e),'seconds':time.monotonic()-t}
        for ridge in (1.,10.):
            call(f'affine_{ridge}',lambda ridge=ridge:b.gaussian_ratio(fxs,fys,[(vxs,vys),(txs,tys)],ridge))
            call(f'copula_{ridge}',lambda ridge=ridge:b.gaussian_ratio(cx,cy,[(cxs[0],cys[0]),(cxs[1],cys[1])],ridge))
        for degree in (2,3):
            for ridge in (10.,100.):
                for floor in (.1,.5):
                    call(f'sparse_d{degree}_r{ridge}_f{floor}',lambda degree=degree,ridge=ridge,floor=floor:b.sparse_poly(fxs,fys,[(vxs,vys),(txs,tys)],degree,ridge=ridge,floor=floor))
        result['elapsed']=time.monotonic()-start;b.append(path,result)
    b.append(path,{'event':'end','elapsed_seconds':time.monotonic()-start,'budget_hit':time.monotonic()>=end})
if __name__=='__main__':main()
