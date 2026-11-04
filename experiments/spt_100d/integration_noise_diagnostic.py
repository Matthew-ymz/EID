"""Common-channel neural TM marginal integration sensitivity, no structural labels."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import importlib.util,json,math,sys,time,signal
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[2];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('pilot2',P.with_name('nonlinear_pilot.py'));pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
from eid import UniformBox
from eid.estimators.neural import NeuralEIOracle
import torch

def main():
 torch.set_num_threads(1);start=time.monotonic();out=ROOT/'docs/log/spt_100d_stress/results/integration_noise_diagnostic.jsonl';cfg=pilot.base.mk_system(6,101,0);cfg['sigma']=0.8;x,y=pilot.base.data(cfg,4096,10101)
 with out.open('w') as f:
  for draws in (128,1024,8192,16384):
   t=time.monotonic()
   try:
    o=NeuralEIOracle(x[:3276],y[:3276],UniformBox([[-math.pi,math.pi]]*6),[[i] for i in range(6)],evaluation_count=64,marginal_samples=draws,integration_batch_size=4096,seed=101,context_transform=pilot.context,epochs=100,hidden_features=64,layers=2,batch_size=256,threads=1,context_selection=False)
    all6=o.ei(tuple(range(6)));left=o.ei((0,1,2));right=o.ei((3,4,5));syn=all6-left-right
    row={'sigma':0.8,'draws':draws,'evaluation_count':64,'ei_all_nats':all6,'ei_left_nats':left,'ei_right_nats':right,'syn_nats':syn,'tol_nats':0.06931471805599453,'valid':syn>=-0.06931471805599453,'elapsed_seconds':time.monotonic()-t,'total_seconds':time.monotonic()-start}
   except Exception as e:row={'sigma':0.8,'draws':draws,'error':repr(e),'total_seconds':time.monotonic()-start}
   f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
if __name__=='__main__':main()
