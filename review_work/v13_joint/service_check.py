from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'data'))
os.environ['NOTO_REVISION_OUTPUT']='v13_joint_20261107'
import run_v13_structure as r
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def job(task):
 curve,horizon,delayed,s=task
 r.w.load_data(r.INPUT)
 r.w.c1.update({j:curve for j in r.w.DEMAND})
 a=r.w.simulate_cell('split','soft',r.w.sample_scenario(np.random.default_rng(10000+s)),np.random.default_rng(20000+100*s+1),horizon=horizon,arrival_delay=delayed)
 return dict(curvature=curve,horizon=horizon,delayed=delayed,replicate=s,Y_min=a['Y_min'],psi_max=a['psi_max'],tc=a['tc'],tdc=a['tdc'])
if __name__=='__main__':
 jobs=[(c,h,d,s) for c in [.01,.02,.04] for h in [48,72] for d in [False,True] for s in range(256)]
 with ProcessPoolExecutor(max_workers=8) as pool: rows=list(pool.map(job,jobs,chunksize=8))
 pd.DataFrame(rows).to_csv(r.OUT/'service_robustness_replicates.csv',index=False)
 records=[]
 for (c,h,d),g in pd.DataFrame(rows).groupby(['curvature','horizon','delayed']):
  for threshold in [.15,.2,.25]:
   records.append(dict(curvature=c,horizon=h,delayed=bool(d),threshold=threshold,N=256,R_sys=float((g.Y_min>=.5).mean()),R_dc=float((g.psi_max<=threshold).mean()),wilson_dc=r.w._wilson(g.psi_max<=threshold)))
 (r.OUT/'service_robustness.json').write_text(json.dumps(records,indent=2))
 print('SERVICE COMPLETE')
