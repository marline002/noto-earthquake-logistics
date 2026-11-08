"""Check the V13 data/code/result package without manuscript files."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
counts={}
for name in ['data/v13_structure_20261107/run_manifest.json','data/v13_lp_20261107/manifest.json','data/v13_20261107/manifest.json','data/v13_joint_20261107/run_manifest.json']:
 m=json.loads((R/name).read_text(encoding='utf-8'));items=m.get('files',m.get('sha256'))
 for path,digest in items.items():
  target=R/path.replace('\\','/')
  assert target.is_file() and sha(target)==digest,(name,path)
 counts[name]=len(items)
D=R/'data/v13_structure_20261107';frames={};ordinary=0
for stage in ['main','sweeps','robust']:
 df=pd.read_csv(D/(stage+'_replicates.csv'));frames[stage]=df;ordinary+=len(df)
 summary=json.loads((D/(stage+'_summary.json')).read_text(encoding='utf-8'))
 for key,g in df.groupby('config'):
  assert len(g)==256
  assert abs((g.Y_min>=.5).mean()-summary[key]['R_sys'])<1e-12
  assert abs((g.psi_max<=.2).mean()-summary[key]['R_dc'])<1e-12
  assert ((g.Y_min<.5)==(g.tc<=72)).all()
  assert ((g.psi_max>.2)==(g.tdc<=72)).all()
ref=frames['main'][frames['main'].config.eq('M0')].sort_values('replicate')
for key in ['R_3','D_1.0','CV_0.25','demand_1.0','K_6','gamma_mult_1.0','m_shift_0','beta_mult_1.0','delta_mult_1.0']:
 z=frames['sweeps'][frames['sweeps'].config.eq(key)].sort_values('replicate')
 assert np.array_equal(z[['Y_min','psi_max','tc','tdc']].to_numpy(),ref[['Y_min','psi_max','tc','tdc']].to_numpy()),key
j=pd.read_csv(R/'data/v13_joint_20261107/joint_replicates.csv')
for label in ['M0','M1','M3','M4','soft_fastest','throughput']:
 channel='fastest' if label=='soft_fastest' else label
 key=next(k for k in j.config.unique() if k.startswith('reference') and k.endswith(channel))
 z=j[j.config.eq(key)].sort_values('replicate');q=frames['main'][frames['main'].config.eq(label)].sort_values('replicate')
 assert np.allclose(z[['Y_min','psi_max','tc','tdc']],q[['Y_min','psi_max','tc','tdc']],atol=1e-12,rtol=0),label
mission=pd.read_csv(R/'data/v13_20261107/mission_replicates.csv')
assert len(mission)==3072 and mission.balance_error.abs().max()<1e-7
lp=pd.read_csv(R/'data/v13_lp_20261107/replicates.csv')
for task,baseline in [('immediate',ref),('mission',mission[mission.config.eq('request')].sort_values('replicate'))]:
 z=lp[lp.task.eq(task)&lp.policy.eq('heuristic')].sort_values('replicate')
 assert np.allclose(z[['Y_min','psi_max','tc','tdc']],baseline[['Y_min','psi_max','tc','tdc']],atol=1e-12,rtol=0)
g=pd.read_csv(R/'data/v13_lp_20261107/epoch_gaps.csv')
assert (g.optimum+1e-6>=g.heuristic).all()
raw=np.load(D/'gsa_raw.npz');assert raw['indicators'].shape==(2,192,128,2)
print(json.dumps({'manifest_file_counts':counts,'ordinary_evaluations':ordinary,'Sobol_evaluations':49152,'LP_evaluations':len(lp),'mission_evaluations':len(mission),'all_checks_passed':True},indent=2))
