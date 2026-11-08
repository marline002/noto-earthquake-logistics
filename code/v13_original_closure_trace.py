from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import v13_experiments as x
R=Path(__file__).resolve().parents[1];out=R/'result/v13_structure_20261107'
w=x.load();w.load_data(R/'data/v13_structure_20261107/inputs');rows=[]
for name,kw in [('M0',{}),('closure_stress',{'initial_closed':['WJ-SZ','SZ-WJ']})]:
 a=w.simulate_cell('split','soft',w.sample_scenario(np.random.default_rng(10000)),np.random.default_rng(20001),return_diagnostics=True,**kw)
 saved=pd.read_csv(out/'main_replicates.csv');q=saved[saved.config.eq(name)&saved.replicate.eq(0)].iloc[0]
 assert abs(q.Y_min-a['Y_min'])<1e-12 and abs(q.psi_max-a['psi_max'])<1e-12
 ds=[]
 for v in a['diagnostics']:
  ds.append({'epoch':v['epoch'],'capacities':v['capacities'],'attempted':v['attempted'],'plan':v['plan'],'requests':{j:[{'components':list(p),'amount':val} for p,val in qs.items()] for j,qs in v['requests'].items()}})
 (out/(name+'_diagnostic0.json')).write_text(json.dumps(ds,indent=2))
 np.savez_compressed(out/(name+'_diagnostic0.npz'),Y=a['Y'],psi=a['psi'],H=a['health_history'],driver=a['driver_history'])
 for i,c in enumerate(w.COMPONENTS):rows.append(dict(config=name,component=c,H_initial=a['H_initial'][c],H_final=a['H_final'][c],sum_driver=a['driver_history'][:,i].sum(),Y_min=a['Y_min']))
pd.DataFrame(rows).to_csv(out/'closure_diagnostic_summary.csv',index=False)
files=[Path(__file__)]+list(out.glob('*diagnostic*'))
p=out/'run_manifest.json';m=json.loads(p.read_text());m['files'].update({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files});p.write_text(json.dumps(m,indent=2))
print(pd.DataFrame(rows).query('component == "WJ"').to_string(index=False))
