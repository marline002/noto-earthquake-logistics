from pathlib import Path
import os,sys,json,hashlib,importlib.util
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data'/'v13_20261107'; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'data'))
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import beta
original=ROOT/'data'/'v13_core.py'
model=OUT/'v13_model.py'
def prepare():
 s=original.read_text(encoding='utf-8')
 s=s.replace('shared_roads=False, initial_closed=()):','shared_roads=False, initial_closed=(), initial_stock_hours=0., retry_hours=None):')
 s=s.replace('history_H, history_A = [], []','''history_H, history_A = [], []
    initial_stock = {d: arrivals[d]/delta_t_D*initial_stock_hours for d in DEMAND}
    for d in DEMAND:
        transit[d][0] = initial_stock[d]
    dispatched_total=0.
    served_total=0.''')
 s=s.replace('if load_driver == "throughput":','''if retry_hours is not None:
                if retry_hours <= 0: raise ValueError('positive retry clock required')
                driver = {c: 0. for c in COMPONENTS}
                for j, paths in Q.items():
                    B = sum(g for g,age in routing_batches[j])
                    D = DEMAND_D[j]*scale*demand_mult
                    factor = (D+B*Delta_t_R/retry_hours)/(D+B) if D+B>0 else 0.
                    for path,amount in paths.items():
                        for c in path: driver[c]+=amount*factor
            if load_driver == "throughput":''')
 # Q nesting checked against route_and_plan below; empty baseline loads in this experiment.
 s=s.replace('transit[d][due] = transit[d].get(due, 0.)+flow*delta_t_D/Delta_t_R','''transit[d][due] = transit[d].get(due, 0.)+flow*delta_t_D/Delta_t_R
                        dispatched_total += flow*delta_t_D/Delta_t_R''')
 s=s.replace('delivered[d] = value','delivered[d] = value\n                served_total += value')
 s=s.replace('"degradation_updates": nsub if mode != "binary" else 0}', '''"initial_stock": initial_stock, "dispatched_total": dispatched_total, "served_total": served_total,
            "degradation_updates": nsub if mode != "binary" else 0}''')
 model.write_text(s,encoding='utf-8')
 # Geometric lower bounds are an explicitly changed scenario, not original measured routes.
 import run_v13_structure as r
 os.environ['NOTO_REVISION_MODE']='geometry'
 r.OUT=OUT; r.INPUT=OUT/'inputs';r.prepare_inputs()

def load():
 spec=importlib.util.spec_from_file_location('v13_model',model);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w);w.load_data(OUT/'inputs');return w
def job(task):
 label,s,kw=task;w=load();kw=dict(kw);dt=kw.get('Delta_t_R',3.)
 stock=sum(w.DEMAND_D.values())/3*12
 # Prepositioned stock is debited from the first source interval budget.
 per_source=stock/len(w.SUPPLY)
 def budgets(r):return {i:w.NODE_CAP[i]*dt/3-(per_source*dt/12 if r*dt<12 else 0) for i in w.SUPPLY}
 a=w.simulate_cell(kw.pop('pi','split'),kw.pop('m','soft'),w.sample_scenario(np.random.default_rng(10000+s)),np.random.default_rng(20000+100*s+1),arrival_delay=True,initial_stock_hours=12,supply_budgets=budgets,return_diagnostics=(s==0),**kw)
 # Initial stock + dispatched mass balances served mass + remaining stock/backlog pipeline.
 residual=sum(a['initial_stock'].values())+a['dispatched_total']-a['served_total']-sum(a['in_transit_final'].values())
 assert abs(residual)<1e-7,(label,s,residual)
 if s==0:np.savez_compressed(OUT/(label+'_trace.npz'),Y=a['Y'],psi=a['psi'],H=a['health_history'],driver=a['driver_history'])
 return dict(config=label,replicate=s,Y_min=a['Y_min'],psi_max=a['psi_max'],tc=a['tc'],tdc=a['tdc'],stock=sum(a['initial_stock'].values()),dispatched=a['dispatched_total'],served=a['served_total'],remaining=sum(a['in_transit_final'].values()),balance_error=residual)
def interval(a,b):
 n=len(a);p=int(np.sum(a&~b));q=int(np.sum(~a&b))
 def bounds(x):return (0 if x==0 else beta.ppf(.0125,x,n-x+1),1 if x==n else beta.ppf(.9875,x+1,n-x))
 l,u=bounds(p);v,z=bounds(q);return (p-q)/n,l-z,u-v,p,q
if __name__=='__main__':
 prepare()
 # Q is keyed by (origin,destination), values map node paths to requested amounts.
 configs={'request':{},'throughput':{'load_driver':'throughput'},'mean':{'m':'soft_mean'},'static':{'m':'static'},'frozen':{'pi':'frozen_split'},'fastest':{'pi':'fastest'},'clock3':{'retry_hours':3.},'request_dt1':{'Delta_t_R':1.},'request_dt6':{'Delta_t_R':6.},'clock3_dt1':{'Delta_t_R':1.,'retry_hours':3.},'clock3_dt6':{'Delta_t_R':6.,'retry_hours':3.},'closed':{'initial_closed':('WJ-SZ','SZ-WJ')}}
 # Pilot every branch before dispatching the batch.
 for label,kw in configs.items():job((label,0,kw))
 print('PILOT PASS',flush=True)
 with ProcessPoolExecutor(max_workers=8) as pool: rows=list(pool.map(job,[(k,s,v) for k,v in configs.items() for s in range(256)],chunksize=8))
 df=pd.DataFrame(rows);df.to_csv(OUT/'mission_replicates.csv',index=False)
 summaries=[];contrasts=[];ref=df[df.config=='request'].sort_values('replicate')
 for k,g in df.groupby('config'):
  g=g.sort_values('replicate');row=dict(config=k,N=len(g))
  for endpoint,col,op in [('sys','Y_min',lambda x:x>=.5),('dc','psi_max',lambda x:x<=.2)]:
   a=op(g[col].to_numpy());b=op(ref[col].to_numpy());row['R_'+endpoint]=float(a.mean());row['n_'+endpoint]=int(a.sum())
   delta,l,u,p,q=interval(a,b);contrasts.append(dict(config=k,endpoint=endpoint,delta=delta,lo=l,hi=u,n10=p,n01=q,N=len(g)))
  summaries.append(row)
 pd.DataFrame(summaries).to_csv(OUT/'mission_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/'mission_contrasts.csv',index=False)
 (OUT/'mission_configs.json').write_text(json.dumps(configs,indent=2))
 files=[Path(__file__),original]+list(OUT.glob('*.py'))+list(OUT.glob('*.csv'))+list(OUT.glob('*.json'))+list((OUT/'inputs').glob('*'))+list(OUT.glob('*.npz'))
 (OUT/'manifest.json').write_text(json.dumps({'command':'python review_work/v13_experiments.py','N':256,'initial_stock_hours':12,'stock_source_debit':'proportional debit from first 12 hours of nominal source budgets','scenario':'geometric lower-bound link lengths and times','seed_scenario':'10000+s','seed_process':'20000+100*s+1','files':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}},indent=2))
 print(pd.DataFrame(summaries).to_string(index=False),flush=True)
