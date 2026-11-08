"""Same-task per-epoch LP benchmark; not a horizon-optimal control policy."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,sys
from scipy.optimize import linprog
from concurrent.futures import ProcessPoolExecutor
import v13_experiments as x
R=Path(__file__).resolve().parents[1];OUT=R/'result/v13_lp_20261107';OUT.mkdir(exist_ok=True)
def job(args):
 task,s,policy=args;w=x.load()
 if task=='immediate':w.load_data(R/'data/v13_joint_20261107/inputs')
 base=w.route_and_plan;gaps=[]
 def route(pi,m,C,batches,dt,dm,Z,r=0,prev_attempted=None,supply_budgets=None,baseline_loads=None,path_pools=None):
  attempted,plan,Q=base(pi,m,C,batches,dt,dm,Z,r,prev_attempted,supply_budgets,baseline_loads,path_pools)
  paths=[(j,p) for j,qs in Q.items() for p,q in qs.items() if all(C[c]>0 for c in p)]
  heuristic=sum(f for a in plan.values() for p,f in a)
  if not paths:
   assert heuristic<1e-7;gaps.append([r,heuristic,0.,0.]);return attempted,plan,Q
  A=[];b=[]
  for c in w.COMPONENTS:
   A.append([float(c in p) for j,p in paths]);b.append(C[c])
  for j in w.DEMAND:
   A.append([float(j==k) for k,p in paths]);b.append(sum(Q[j].values()))
  for i in w.SUPPLY:
   A.append([float(p[0]==i) for j,p in paths]);b.append((supply_budgets or {}).get(i,w.C0_DICT[i]*dt/3))
  A=np.array(A);b=np.array(b)
  hv={(j,tuple(p)):f for j,plans in plan.items() for p,f in plans}
  assert set(hv)<=set(paths),'Heuristic support outside LP candidate set'
  h=np.array([hv.get(k,0.) for k in paths]);assert (A@h-b).max()<1e-6
  assert abs(h.sum()-heuristic)<1e-7
  opt=linprog(-np.ones(len(paths)),A_ub=A,b_ub=b,bounds=(0,None),method='highs')
  assert opt.success,opt.message
  vals=np.maximum(opt.x,0.);assert (A@vals-b).max()<1e-6
  total=vals.sum();assert total+1e-6>=heuristic,(total,heuristic)
  gaps.append([r,heuristic,total,(total-heuristic)/total if total>1e-10 else 0.])
  if policy=='lp':
   plan={j:[] for j in w.DEMAND}
   for (j,p),f in zip(paths,vals):
    if f>0:plan[j].append((list(p),float(f)))
  return attempted,plan,Q
 w.route_and_plan=route
 kw={}
 if task=='mission':
  stock=sum(w.DEMAND_D.values())/3*12
  def budgets(r):return {i:w.NODE_CAP[i]-(stock/len(w.SUPPLY)/4 if r<4 else 0) for i in w.SUPPLY}
  kw=dict(arrival_delay=True,initial_stock_hours=12,supply_budgets=budgets)
 a=w.simulate_cell('split','soft',w.sample_scenario(np.random.default_rng(10000+s)),np.random.default_rng(20000+100*s+1),**kw)
 if task=='mission':assert abs(sum(a['initial_stock'].values())+a['dispatched_total']-a['served_total']-sum(a['in_transit_final'].values()))<1e-7
 return dict(task=task,policy=policy,replicate=s,Y_min=a['Y_min'],psi_max=a['psi_max'],tc=a['tc'],tdc=a['tdc']),[dict(task=task,policy=policy,replicate=s,epoch=r,heuristic=h,optimum=o,relative_gap=g) for r,h,o,g in gaps]
if __name__=='__main__':
 rows=[];gaps=[]
 with ProcessPoolExecutor(max_workers=8) as pool:
  for i,(row,g) in enumerate(pool.map(job,[(task,s,p) for task in ['immediate','mission'] for p in ['heuristic','lp'] for s in range(256)],chunksize=4)):
   rows.append(row);gaps.extend(g)
   if i%128==127:print(i+1,'/1024',flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'replicates.csv',index=False);pd.DataFrame(gaps).to_csv(OUT/'epoch_gaps.csv',index=False)
 sums=[]
 for task in ['immediate','mission']:
  a=df[(df.task==task)&(df.policy=='lp')].sort_values('replicate');b=df[(df.task==task)&(df.policy=='heuristic')].sort_values('replicate')
  for ep,col,op in [('sys','Y_min',lambda z:z>=.5),('dc','psi_max',lambda z:z<=.2)]:
   av=op(a[col].to_numpy());bv=op(b[col].to_numpy());d,l,u,p,q=x.interval(av,bv);sums.append(dict(task=task,endpoint=ep,R_LP=av.mean(),R_heuristic=bv.mean(),delta=d,lo=l,hi=u,n10=p,n01=q))
 pd.DataFrame(sums).to_csv(OUT/'summary.csv',index=False)
 files=[Path(__file__),R/'code/v13_model.py']+list(OUT.glob('*.csv'))
 (OUT/'manifest.json').write_text(json.dumps({'N':256,'objective':'maximize planned interval departures over all capacity-positive paths in current request ledger, including zero request weights; component, destination-request and source-budget bounds; retain original requested-load driver','heuristic_feasibility_check':'Every epoch checks heuristic support belongs to LP paths and every LP inequality on heuristic path-flow vector, in addition to LP feasibility and objective dominance','files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
 print(pd.DataFrame(sums).to_string(index=False),flush=True)
