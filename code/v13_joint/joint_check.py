from pathlib import Path
import sys,os,json,hashlib,platform
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
os.environ['NOTO_REVISION_OUTPUT']='v13_joint_20261107'
import run_v13_structure as r
import numpy as np
import pandas as pd
from scipy.stats import beta

def interval(a,b):
 n=len(a); n10=int(np.sum(a&~b));n01=int(np.sum(~a&b))
 def cp(k):return (0. if k==0 else beta.ppf(.0125,k,n-k+1),1. if k==n else beta.ppf(.9875,k+1,n-k))
 l,h=cp(n10);ll,hh=cp(n01)
 return float(np.mean(a.astype(float)-b.astype(float))),float(l-hh),float(h-ll),n10,n01

if __name__=='__main__':
 r.prepare_inputs();r.w.load_data(r.INPUT);r.metadata()
 bundles={'reference':{},'low':dict(gamma_mult=.5,m_shift=-.5,beta_mult=.8,delta_mult=.5),
 'high':dict(gamma_mult=2.,m_shift=.5,beta_mult=1.2,delta_mult=2.),
 'mixed_A':dict(gamma_mult=2.,m_shift=-.5,beta_mult=.8,delta_mult=2.),
 'mixed_B':dict(gamma_mult=.5,m_shift=.5,beta_mult=1.2,delta_mult=.5)}
 configs={};cal={}
 for name,kw in bundles.items():
  lam,report=r.w.calibrate_exogenous_matched(N_cal=64,seed_cal=999,return_report=True,verbose=False,**kw)
  cal[name]=dict(N=64,seed=999,components=report)
  for cell,extra in [('M0',{}),('M1',dict(m='soft_mean')),('M3',dict(m='exogenous_matched',lambda_bar=lam)),('M4',dict(pi='frozen_split')),('fastest',dict(pi='fastest')),('throughput',dict(load_driver='throughput'))]:
   configs[name+'__'+cell]=dict(kw,**extra)
 (r.OUT/'calibration.json').write_text(json.dumps(cal,indent=2))
 data,sums=r.collect(configs,256,8,'joint')
 rows=[];residuals=[]
 for name in bundles:
  ref=data[name+'__M0']
  for cell in ['M1','M3','M4','fastest','throughput']:
   comp=data[name+'__'+cell]
   for endpoint in ['sys','dc']:
    def indicator(rs):return np.array([v['Y_min']>=.5 if endpoint=='sys' else v['psi_max']<=.2 for v in rs])
    delta,lo,hi,n10,n01=interval(indicator(comp),indicator(ref))
    rows.append(dict(bundle=name,cell=cell,endpoint=endpoint,delta=delta,lo=lo,hi=hi,n10=n10,n01=n01,N=256))
  for c in r.w.COMPONENTS:
   d=np.array([(b['H_initial'][c]-b['H_final'][c])-(a['H_initial'][c]-a['H_final'][c]) for a,b in zip(ref,data[name+'__M3'])])
   residuals.append(dict(bundle=name,component=c,mean_residual=float(d.mean()),paired_se=float(d.std(ddof=1)/16)))
 pd.DataFrame(rows).to_csv(r.OUT/'joint_contrasts.csv',index=False)
 pd.DataFrame(residuals).to_csv(r.OUT/'M3_evaluation.csv',index=False)
 print('JOINT COMPLETE',flush=True)
