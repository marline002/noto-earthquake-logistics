"""Audited revision experiments. Run --stage main, sweeps, robust, gsa, then figures.

All output arrays, including trajectories and calibration residuals, are saved.
Original input CSVs are immutable. Corrected inputs are scenario assumptions.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, hashlib, importlib.util, json, math, platform, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc, norm

BASE=Path(__file__).resolve().parent
DATA=BASE.parent/'data'/'noto_data'
OUT=BASE.parent/'result'/os.environ.get('NOTO_REVISION_OUTPUT','v13_structure_20261107')
INPUT=BASE.parent/'data'/OUT.name/'inputs'
MODEL=BASE/'v13_core.py'
spec=importlib.util.spec_from_file_location('noto_revision',MODEL)
w=importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

def prepare_inputs():
    INPUT.mkdir(parents=True,exist_ok=True)
    nodes=pd.read_csv(DATA/'nodes_noto.csv')
    links=pd.read_csv(DATA/'links_noto.csv')
    pop=pd.read_csv(DATA/'population_demand_noto.csv')
    pos=nodes.set_index('node')[['lon','lat']].to_dict('index')
    audit=[]
    for idx,row in links.iterrows():
        p,q=pos[row.u],pos[row.v]
        x,y=np.radians([p['lat'],q['lat']]); dl=np.radians(q['lon']-p['lon'])
        dist=6371.0088*2*math.asin(math.sqrt(math.sin((y-x)/2)**2+math.cos(x)*math.cos(y)*math.sin(dl/2)**2))
        length=max(row.length_km,dist)
        links.loc[idx,'length_km']=length
        links.loc[idx,'travel_time_h']=row.travel_time_h*length/row.length_km
        audit.append(dict(u=row.u,v=row.v,original_length_km=row.length_km,
                          great_circle_km=dist,scenario_length_km=length,
                          original_time_h=row.travel_time_h,scenario_time_h=links.loc[idx,'travel_time_h']))
    if os.environ.get('NOTO_REVISION_MODE','csv') == 'csv':
        links=pd.read_csv(DATA/'links_noto.csv')
    nodes.to_csv(INPUT/'nodes_noto.csv',index=False)
    links.to_csv(INPUT/'links_noto.csv',index=False)
    pop.to_csv(INPUT/'population_demand_noto.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'geometry_audit.csv',index=False)
    pop['implied_kg_per_person_per_day']=pop.demand_t_per_3h*1000*8/pop.population_2023
    pop['derivation_status']='reverse calculation from supplied tonnage; no evacuee conversion record'
    pop.to_csv(OUT/'demand_audit.csv',index=False)

def init_worker():
    w.load_data(INPUT)

def task(job):
    label,s,kw=job
    kw=dict(kw)
    beta=kw.pop('beta_mult',1.)
    source=kw.pop('original_inputs',False)
    gsa_x=kw.pop('gsa_x',None)
    k=kw.pop('K',6)
    if w.K != k:
        w.set_K(k)
    scen_seed,proc_seed=10000+s,20000+100*s+1
    if gsa_x is not None:
        batch,point,rep,g,m,b,d=gsa_x
        scen_seed=np.random.SeedSequence([20261029,batch,point,rep,101])
        proc_seed=np.random.SeedSequence([20261029,batch,point,rep,202])
        beta=b;kw.update(gamma_mult=g,m_shift=m,delta_mult=d)
    if source:
        w.load_data(DATA)
    result=w.simulate_cell(kw.pop('pi','split'),kw.pop('m','soft'),
        w.sample_scenario(np.random.default_rng(scen_seed),beta),
        np.random.default_rng(proc_seed),**kw)
    if source:
        w.load_data(INPUT)
    # Preserve source numbers for all figures and timing analyses.
    return label,s,{k:result[k] for k in ('Y_min','psi_max','tc','tdc','Y','psi','H_initial','H_final','health_history','driver_history')}

def summary(rs):
    y=np.array([r['Y_min'] for r in rs]);p=np.array([r['psi_max'] for r in rs])
    out=dict(N=len(rs),R_sys=float((y>=w.Y_CRIT).mean()),R_dc=float((p<=w.PSI_CRIT).mean()),
             n_sys=int((y>=w.Y_CRIT).sum()),n_dc=int((p<=w.PSI_CRIT).sum()),
             wilson_R=w._wilson(y>=w.Y_CRIT),wilson_Rdc=w._wilson(p<=w.PSI_CRIT),
             q05=float(np.quantile(y,.05)),q95=float(np.quantile(y,.95)))
    for field,fail in [('tc',y<w.Y_CRIT),('tdc',p>w.PSI_CRIT)]:
        t=np.array([r[field] for r in rs]);cross=t<=w.T
        assert np.array_equal(cross,fail), (field, np.where(cross!=fail))
        out['n_'+field]=int(cross.sum())
        out['mean_'+field+'_cond']=float(t[cross].mean()) if cross.any() else None
        out['mean_'+field+'_sentinel']=float(t.mean())
        out['n_'+field+'_at_T']=int((t==w.T).sum())
    return out

def collect(configs,N,workers,filename):
    jobs=[(label,s,kw) for label,kw in configs.items() for s in range(N)]
    data={label:[None]*N for label in configs}
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for i,(label,s,r) in enumerate(pool.map(task,jobs,chunksize=8)):
            data[label][s]=r
            if (i+1)%1000==0:
                print(filename,i+1,'/',len(jobs),'elapsed',round(time.time()-start,1),flush=True)
    arrays={}
    rows=[]
    sums={}
    for label,rs in data.items():
        sums[label]=summary(rs)
        for field in ('Y','psi','health_history','driver_history'):
            if np.asarray(rs[0][field]).size:
                arrays[label+'__'+field]=np.asarray([r[field] for r in rs])
        for s,r in enumerate(rs):
            row={k:r[k] for k in ('Y_min','psi_max','tc','tdc')}
            rows.append(dict(config=label,replicate=s,**row))
            for field in ('H_initial','H_final'):
                for c in w.COMPONENTS:
                    arrays.setdefault(label+'__'+field,[])
                # outside per-component loop, preserve sorted inventory
                arrays[label+'__'+field].append([r[field][c] for c in w.COMPONENTS])
    np.savez_compressed(OUT/(filename+'_arrays.npz'),**arrays)
    pd.DataFrame(rows).to_csv(OUT/(filename+'_replicates.csv'),index=False)
    (OUT/(filename+'_summary.json')).write_text(json.dumps(sums,indent=2),encoding='utf-8')
    (OUT/(filename+'_configs.json')).write_text(json.dumps(configs,indent=2),encoding='utf-8')
    return data,sums

def main_stage(workers,N):
    w.load_data(INPUT)
    lam,report=w.calibrate_exogenous_matched(N_cal=64,seed_cal=999,return_report=True)
    (OUT/'M3_calibration.json').write_text(json.dumps(dict(N=64,seed=999,components=report),indent=2))
    configs={
        'M0':{},'M1':dict(m='soft_mean'),'M2':dict(m='static'),
        'M3':dict(m='exogenous_matched',lambda_bar=lam),
        'M4':dict(pi='frozen_split'),'M5':dict(m='link_only'),'M6':dict(m='node_only'),
        'soft_fastest':dict(pi='fastest'), 'binary_split':dict(m='binary'),
        'binary_fastest':dict(pi='fastest',m='binary'),
        'throughput':dict(load_driver='throughput'),
        'cap_only':dict(delivery_rule='cap'),
        'delayed':dict(arrival_delay=True),
        'combined':dict(load_driver='throughput',delivery_rule='cap',arrival_delay=True),
        'shared_road':dict(shared_roads=True),
        'static_cap':dict(m='static',delivery_rule='cap'),
        'static_cap_fastest':dict(pi='fastest',m='static',delivery_rule='cap'),
        'closure_stress':dict(initial_closed=['WJ-SZ','SZ-WJ']),
        'legacy_M0':dict(original_inputs=True)}
    data,sums=collect(configs,N,workers,'main')
    # Auditable componentwise independent evaluation matching residuals.
    evalrows=[]
    for c in w.COMPONENTS:
        a=np.array([r['H_initial'][c]-r['H_final'][c] for r in data['M0']])
        b=np.array([r['H_initial'][c]-r['H_final'][c] for r in data['M3']])
        diff=b-a
        evalrows.append(dict(component=c,lambda_per_h=lam[c],calibration_target=report[c]['target_health_loss'],
            calibration_residual=report[c]['residual'],M0_eval_loss=float(a.mean()),M3_eval_loss=float(b.mean()),
            eval_residual=float(diff.mean()),eval_paired_se=float(diff.std(ddof=1)/np.sqrt(N))))
    pd.DataFrame(evalrows).to_csv(OUT/'M3_independent_evaluation.csv',index=False)
    # Concordance plus conservative finite-sample-valid intervals.
    rec={label:dict(Y_min=np.array([r['Y_min'] for r in rs]),psi_max=np.array([r['psi_max'] for r in rs]),replicate_id=np.arange(N)) for label,rs in data.items()}
    paired={label:w.paired_comparison(rec,label,'M0') for label in configs if label!='M0'}
    (OUT/'paired.json').write_text(json.dumps(paired,indent=2))
    did={}
    for endpoint in ('sys','dc'):
        key='Y_min' if endpoint=='sys' else 'psi_max'
        indicators={label:(r[key]>=w.Y_CRIT if endpoint=='sys' else r[key]<=w.PSI_CRIT).astype(float) for label,r in rec.items()}
        z=(indicators['M0']-indicators['binary_split'])-(indicators['soft_fastest']-indicators['binary_fastest'])
        mean,se,lo,hi=w._paired_interval(z)
        did[endpoint]=dict(mean=mean,se=se,lo=lo,hi=hi)
    (OUT/'did.json').write_text(json.dumps(did,indent=2))
    print(json.dumps(sums,indent=2),flush=True)

def sweep_stage(workers,N):
    configs={}
    for dt in (1,2,3,4,6): configs['R_'+str(dt)]=dict(Delta_t_R=dt)
    for dt in (3.,1.,.5,.25,.125): configs['D_'+str(dt)]=dict(delta_t_D=dt)
    for cv in (0,.05,.10,.15,.20,.25,.30,.40,.50): configs['CV_'+str(cv)]=dict(cv=cv)
    for d in (.5,.75,1.,1.25,1.5,1.75,2.): configs['demand_'+str(d)]=dict(demand_mult=d)
    for k in (2,4,6,8,10): configs['K_'+str(k)]=dict(K=k)
    for name,values in [('gamma_mult',(.5,1.,2.)),('m_shift',(-.5,0,.5)),('beta_mult',(.8,1.,1.2)),('delta_mult',(.5,1.,2.))]:
        for v in values: configs[name+'_'+str(v)]=dict(**{name:v})
    collect(configs,N,workers,'sweeps')

def robust_stage(workers,N):
    configs={}
    for label,kw in [('throughput',dict(load_driver='throughput')),
                     ('cap_only',dict(delivery_rule='cap')),
                     ('delayed',dict(arrival_delay=True)),
                     ('combined',dict(load_driver='throughput',delivery_rule='cap',arrival_delay=True))]:
        for channel,extra in [('M4',dict(pi='frozen_split')),('M5',dict(m='link_only')),
                              ('M6',dict(m='node_only')),('fastest',dict(pi='fastest'))]:
            configs[label+'_'+channel]=dict(**kw,**extra)
    configs['trace_M0']=dict(return_diagnostics=True)
    collect(configs,N,workers,'robust')

def gsa_stage(workers,base=32,budgets=(32,128)):
    bounds=np.array([[.5,2.],[-.5,.5],[.8,1.2],[.5,2.]])
    Xs=[];jobs=[]
    for batch in range(2):
        ab=qmc.Sobol(8,scramble=True,seed=77+batch).random_base2(int(math.log2(base)))
        a=bounds[:,0]+ab[:,:4]*(bounds[:,1]-bounds[:,0])
        b=bounds[:,0]+ab[:,4:]*(bounds[:,1]-bounds[:,0])
        x=[]
        for i in range(base):
            x.extend([a[i],b[i]])
            for j in range(4):
                v=a[i].copy();v[j]=b[i,j];x.append(v)
        Xs.append(np.array(x))
        for point,v in enumerate(x):
            for rep in range(max(budgets)):
                jobs.append((str(batch),point*max(budgets)+rep,dict(gsa_x=[batch,point,rep,*v])))
    result=np.empty((2,base*6,max(budgets),2),dtype=np.uint8)
    start=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        for i,(batch,key,r) in enumerate(pool.map(task,jobs,chunksize=16)):
            point,rep=divmod(key,max(budgets))
            result[int(batch),point,rep]=[r['Y_min']>=w.Y_CRIT,r['psi_max']<=w.PSI_CRIT]
            if (i+1)%2000==0: print('gsa',i+1,'/',len(jobs),'elapsed',round(time.time()-start,1),flush=True)
    np.savez_compressed(OUT/'gsa_raw.npz',indicators=result,designs=np.array(Xs),budgets=budgets)
    rows=[];noise=[]
    def estimate(y,nv):
        # Conditional-reliability target. Independent MC noises for each point.
        # Noise unbiased variance of p-hat = p-hat*(1-p-hat)/(m-1).
        a,b=y[:,0],y[:,1];av,bv=nv[:,0],nv[:,1]
        pooled=np.r_[a,b];mu=pooled.mean()
        den=np.var(pooled,ddof=1)-np.mean(np.r_[av,bv])
        s1=[];st=[]
        for j in range(4):
            c=y[:,j+2];cv=nv[:,j+2]
            s1.append(np.mean(b*(c-a))/den if den>0 else np.nan)
            st.append((.5*np.mean((a-c)**2)-.5*np.mean(av+cv))/den if den>0 else np.nan)
        return np.array(s1),np.array(st),den
    for batch in range(2):
        for budget in budgets:
            for endpoint in range(2):
                y=result[batch,:,:budget,endpoint].mean(axis=1).reshape(base,6)
                nv=y*(1-y)/(budget-1)
                s1,st,den=estimate(y,nv)
                noise.append(dict(batch=batch,budget=budget,endpoint=endpoint,base_N=base,
                    noise_fraction=float(nv.mean()/np.var(y,ddof=1)),signal_variance=float(den),pooled_AB_variance=float(np.var(np.r_[y[:,0],y[:,1]],ddof=1)),pooled_AB_noise=float(np.mean(np.r_[nv[:,0],nv[:,1]]))))
                rng=np.random.default_rng(71)
                boot_full=[estimate(y[ix],nv[ix]) for ix in rng.integers(0,base,(500,base))]
                noise[-1]['bootstrap_positive_fraction']=float(np.mean([z[2]>0 for z in boot_full]))
                boot=np.array([z[:2] for z in boot_full])
                for j,name in enumerate(('gamma','m_shift','beta','delta')):
                    rows.append(dict(batch=batch,budget=budget,endpoint=endpoint,factor=name,S1=s1[j],ST=st[j],
                        S1_lo=float(np.nanquantile(boot[:,0,j],.025)),S1_hi=float(np.nanquantile(boot[:,0,j],.975)),
                        ST_lo=float(np.nanquantile(boot[:,1,j],.025)),ST_hi=float(np.nanquantile(boot[:,1,j],.975))))
    pd.DataFrame(rows).to_csv(OUT/'gsa_indices.csv',index=False)
    pd.DataFrame(noise).to_csv(OUT/'gsa_noise.csv',index=False)
    print(pd.DataFrame(noise).to_string(index=False),flush=True)

def metadata():
    import scipy,networkx
    paths=[MODEL,Path(__file__)]+list(INPUT.glob('*.csv'))+list((DATA).glob('*.csv'))
    doc=dict(date='2026-11-07',python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,networkx=networkx.__version__,component_order=w.COMPONENTS,
        ordinary_seeds=dict(scenario='10000+s',process='20000+100*s+1'),
        calibration_seeds='SeedSequence([999,101,s]), SeedSequence([999,202,s]), endpoint SeedSequence([999,991])',
        gsa_seeds='SeedSequence([20261029,batch,point,rep,101 or 202]); independent across points',
        hash_sha256={str(p.relative_to(BASE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'metadata.json').write_text(json.dumps(doc,indent=2))

def figures_stage(figure_dir=None):
    """Recreate publication plots from saved records; never rerun simulations.

    Example: python code/run_v13_structure.py --stage figures
    Uses --output-dir as the saved-result directory. PDF and PNG exports go
    to --figure-dir (default: figs/v13_structure). Also exports the CV
    diagnostic and a Python reconstruction of the closed-loop schematic.
    The original manuscript and its legacy closed_loop.jpg are not rewritten.
    Dependencies: numpy, pandas, scipy, matplotlib; bundled alignment auditor.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from audit_panel_alignment import require_matplotlib_panel_alignment
    root = BASE.parent
    FIG = Path(figure_dir).resolve() if figure_dir else root/'figs/v13_structure'
    FIG.mkdir(parents=True, exist_ok=True)
    required = ['main_summary.json', 'sweeps_summary.json', 'main_replicates.csv',
                'robust_replicates.csv', 'main_arrays.npz', 'robust_arrays.npz',
                'metadata.json']
    missing = [str(OUT/name) for name in required if not (OUT/name).is_file()]
    missing += [str(INPUT/name) for name in ['nodes_noto.csv','links_noto.csv'] if not (INPUT/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing saved figure inputs:\n' + '\n'.join(missing))
    def load(name):
        return json.loads((OUT/name).read_text(encoding='utf-8'))
    main=load('main_summary.json'); sweep=load('sweeps_summary.json')
    rep=pd.read_csv(OUT/'main_replicates.csv')
    rr=pd.read_csv(OUT/'robust_replicates.csv')
    ma=np.load(OUT/'main_arrays.npz'); ra=np.load(OUT/'robust_arrays.npz')
    components=load('metadata.json')['component_order']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'ps.fonttype':42,'savefig.dpi':300,'axes.labelsize':9,'legend.fontsize':8})
    manifest=[]
    def wilson(p,n):
        z=norm.ppf(.975); den=1+z*z/n
        c=(p+z*z/(2*n))/den
        h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return max(0.,c-h),min(1.,c+h)
    def save(fig,name,axes=None):
        fig.canvas.draw()
        if axes is not None:
            require_matplotlib_panel_alignment(fig,axes=axes,
                json_out=str(FIG/(name+'.alignment.json')))
        for extension in ('pdf','png'):
            target=FIG/(name+'.'+extension)
            fig.savefig(target,dpi=240)
            manifest.append(str(target))
        plt.close(fig)
    # Figure contract: show conditional model comparisons with source records and
    # keep confidence intervals distinct from inter-realization dispersion.
    nodes=pd.read_csv(INPUT/'nodes_noto.csv');links=pd.read_csv(INPUT/'links_noto.csv')
    fig,ax=plt.subplots(figsize=(6.4,5.1),layout='constrained')
    pos=nodes.set_index('node')
    for r in links.itertuples():
        if r.u<r.v:ax.plot([pos.loc[r.u,'lon'],pos.loc[r.v,'lon']],[pos.loc[r.u,'lat'],pos.loc[r.v,'lat']],color='.75',lw=1.2,zorder=1)
    sc=ax.scatter(nodes.lon,nodes.lat,c=nodes.pga_g,cmap='YlOrRd',vmin=.2,vmax=1.,s=90,edgecolor='black',lw=.6,zorder=3)
    offsets={'KZ':(-20,-8),'SK':(-23,3),'AN':(-23,10),'NT':(8,-10)}
    for r in nodes.itertuples():ax.annotate(r.node,(r.lon,r.lat),xytext=offsets.get(r.node,(6,5)),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',xlim=(136.58,137.36),ylim=(36.50,37.50))
    fig.colorbar(sc,ax=ax,label='Supplied scenario PGA (g)',shrink=.8)
    save(fig,'network')

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#777777']
    for ax,metric,lim,crit in zip(axes,['Y','psi'],[(0,1.03),(0,.36)],[.5,.2]):
        for i,label in enumerate(['M0','M1','M2','M3','M4','M5','M6']):
            a=ma[label+'__'+metric];t=np.arange(a.shape[1])+(metric=='psi')
            ax.plot(t,a.mean(axis=0),lw=1.2,label=label,c=colors[i])
            if label=='M0':ax.fill_between(t,*np.quantile(a,[.05,.95],axis=0),color=colors[i],alpha=.12)
        ax.axhline(crit,color='black',ls=':',lw=.9)
        ax.set(xlabel='Time (h)',ylabel='Aggregate delivery Y' if metric=='Y' else 'Normalized deprivation',ylim=lim,xlim=(0,72))
    for i,ax in enumerate(axes):ax.set_title(chr(97+i),loc='left',fontweight='bold')
    axes[0].legend(ncol=2,loc='lower left',frameon=False,fontsize=7)
    save(fig,'mechanisms',axes)

    fig,ax=plt.subplots(figsize=(6.3,3.0),layout='constrained')
    thresholds=np.linspace(.2,.9,15);threshold_rows=[]
    for label,c in [('M0',colors[0]),('M2',colors[2]),('binary_split',colors[4])]:
        y=rep[rep.config==label].Y_min.to_numpy();p=np.array([(y>=v).mean() for v in thresholds]);ci=np.array([wilson(v,len(y)) for v in p])
        ax.errorbar(thresholds,p,yerr=np.vstack((p-ci[:,0],ci[:,1]-p)),fmt='o-',ms=3,capsize=2,label=label,color=c)
        threshold_rows.extend(dict(config=label,threshold=t,R=value,lo=lo,hi=hi) for t,value,(lo,hi) in zip(thresholds,p,ci))
    ax.set(xlabel='Delivery threshold',ylabel='Reliability',ylim=(0,1.02));ax.legend(frameon=False)
    pd.DataFrame(threshold_rows).to_csv(OUT/'threshold_series.csv',index=False)
    save(fig,'thresholds')

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    baseline=rep[rep.config=='M0']
    for ax,key,title in zip(axes,['tc','tdc'],['Delivery crossing','Deprivation crossing']):
        vals=baseline[key].to_numpy();events=vals[vals<=72]
        counts,_,_=ax.hist(events,bins=np.arange(-.5,73.6,2),color=colors[0],edgecolor='white')
        ax.set_ylim(0,max(counts)*1.55)
        ax.set(xlabel='Crossing time (h)',ylabel='Event count',xlim=(-1,73),title=title)
        ax.text(.03,.95,f'Events: {len(events)} / {len(vals)}\nCensored: {np.sum(vals>72)}',transform=ax.transAxes,va='top',fontsize=8)
    save(fig,'crossing_times',axes)

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for ax,key in zip(axes,['R_sys','R_dc']):
        labels=['M0','soft_fastest','binary_split','binary_fastest'];ys=np.array([main[v][key] for v in labels]);cis=np.array([main[v]['wilson_R' if key=='R_sys' else 'wilson_Rdc'] for v in labels])
        ax.bar(np.arange(4),ys,color=[colors[0],colors[1],colors[4],colors[6]],width=.65)
        ax.errorbar(np.arange(4),ys,yerr=np.vstack((ys-cis[:,0],cis[:,1]-ys)),fmt='none',color='black',capsize=2)
        ax.set_xticks(range(4),['Soft\nsplit','Soft\nfastest','Binary\nsplit','Binary\nfastest'])
        ax.set(ylim=(0,1.03),ylabel='Delivery reliability' if key=='R_sys' else 'Deprivation reliability')
    save(fig,'factorial',axes)

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for ax,prefix,xlabel in zip(axes,['R_','D_'],['Routing interval (h)','Degradation substep (h)']):
        labs=sorted([v for v in sweep if v.startswith(prefix)],key=lambda v:float(v.split('_')[1]));x=np.array([float(v.split('_')[1]) for v in labs])
        for key,c,mark in [('R_sys',colors[0],'o'),('R_dc',colors[4],'s')]:
            ys=np.array([sweep[v][key] for v in labs]);cis=np.array([sweep[v]['wilson_R' if key=='R_sys' else 'wilson_Rdc'] for v in labs]);n=sweep[labs[0]]['N']
            ax.errorbar(x,ys,yerr=np.vstack((ys-cis[:,0],cis[:,1]-ys)),fmt=mark+'-',color=c,ms=3,capsize=2,label='Delivery' if key=='R_sys' else 'Deprivation')
        if prefix=='D_':ax.set_xscale('log',base=2)
        ax.set(xlabel=xlabel,ylabel='Reliability',ylim=(.68,1.02));ax.legend(frameon=False,fontsize=7)
    save(fig,'grids',axes)

    fig,ax=plt.subplots(figsize=(6.3,3.0),layout='constrained')
    for key,c in [('R_sys',colors[0]),('R_dc',colors[4])]:
        labs=sorted([v for v in sweep if v.startswith('CV_')],key=lambda v:float(v[3:]));x=np.array([float(v[3:]) for v in labs]);y=np.array([sweep[v][key] for v in labs])
        bounds=np.array([wilson(v,sweep[labs[0]]['N']) for v in y])
        ax.errorbar(x,y,yerr=np.vstack((y-bounds[:,0],bounds[:,1]-y)),fmt='o-',ms=3,capsize=2,label='Delivery' if key=='R_sys' else 'Deprivation',color=c)
    ax.set(xlabel='Reference increment CV',ylabel='Reliability',ylim=(.7,1.02));ax.legend(frameon=False)
    save(fig,'cv')

    # Joint network trajectory, selected transparently as the lowest-index replicate
    # with delivery failure in the saved ensemble. It illustrates model propagation,
    # not a controlled causal mediation or an observed field cascade.
    trace_ids=rr[(rr.config=='trace_M0')&(rr.Y_min<.5)].replicate
    if trace_ids.empty:
        raise ValueError('joint_trace requires a saved trace_M0 delivery-failure replicate')
    example=int(trace_ids.min())
    hh=ra['trace_M0__health_history'][example];aa=ra['trace_M0__driver_history'][example]
    selected=sorted(range(len(components)),key=lambda j:float(ra['trace_M0__H_final'][example,j]-ra['trace_M0__H_initial'][example,j]))[:3]
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for j,c in zip(selected,colors):
        axes[0].plot(np.arange(len(hh)),hh[:,j],label=components[j],color=c)
        axes[1].step(np.arange(len(aa)),aa[:,j],where='post',label=components[j],color=c)
    axes[0].set(xlabel='Time (h)',ylabel='Component health',ylim=(0,1.03))
    axes[1].set(xlabel='Time (h)',ylabel='Requested amount (t per 3 h)')
    for ax in axes:ax.legend(frameon=False,fontsize=7)
    save(fig,'joint_trace',axes)
    pd.DataFrame(dict(time_h=np.arange(len(hh)),**{components[j]+'_H':hh[:,j] for j in selected},**{components[j]+'_request':aa[:,j] for j in selected})).to_csv(OUT/'joint_trace.csv',index=False)

    # Reconstruct the existing feedback schematic as editable PDF text.
    fig,ax=plt.subplots(figsize=(12,3.6),layout='constrained')
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    left=r'$C_r \rightarrow \mathcal{P}_r \rightarrow A_r^\pi \rightarrow F_r^{\pi,m} \rightarrow$'
    right=r'$\rightarrow C_{r+1} \rightarrow \mathcal{P}_{r+1}$'
    ax.text(.01,.5,left,ha='left',va='center',fontsize=15)
    for y,q,nxt in [(.86,'0','1'),(.54,'1','2'),(.14,'M-1','M')]:
        text=(r'$\rho_{r,'+q+r'} \rightarrow \Delta H_{r,'+q+
              r'} \rightarrow C_{r,'+nxt+r'} \rightarrow F_{r,'+q+r'}^{\pi,m}$')
        ax.text(.58,y,text,ha='center',va='center',fontsize=15)
    ax.text(.58,.70,r'$\downarrow$',ha='center',va='center',fontsize=19)
    ax.text(.58,.34,r'$\vdots$',ha='center',va='center',fontsize=19)
    for x,dx in [(.365,.012),(.795,-.012)]:
        ax.plot([x+dx,x,x,x+dx],[.98,.98,.02,.02],color='black',lw=1)
    ax.text(.805,.5,right,ha='left',va='center',fontsize=15)
    save(fig,'closed_loop')
    ma.close(); ra.close()
    (FIG/'figure_manifest.json').write_text(json.dumps(dict(
        source_directory=OUT.relative_to(BASE.parent).as_posix(), files=[Path(x).relative_to(BASE.parent).as_posix() for x in manifest],
        note='Empirical trajectory bands: 5th–95th percentiles; reliability bars: Wilson 95% intervals; events include 72 h.'
    ),indent=2),encoding='utf-8')
    print('Figures generated in', FIG, flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage',choices=['main','sweeps','gsa','robust','figures'],required=True)
    parser.add_argument('--figure-dir',help='Figure export directory (figures stage only)')
    parser.add_argument('--N',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 2))
    parser.add_argument('--input-mode',choices=['csv','bounded'],default='csv')
    parser.add_argument('--output-dir',default='v13_structure_20261107')
    parser.add_argument('--gsa-base',type=int,default=32)
    args=parser.parse_args()
    os.environ['NOTO_REVISION_OUTPUT']=args.output_dir
    os.environ['NOTO_REVISION_MODE']=args.input_mode
    OUT=BASE.parent/'result'/args.output_dir
    INPUT=BASE.parent/'data'/OUT.name/'inputs'
    if args.stage=='figures':
        figures_stage(args.figure_dir)
        sys.exit(0)
    prepare_inputs();w.load_data(INPUT);metadata()
    if args.stage=='main': main_stage(args.workers,args.N)
    if args.stage=='sweeps': sweep_stage(args.workers,args.N)
    if args.stage=='gsa': gsa_stage(args.workers,base=args.gsa_base)
    if args.stage=='robust': robust_stage(args.workers,args.N)
