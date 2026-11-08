from pathlib import Path
import sys,os,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
os.environ['NOTO_REVISION_OUTPUT']='v13_structure_20261107'
import run_v13_structure as d
from matplotlib.axes import Axes
original=Axes.errorbar
def errorbar(self,*a,**kw):
 for k in ['xerr','yerr']:
  if k in kw and kw[k] is not None:
   e=np.asarray(kw[k]);assert e.min()>=-1e-12
   kw[k]=np.maximum(e,0.)
 return original(self,*a,**kw)
Axes.errorbar=errorbar
d.figures_stage(R/'figs/v13_structure')
out=R/'result/v13_structure_20261107'
files=[Path(__file__),R/'code/start_v13_structure.py',R/'code/run_v13_structure.py',R/'code/v13_core.py',R/'code/audit_panel_alignment.py']+[p for p in out.rglob('*') if p.is_file() and p.name!='run_manifest.json']+[p for p in (R/'figs/v13_structure').glob('*') if p.is_file()]
(out/'run_manifest.json').write_text(json.dumps({'date':'2026-11-07','N':256,'M3_Ncal':64,'input_mode':'original CSV','gsa_base':32,'gsa_batches':2,'gsa_inner':[32,128],'figure_roundoff':'Only negative error-bar lengths >= -1e-12 are clipped to zero; CI endpoints unchanged','files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
