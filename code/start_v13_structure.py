from pathlib import Path
import shutil, subprocess, sys, hashlib, json
R=Path(__file__).resolve().parents[1];W=R/'result/logs/v13_structure';W.mkdir(parents=True,exist_ok=True)
p=R/'code/run_v13_structure.py'
for stage in ['main','sweeps','robust','gsa','figures']:
 cmd=[sys.executable,str(p),'--stage',stage,'--N','256','--workers','8','--input-mode','csv','--output-dir','v13_structure_20261107']
 if stage=='figures':cmd=[sys.executable,str(R/'code/v13_finish_figures.py')]
 print('START',stage,flush=True)
 with (W/(stage+'.log')).open('w',encoding='utf-8') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',stage,flush=True)
print('COMPLETE',flush=True)
