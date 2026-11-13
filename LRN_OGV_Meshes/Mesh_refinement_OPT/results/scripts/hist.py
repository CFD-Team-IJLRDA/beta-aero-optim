import re,glob,os,sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from cases import cases
pat=re.compile(r"Solving for (\w+), Initial residual = ([\d.eE+-]+)")
np.save
H={}
for model,m,path in cases():
    log=sorted(glob.glob(path+"/log.rhoSimpleFoam*"),key=os.path.getmtime)[-1]
    hist={}; it=-1; seen=set()
    for line in open(log,errors="ignore"):
        if line.startswith("Time = "): it+=1; seen=set()
        mm=pat.search(line)
        if mm and it>=0 and mm.group(1) not in seen:
            seen.add(mm.group(1)); hist.setdefault(mm.group(1),[]).append(float(mm.group(2)))
    H[f"{model}|{m}"]={k:np.array(v) for k,v in hist.items()}
import pickle; pickle.dump(H,open(os.path.dirname(os.path.abspath(__file__))+"/hist.pkl","wb"))
for k,h in H.items():
    if "LM" not in k: continue
    p=h["p"]; g=h["gammaInt"]
    print(k," p:", " ".join(f"{p[i]:.0e}" for i in [10,100,300,500,1000,1500,2000,3000,4000,4999]))
    print(" "*len(k),"  gInt:", " ".join(f"{g[i]:.0e}" for i in [10,100,300,500,1000,1500,2000,3000,4000,4999]))
