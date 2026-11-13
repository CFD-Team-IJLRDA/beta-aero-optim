import re,glob,os,sys,json
import numpy as np
sys.path.insert(0,os.path.dirname(__file__)); from cases import cases
pat=re.compile(r"Solving for (\w+), Initial residual = ([\d.eE+-]+)")
res={}
for model,m,path in cases():
    logs=[l for l in glob.glob(path+"/log.rhoSimpleFoam*")]
    log=sorted(logs,key=os.path.getmtime)[-1]
    hist={}; it=-1; seen=set(); times=[]
    for line in open(log,errors="ignore"):
        if line.startswith("Time = "):
            it+=1; seen=set(); times.append(float(line.split()[2]))
        mm=pat.search(line)
        if mm and it>=0:
            f=mm.group(1)
            if f in seen: continue   # first solve per iteration only
            seen.add(f); hist.setdefault(f,[]).append((it,float(mm.group(2))))
    end = "End" in open(log,errors="ignore").read()[-2000:]
    r={}
    for f,h in hist.items():
        a=np.array([v for _,v in h])
        r[f]=dict(first=a[:20].max(),last=a[-1],max_last1000=a[-1000:].max(),min_last1000=a[-1000:].min(),
                  med_4000_4500=float(np.median(a[-1000:-500])),med_last500=float(np.median(a[-500:])))
    res[f"{model}|{m}"]=dict(log=os.path.basename(log),niter=len(times),tend=times[-1],end=end,fields=r)
json.dump(res,open(os.path.join(os.path.dirname(__file__),"residuals.json"),"w"),indent=1)
for k,v in res.items():
    print(k,v["log"],"iters",v["niter"],"t",v["tend"],"ended" if v["end"] else "NOT ENDED")
    for f,r in v["fields"].items():
        print(f"   {f:9s} last {r['last']:.2e}  last1000 [{r['min_last1000']:.1e},{r['max_last1000']:.1e}]  median 4000-4500 {r['med_4000_4500']:.1e} -> last500 {r['med_last500']:.1e}")
