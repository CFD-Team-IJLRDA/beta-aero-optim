import os,sys,pickle
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
here=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,here)
Q=pickle.load(open(here+"/qoi.pkl","rb")); H=pickle.load(open(here+"/hist.pkl","rb"))
g=1.4; R=287.05; pitch=0.04039
les=os.path.expanduser("~/Scifi-Turbo/NeuralMixture-update/Musicaa_ProcessedData_Coarse/ADP_Coarse_LES_")
L1=pd.read_csv(les+"MP1.csv").dropna(); L2=pd.read_csv(les+"MP2.csv").dropna(); Lm=pd.read_csv(les+"Mis.csv").dropna()
def p0f(p,T,U): return p*(1+0.5*(g-1)*(U**2/(g*R*T)))**(g/(g-1))
U1=np.sqrt(L1.u**2+L1.v**2+L1.w**2); w1=L1.rho*L1.u
p01=np.sum(w1*p0f(L1.p,L1["T"],U1))/w1.sum(); p1=np.sum(w1*L1.p)/w1.sum()
lossL=(p01-p0f(L2.p,L2["T"],np.sqrt(L2.u**2+L2.v**2+L2.w**2)))/(p01-p1); w2=L2.rho*L2.u
lesQ=dict(omega=float(np.sum(w2*lossL)/w2.sum()),alpha2=float(np.degrees(np.arctan2(np.sum(w2*L2.v),np.sum(w2*L2.u)))),
          alpha1=float(np.degrees(np.arctan2(np.sum(w1*L1.v),np.sum(w1*L1.u)))),M1=float(np.sum(w1*L1.M)/w1.sum()))
yL=((L2.y/1000)%pitch)/pitch; o=np.argsort(yL); yL=yL.values[o]; lossL=lossL.values[o]
meshes=["M30k","M45k","M60k","M90k","M120k"]; col=dict(zip(meshes,["#d62728","#ff7f0e","#2ca02c","#1f77b4","#9467bd"]))
models=["kOmegaSST","kOmegaSSTLM"]
fig,ax=plt.subplots(3,2,figsize=(14,15))
for j,mo in enumerate(models):
    a=ax[0,j]
    for m in meshes:
        h=H[f"{mo}|{m}"]["p"]; a.semilogy(h,color=col[m],lw=1,label=m+(" (10000 it, old solver)" if (mo=="kOmegaSST" and m=="M60k") else ""))
    a.set_title(f"{mo}: p initial residual"); a.set_xlabel("iteration"); a.legend(fontsize=8); a.grid(alpha=.3)
    a=ax[1,j]
    a.plot(Lm.xw/Lm.xw.max(),Lm.M_is,"k.",ms=2,label="Coarse LES")
    for m in meshes:
        q=Q[f"{mo}|{m}"]
        for s,ls in (("SS","-"),("PS","--")): a.plot(q[s]["x"],q[s]["Mis"],ls,color=col[m],lw=1.2,label=m if s=="SS" else None)
    a.set_title(f"{mo}: isentropic Mach"); a.set_xlabel("x/c"); a.set_ylabel("M_is"); a.legend(fontsize=8); a.grid(alpha=.3)
    a=ax[2,j]
    a.plot(yL,lossL,"k.",ms=2,label="Coarse LES")
    for m in meshes:
        q=Q[f"{mo}|{m}"]; a.plot(q["mp2"]["y"]/pitch,q["mp2"]["loss"],color=col[m],lw=1.2,label=m)
    a.set_title(f"{mo}: total-pressure loss at MP2"); a.set_xlabel("y/pitch"); a.set_ylabel("ω"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(here+"/mesh_study.png",dpi=130)
# zooms: wake core and suction peak
fig,ax=plt.subplots(2,2,figsize=(14,9))
for j,mo in enumerate(models):
    a=ax[0,j]; a.plot(Lm.xw/Lm.xw.max(),Lm.M_is,"k.",ms=3,label="Coarse LES")
    for m in meshes:
        q=Q[f"{mo}|{m}"]; a.plot(q["SS"]["x"],q["SS"]["Mis"],color=col[m],lw=1.2,label=m)
    a.set_xlim(0,0.5); a.set_title(f"{mo}: suction side M_is, front half"); a.legend(fontsize=8); a.grid(alpha=.3)
    a=ax[1,j]; a.plot(yL,lossL,"k.",ms=3,label="Coarse LES")
    for m in meshes:
        q=Q[f"{mo}|{m}"]; yy=q["mp2"]["y"]/pitch; a.plot(yy,q["mp2"]["loss"],color=col[m],lw=1.2,label=m)
    pk=yy[np.argmax(Q[f"{mo}|M120k"]["mp2"]["loss"])]; a.set_xlim(pk-0.2,pk+0.2); a.set_title(f"{mo}: wake core at MP2"); a.legend(fontsize=8); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(here+"/mesh_study_zoom.png",dpi=130)
# table
rows=[]
for mo in models:
    ref=Q[f"{mo}|M120k"]
    for m in meshes:
        q=Q[f"{mo}|{m}"]
        def dMis(s):
            x=np.linspace(0.02,0.98,300); return np.sqrt(np.mean((np.interp(x,q[s]["x"],q[s]["Mis"])-np.interp(x,ref[s]["x"],ref[s]["Mis"]))**2))
        yy=q["mp2"]["y"]/pitch; l=q["mp2"]["loss"]
        rows.append(dict(model=mo,mesh=m,cells=q["ncells"],iters=int(q["time"]),p_res_final=H[f"{mo}|{m}"]["p"][-1],
            M1=q["M1"],alpha1=q["alpha1"],alpha2=q["alpha2"],omega=q["omega"],
            omega_vs_finest_pct=100*(q["omega"]/ref["omega"]-1),wake_peak=l.max(),wake_peak_y=yy[np.argmax(l)],
            Mis_SS_max=q["SS"]["Mis"].max(),rms_dMis_SS_vs_finest=dMis("SS"),rms_dMis_PS_vs_finest=dMis("PS")))
T=pd.DataFrame(rows); T.to_csv(here+"/mesh_study_table.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(T.to_string(index=False,float_format=lambda v:f"{v:.4g}"))
print("Coarse LES:",{k:round(v,4) for k,v in lesQ.items()}, "wake peak", lossL.max())
