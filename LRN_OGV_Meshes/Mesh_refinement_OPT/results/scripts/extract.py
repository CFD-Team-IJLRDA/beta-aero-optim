import os,sys,glob,pickle
import numpy as np, pyvista as pv, vtk
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from cases import cases
g=1.4; R=287.05; mu=1.704e-5
def p0f(p,T,U): return p*(1+0.5*(g-1)*(U**2/(g*R*T)))**(g/(g-1))
def read(path):
    times=[float(os.path.basename(t)) for t in glob.glob(path+"/[0-9]*") if os.path.basename(t).replace('.','').isdigit()]
    tl=max(times)
    r=vtk.vtkOpenFOAMReader(); r.SetFileName(os.path.join(path,"x.foam"))   # dummy name, reader uses the case dir
    r.UpdateInformation(); r.SetTimeValue(tl)
    r.SetCreateCellToPoint(1); r.EnableAllPatchArrays(); r.EnableAllCellArrays()
   
    for i in range(r.GetNumberOfPatchArrays()):
        n=r.GetPatchArrayName(i); r.SetPatchArrayStatus(n, 1 if n in ("internalMesh","patch/airfoil") else 0)
    r.Update()
    mb=pv.wrap(r.GetOutput())
    return tl, mb
def find(mb,name):
    for i in range(mb.n_blocks):
        b=mb[i]; k=mb.get_block_name(i)
        if k==name: return b
        if isinstance(b,pv.MultiBlock):
            f=find(b,name)
            if f is not None: return f
    return None
out={}
for model,m,path in cases():
    tl,mb=read(path)
    vol=find(mb,"internalMesh"); air=find(mb,"airfoil")
    zc=0.5*(vol.bounds[4]+vol.bounds[5])
    def line(x,y0,y1,n=447):
        s=vol.sample_over_line((x,y0,zc),(x,y1,zc),resolution=n-1)
        U=s.point_data["U"]; return dict(y=s.points[:,1],p=s.point_data["p"],T=s.point_data["T"],rho=s.point_data["rho"],u=U[:,0],v=U[:,1])
    mp1=line(-0.02011,-0.028578,0.011736); mp2=line(0.087254,7.606e-5,0.04039)
    U1=np.hypot(mp1["u"],mp1["v"]); w1=mp1["rho"]*mp1["u"]
    p01=np.sum(w1*p0f(mp1["p"],mp1["T"],U1))/w1.sum(); p1=np.sum(w1*mp1["p"])/w1.sum()
    U2=np.hypot(mp2["u"],mp2["v"]); p02=p0f(mp2["p"],mp2["T"],U2)
    loss=(p01-p02)/(p01-p1); w2=mp2["rho"]*mp2["u"]
    omega_ma=np.sum(w2*loss)/w2.sum()
    alpha2=np.degrees(np.arctan2(np.sum(w2*mp2["v"]),np.sum(w2*mp2["u"])))
    alpha1=np.degrees(np.arctan2(np.sum(w1*mp1["v"]),np.sum(w1*mp1["u"])))
    M1=np.sum(w1*U1/np.sqrt(g*R*mp1["T"]))/w1.sum()
    # blade: slice the wall patch at mid-span -> ordered contour
    cc=air.cell_centers()
    sl=air.slice(normal="z",origin=(0,0,zc))
    pts=sl.points[:,:2]
    # order the polyline loop
    lines=sl.lines.reshape(-1,3)[:,1:]
    adj={}
    for a,b in lines: adj.setdefault(a,[]).append(b); adj.setdefault(b,[]).append(a)
    order=[lines[0,0]]; prev=None
    while True:
        nxt=[q for q in adj[order[-1]] if q!=prev]
        if not nxt or nxt[0]==order[0]: break
        prev=order[-1]; order.append(nxt[0])
    order=np.array(order); P=pts[order]
    pw=sl.point_data["p"][order]
    # wall shear from wall-adjacent cell: nearest internal cell to each face, tangential velocity / distance
    vc=vol.cell_centers(); 
    from scipy.spatial import cKDTree
    C=vc.points; mid=np.abs(C[:,2]-zc)<1e9
    tree=cKDTree(C[:,:2])
    d,idx=tree.query(P)
    Uc=vol.cell_data["U"][idx][:,:2]
    ile=np.argmin(P[:,0]); ite=np.argmax(P[:,0])
    # roll so contour starts at LE
    P=np.roll(P,-ile,axis=0); pw=np.roll(pw,-ile); Uc=np.roll(Uc,-ile,axis=0); d=np.roll(d,-ile)
    ite=int(np.argmax(P[:,0]))
    br=[slice(0,ite+1), np.r_[ite:len(P),0]]
    sides={}
    for b in br:
        PP=P[b]; tang=np.gradient(PP,axis=0); tang/=np.linalg.norm(tang,axis=1)[:,None]
        ut=np.sum(Uc[b]*tang,axis=1)        # + along LE->TE (branch 2 reversed below)
        sides.setdefault("raw",[]).append((PP,pw[b],ut,d[b]))
    out_s={}
    for PP,pp,ut,dd in sides["raw"]:
        if PP[0,0]>PP[-1,0]: PP=PP[::-1]; pp=pp[::-1]; ut=-ut[::-1]; dd=dd[::-1]
        out_s.setdefault("list",[]).append((PP,pp,ut,dd))
    a,b=out_s["list"]
    ss,ps=(a,b) if a[1].mean()<b[1].mean() else (b,a)
    chord_x=P[:,0].max()-P[:,0].min(); x0=P[:,0].min()
    def side(s):
        PP,pp,ut,dd=s
        Mis=np.sqrt(np.maximum((p01/pp)**((g-1)/g)-1,0)*2/(g-1))
        return dict(x=(PP[:,0]-x0)/chord_x,Mis=Mis,tau=mu*ut/dd,p=pp)
    out[f"{model}|{m}"]=dict(time=tl,ncells=vol.n_cells,mp2=dict(y=mp2["y"],loss=loss),omega=omega_ma,alpha1=alpha1,alpha2=alpha2,M1=M1,p01=p01,p1=p1,
                             SS=side(ss),PS=side(ps))
    S=out[f"{model}|{m}"]["SS"]
    neg=S["x"][(S["tau"]<0)&(S["x"]>0.02)&(S["x"]<0.995)]
    print(f"{model:12s}{m:6s} t={tl:.0f} cells={vol.n_cells} M1={M1:.4f} a1={alpha1:.2f} a2={alpha2:.3f} omega={omega_ma:.5f} MisMaxSS={S['Mis'].max():.4f}  SS sep x/c: {neg.min() if len(neg) else '-'} .. {neg.max() if len(neg) else '-'}",flush=True)
pickle.dump(out,open(os.path.dirname(os.path.abspath(__file__))+"/qoi.pkl","wb"))
