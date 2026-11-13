import os
B=os.path.expanduser("~/Scifi-Turbo")
MESH=[("M30k","LRN_OGV_GRID_FOR_OPT2_M30k"),("M45k","LRN_OGV_GRID_FOR_OPT2_M45k"),("M60k","LRN_OGV_GRID_FOR_OPT2"),("M90k","LRN_OGV_GRID_FOR_OPT2_M90k"),("M120k","LRN_OGV_GRID_FOR_OPT2_M120k")]
def cases():
    out=[]
    for model in ["kOmegaSST","kOmegaSSTLM"]:
        for m,d in MESH:
            sub = "kOmegaSSTLM_NP" if (m=="M60k" and model=="kOmegaSSTLM") else model
            out.append((model,m,os.path.join(B,d,"ADP",sub)))
    return out
