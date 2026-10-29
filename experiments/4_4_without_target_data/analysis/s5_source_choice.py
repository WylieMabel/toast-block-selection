# Sec 4.4: does the source's block ranking pick good blocks on the target? Run from repo root.
# Source ranking: translator fitted on source, head trained on source, scored on source test set.
# Target ranking: translator fitted on source or target, head trained on target, scored on target test set.
# Head setting (retrained / frozen) is the same on both sides.
import pandas as pd, re, glob
from scipy.stats import spearmanr
df=pd.concat([pd.read_csv(p) for p in glob.glob("experiments/4_4_without_target_data/accuracies/*.csv")],ignore_index=True)
def hs(a):
    s=re.findall(r"\((\d+),\s*(\d+)\)",str(a)); return int(s[0][0]) if len(s)==1 and int(s[0][1])-int(s[0][0])==1 else (-1 if a=="[]" else None)
df["H"]=df.approx_layer.map(hs); df=df.dropna(subset=["H"]).drop_duplicates(["model","dataset","fit_dataset","H"],keep="last")
P={"chestxray14_multilabel":"chestxray_pneumonia","chestxray_pneumonia":"chestxray14_multilabel","patchcamelyon":"tcga_ut","tcga_ut":"patchcamelyon"}
MN={"rad-dino":"RAD-DINO","rad-dino-maira-2":"RAD-DINO-MAIRA-2","dinov2-base":"DINOv2-B","vit-large-patch16-224":"ViT-L","phikon-v2":"Phikon-v2"}
DN={"chestxray14_multilabel":"ChestX-ray14","chestxray_pneumonia":"Pneumonia","patchcamelyon":"PCam","tcga_ut":"TCGA-UT"}
COL={"retrained":"auroc_mean","frozen":"frozen_head_auroc_mean"}
def drops(m,d,fit,col):
    g=df[(df.model==m)&(df.dataset==d)]; base=g[g.H==-1].auroc_mean.iloc[0]
    r=g[(g.fit_dataset==fit)&(g.H>=0)].set_index("H")[col]
    return ((base-r)*100).dropna()
R=[]
for m in df.model.unique():
    for t,s in P.items():
        if not ((df.model==m)&(df.dataset==t)).any(): continue
        for head,col in COL.items():
            src=drops(m,s,s,col)
            for tr in ["source","target"]:
                tgt=drops(m,t,s if tr=="source" else t,col); i=src.index.intersection(tgt.index)
                pick=src[i].nsmallest(5).index
                R.append({"Model":MN[m.split("/")[1]],"Source":DN[s],"Target":DN[t],"Head setting":head,
                    "Source rank: translator / head":"source / source","Target drops: translator / head":f"{tr} / target",
                    "rho":spearmanr(src[i],tgt[i])[0],"Top-5 (pp)":tgt[pick].mean(),"Best 5 (pp)":tgt[i].nsmallest(5).mean(),
                    "Gap (pp)":tgt[pick].mean()-tgt[i].nsmallest(5).mean(),"Worst pick (pp)":tgt[pick].max(),"Mean block (pp)":tgt[i].mean()})
R=pd.DataFrame(R).round(2); R.to_csv("experiments/4_4_without_target_data/analysis/s5_source_choice.csv",index=False)
pd.set_option("display.width",300); print(R.to_string(index=False))
print(R.groupby(["Head setting","Target drops: translator / head"])[["rho","Top-5 (pp)","Gap (pp)","Worst pick (pp)"]].agg(["mean","max"]).round(2).to_string())
