# Sec 4.4: random block, source-label ranking and angular-distance ranking (on target images). Run from repo root.
# Target drops always: translator fitted on the source, target head, same head setting on both sides.
import pandas as pd, re, glob
from scipy.stats import spearmanr
df=pd.concat([pd.read_csv(p) for p in glob.glob("experiments/4_4_without_target_data/accuracies/*.csv")],ignore_index=True)
def hs(a):
    s=re.findall(r"\((\d+),\s*(\d+)\)",str(a)); return int(s[0][0]) if len(s)==1 and int(s[0][1])-int(s[0][0])==1 else (-1 if a=="[]" else None)
df["H"]=df.approx_layer.map(hs); df=df.dropna(subset=["H"]).drop_duplicates(["model","dataset","fit_dataset","H"],keep="last")
P={"chestxray14_multilabel":"chestxray_pneumonia","chestxray_pneumonia":"chestxray14_multilabel","patchcamelyon":"tcga_ut","tcga_ut":"patchcamelyon"}
MN={"rad-dino":"RAD-DINO","rad-dino-maira-2":"RAD-DINO-MAIRA-2","dinov2-base":"DINOv2-B","vit-large-patch16-224":"ViT-L","phikon-v2":"Phikon-v2"}
MK={"rad-dino":"raddino","rad-dino-maira-2":"raddinomaira2","dinov2-base":"dinov2base","vit-large-patch16-224":"vitlarge","phikon-v2":"phikonv2"}
DN={"chestxray14_multilabel":"ChestX-ray14","chestxray_pneumonia":"Pneumonia","patchcamelyon":"PCam","tcga_ut":"TCGA-UT"}
def drops(m,d,fit,col):
    g=df[(df.model==m)&(df.dataset==d)]; base=g[g.H==-1].auroc_mean.iloc[0]
    return ((base-g[(g.fit_dataset==fit)&(g.H>=0)].set_index("H")[col])*100).dropna()
def angdist(m,d):
    ev=pd.read_csv(f"experiments/4_1-4_2_choosing_blocks/medical/analysis/output_level_eval_s4_{MK[m.split('/')[1]]}_{d.replace('_','')}.csv")
    ev=ev[ev.skip_to-ev.skip_from==1]; return ev.set_index("skip_from").angular_dist
R=[]
for m in df.model.unique():
    for t,s in P.items():
        if not ((df.model==m)&(df.dataset==t)).any(): continue
        ang=angdist(m,t)   # label-free, so computed on the target's own calibration images
        angS=angdist(m,s)  # scenario where only source images are available
        for head,col in [("retrained","auroc_mean"),("frozen","frozen_head_auroc_mean")]:
            src=drops(m,s,s,col); tgt=drops(m,t,s,col); i=src.index.intersection(tgt.index).intersection(ang.index).intersection(angS.index)
            best=tgt[i].min(); bl=src[i].idxmin(); ba=ang[i].idxmin(); bs=angS[i].idxmin()
            R.append({"Group":"general" if MN[m.split('/')[1]] in("ViT-L","DINOv2-B") else "domain",
                "Model":MN[m.split("/")[1]],"Pair":f"{DN[s]} → {DN[t]}","Head":head,"n":len(i),
                "Random regret":tgt[i].mean()-best,
                "Label rho":spearmanr(src[i],tgt[i])[0],"Label pick":f"{bl}→{bl+1}","Label drop":tgt[bl],"Label regret":tgt[bl]-best,
                "Ang rho":spearmanr(ang[i],tgt[i])[0],"Ang pick":f"{ba}→{ba+1}","Ang drop":tgt[ba],"Ang regret":tgt[ba]-best,
                "AngSrc rho":spearmanr(angS[i],tgt[i])[0],"AngSrc pick":f"{bs}→{bs+1}","AngSrc drop":tgt[bs],"AngSrc regret":tgt[bs]-best})
R=pd.DataFrame(R).round(2); R.to_csv("experiments/4_4_without_target_data/analysis/s5_random_angdist.csv",index=False)
pd.set_option("display.width",250); print(R.to_string(index=False))
num=["Random regret","Label rho","Label drop","Label regret","Ang rho","Ang drop","Ang regret","AngSrc rho","AngSrc drop","AngSrc regret"]
print(R.groupby(["Group","Head"])[num].agg(["mean","max"]).round(2).to_string())
print(R.groupby("Head")[num].agg(["mean","max"]).round(2).to_string())
