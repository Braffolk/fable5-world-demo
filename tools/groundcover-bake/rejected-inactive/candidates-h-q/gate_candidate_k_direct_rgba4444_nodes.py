#!/usr/bin/env python3
"""Stored-node/downsample gate for Candidate-K direct premul RGBA4444 FE."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

VERSION="candidate-k-direct-rgba4444-node-gate-v1";SCALES=(4,16);CANDIDATES=((64,180),(80,160),(96,144),(128,128))
def imp(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(v):
    x=v[np.isfinite(v)].astype(np.float64)
    return {"count":int(x.size),"p50":float(np.quantile(x,.5)),"p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),"max":float(np.max(x))}
def area_matrix(dst:int,src:int=256)->np.ndarray:
    w=np.zeros((dst,src),np.float32);scale=src/dst
    for i in range(dst):
        lo=i*scale;hi=(i+1)*scale;a=int(np.floor(lo));b=int(np.ceil(hi))
        for j in range(a,b):w[i,j%src]=max(0.,min(hi,j+1)-max(lo,j))/scale
    return w
def down(x,w):
    if x.ndim==2:return w@x@w.T
    return np.stack([w@x[...,c]@w.T for c in range(x.shape[2])],axis=2)
def up_nearest(x,dst=256):
    src=x.shape[0];idx=np.floor((np.arange(dst)+.5)*src/dst).astype(int);idx=np.clip(idx,0,src-1)
    return x[idx[:,None],idx[None,:]]
def score(hmod,ta,tp,pa,pp):
    ea=np.abs(pa-ta);er=np.max(np.abs(pp-tp),axis=3);qa,qr=q(ea),q(er);mc=0.;mdi=-1
    per=[]
    for j in range(ta.shape[0]):
        support=(ta[j]>1/255)|(pa[j]>1/255);ex=support&((ea[j]>.20)|(er[j]>.15));f=hmod.largest_periodic_component(ex)/max(1,int(np.count_nonzero(support)))
        aq,rq=q(ea[j]),q(er[j]);checks={"A95":aq["p95"]<=.08,"A99":aq["p99"]<=.20,"RGB95":rq["p95"]<=.06,"RGB99":rq["p99"]<=.15,"connected":f<.01};per.append({"index":j,"green":all(checks.values()),"checks":checks,"A":aq,"RGB":rq,"connected":f})
        if f>mc:mc,mdi=float(f),j
    checks={"A95":qa["p95"]<=.08,"A99":qa["p99"]<=.20,"RGB95":qr["p95"]<=.06,"RGB99":qr["p99"]<=.15,"connected":mc<.01,"everyDirection":all(x["green"] for x in per)}
    return {"green":all(checks.values()),"checks":checks,"coverage":qa,"premulRgb":qr,"largestConnected":{"fraction":mc,"direction":mdi},"perDirection":per},{"A":np.max(ea,axis=0),"R":np.max(er,axis=0)}
def heat(x,l):
    t=np.clip(np.nan_to_num(x)/l,0,1);r=np.zeros((*x.shape,3),np.uint8);r[...,0]=np.rint(255*t);r[...,1]=np.rint(255*np.minimum(1,2*t)*(1-.6*t));r[...,2]=np.rint(40*(1-t));return r
def image(path,title,m):
    ps=[("A/.20",heat(m["A"],.2)),("RGB/.15",heat(m["R"],.15))];w=ps[0][1].shape[1]*2;h=ps[0][1].shape[0]*2;im=Image.new("RGB",(2*w,h+44),(15,15,15));d=ImageDraw.Draw(im);d.text((8,4),title,fill=(255,255,255))
    for i,(lab,p) in enumerate(ps):im.paste(Image.fromarray(p,"RGB").resize((w,h),Image.Resampling.NEAREST),(i*w,44));d.text((i*w+8,23),lab,fill=(230,230,230))
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
def main():
    a=argparse.ArgumentParser();a.add_argument("--source",default="src/assets/groundcover/calamagrostis-canescens.gcrp");z=a.parse_args();root=Path(__file__).resolve().parents[2];source=(root/z.source).resolve();hm=imp("h_k",root/"tools/groundcover-bake/analyze_candidate_h_angular_continuity.py");im=imp("i_k",root/"tools/groundcover-bake/gate_candidate_i_vertical_extinction.py");profile=hm.load_profile(source);truth={s:im.build_truth(hm,profile,s) for s in SCALES}
    doc=root/"docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-K-DIRECT-APPEARANCE-FE.md";recipe={"version":VERSION,"source":sha(source),"script":sha(Path(__file__)),"doc":sha(doc),"candidates":CANDIDATES};rh=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest();out=root/"data/work/groundcover-candidate-k-rgba4444"/recipe["source"][:16]/rh[:16];out.mkdir(parents=True,exist_ok=True);results={}
    for pages,res in CANDIDATES:
        w=area_matrix(res);entry={"bytes":pages*res*res*16,"MiB":pages*res*res*16/1048576,"scales":{}}
        for scale in SCALES:
            ta=truth[scale][0];tp=truth[scale][2];sa=np.stack([down(x,w) for x in ta]);sp=np.stack([down(x,w) for x in tp]);pa=np.rint(np.clip(sa,0,1)*15)/15;pp=np.rint(np.clip(sp,0,1)*15)/15
            intrinsic,mi=score(hm,sa,sp,pa,pp);fa=np.stack([up_nearest(x) for x in pa]);fp=np.stack([up_nearest(x) for x in pp]);full,mf=score(hm,ta,tp,fa,fp);entry["scales"][str(scale)]={"intrinsic":intrinsic,"fullAgainst256":full};image(out/"qa"/f"p{pages}-r{res}-s{scale}-intrinsic.png",f"K {pages}@{res} sigma={scale} intrinsic",mi);image(out/"qa"/f"p{pages}-r{res}-s{scale}-full.png",f"K {pages}@{res} sigma={scale} full",mf)
        entry["intrinsicGreen"]=all(entry["scales"][str(s)]["intrinsic"]["green"] for s in SCALES);entry["fullGreen"]=all(entry["scales"][str(s)]["fullAgainst256"]["green"] for s in SCALES);results[f"{pages}@{res}"]=entry
    green=[k for k,v in results.items() if v["fullGreen"]];report={"schema":VERSION,"verdict":"GREEN" if green else "RED","fullGreenCandidates":green,"results":results,"recipe":recipe};(out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"output":str(out),"verdict":report["verdict"],"fullGreenCandidates":green,"summary":{k:{"intrinsic":v["intrinsicGreen"],"full":v["fullGreen"],"MiB":v["MiB"]} for k,v in results.items()}},indent=2))
if __name__=="__main__":main()
