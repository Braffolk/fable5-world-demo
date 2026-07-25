#!/usr/bin/env python3
"""Offline exact-node/downsample gate for Candidate-J3 direct FE records."""

from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw

VERSION="candidate-j3-direct-fe-node-gate-v1"; SCALES=(4,16)

def imp(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def quant(v:np.ndarray)->dict[str,float|int]:
    x=v[np.isfinite(v)].astype(np.float64)
    return {"count":int(x.size),"p50":float(np.quantile(x,.5)),"p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),"max":float(np.max(x))} if x.size else {"count":0,"p50":0.,"p95":0.,"p99":0.,"max":0.}
def nearest(c:np.ndarray,p:np.ndarray)->np.ndarray:
    flat=c.reshape(-1,3); out=np.empty(flat.shape[0],np.uint8)
    for s in range(0,len(flat),65536):
        e=min(len(flat),s+65536); out[s:e]=np.argmin(np.sum((flat[s:e,None]-p[None])**2,axis=2),axis=1)
    return out.reshape(c.shape[:-1])
def reduce_node(a:np.ndarray,d:np.ndarray,p:np.ndarray,v:float,H:float,palette:np.ndarray):
    # 2x2 moment reduction from 256 to 128.
    def blocks(x): return x.reshape(128,2,128,2,*x.shape[2:])
    ab=blocks(a[...,None])[...,0]; abar=np.mean(ab,axis=(1,3))
    pb=blocks(p); pbar=np.mean(pb,axis=(1,3))
    y=np.where(np.isfinite(d),H-v*d,0.0); weighted=a*y
    ynum=np.sum(blocks(weighted[...,None])[...,0],axis=(1,3)); aden=np.sum(ab,axis=(1,3))
    ybar=np.divide(ynum,aden,out=np.zeros_like(ynum),where=aden>1e-8)
    cond=np.divide(pbar,abar[...,None],out=np.zeros_like(pbar),where=abar[...,None]>1e-8)
    aq=np.rint(np.clip(abar,0,1)*31)/31; yq=np.rint(np.clip(ybar/H,0,1)*127)/127*H; cls=nearest(cond,palette)
    dq=(H-yq)/v; pq=aq[...,None]*palette[cls]
    return (abar,np.where(abar>1/255,(H-ybar)/v,np.nan),pbar),(aq,dq,pq),cls
def score(hmod,truth,pred):
    a,d,p=truth; ah,dh,ph=pred; valid=np.isfinite(d)&(a>1/255)
    ea=np.abs(ah-a); ed=np.where(valid,np.abs(dh-d),np.nan); er=np.max(np.abs(ph-p),axis=3)
    qa,qd,qr=quant(ea),quant(ed),quant(er); mc=0.; mdi=-1
    for j in range(a.shape[0]):
        support=valid[j]; ex=support&((ea[j]>.20)|(ed[j]>.10)|(er[j]>.15)); f=hmod.largest_periodic_component(ex)/max(1,int(np.count_nonzero(support)))
        if f>mc:mc,mdi=float(f),j
    checks={"A95":qa["p95"]<=.08,"A99":qa["p99"]<=.20,"D95":qd["p95"]<=.05,"D99":qd["p99"]<=.10,"RGB95":qr["p95"]<=.06,"RGB99":qr["p99"]<=.15,"connected":mc<.01}
    return {"green":all(checks.values()),"checks":checks,"coverage":qa,"depthMetres":qd,"premulRgb":qr,"largestConnected":{"fraction":mc,"direction":mdi}}, {"A":np.max(ea,axis=0),"D":np.nanmax(ed,axis=0),"R":np.max(er,axis=0)}
def expand(x:np.ndarray)->np.ndarray:return np.repeat(np.repeat(x,2,axis=1),2,axis=2)
def heat(x:np.ndarray,l:float):
    t=np.clip(np.nan_to_num(x)/l,0,1); r=np.zeros((*x.shape,3),np.uint8);r[...,0]=np.rint(255*t);r[...,1]=np.rint(255*np.minimum(1,2*t)*(1-.6*t));r[...,2]=np.rint(40*(1-t));return r
def qa(path:Path,title:str,m):
    ps=[("A/.20",heat(m["A"],.2)),("D/.10",heat(m["D"],.1)),("RGB/.15",heat(m["R"],.15))];w=ps[0][1].shape[1]*2;h=ps[0][1].shape[0]*2;im=Image.new("RGB",(3*w,h+44),(15,15,15));dr=ImageDraw.Draw(im);dr.text((8,4),title,fill=(255,255,255))
    for i,(lab,pn) in enumerate(ps):im.paste(Image.fromarray(pn,"RGB").resize((w,h),Image.Resampling.NEAREST),(i*w,44));dr.text((i*w+8,23),lab,fill=(230,230,230))
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",default="src/assets/groundcover/calamagrostis-canescens.gcrp");args=ap.parse_args();root=Path(__file__).resolve().parents[2];source=(root/args.source).resolve()
    hmod=imp("h_for_j3",root/"tools/groundcover-bake/analyze_candidate_h_angular_continuity.py");imod=imp("i_for_j3",root/"tools/groundcover-bake/gate_candidate_i_vertical_extinction.py");profile=hmod.load_profile(source)
    truths={scale:imod.build_truth(hmod,profile,scale) for scale in SCALES}
    atoms={scale:hmod.AtomSet(coverage=truths[scale][0],depth=truths[scale][1],premul_rgb=truths[scale][2]) for scale in SCALES}
    near=[]
    for j in range(len(profile.slices)):
        covered,_,rgb=hmod.decode_slice_base(profile,j);values=rgb[covered]
        if values.size:near.append(values[::max(1,values.shape[0]//2048)])
    palette=hmod.fit_palette(atoms,near)
    doc=root/"docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-J3-DIRECT-FE-CODEC.md";recipe={"version":VERSION,"sourceSha256":sha(source),"scriptSha256":sha(Path(__file__)),"docSha256":sha(doc),"scales":SCALES,"palette":"deterministic joint near+both-filter-scales 16 classes"};rh=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest();out=root/"data/work/groundcover-candidate-j3-direct-fe"/recipe["sourceSha256"][:16]/rh[:16];out.mkdir(parents=True,exist_ok=True)
    report={}
    for scale in SCALES:
        truth=truths[scale];ta,td,tp=truth; small=[[],[],[]];pred=[[],[],[]];classes=[]
        for j,slice_data in enumerate(profile.slices):
            v=float(-slice_data.direction[1]);t,p,c=reduce_node(ta[j],td[j],tp[j],v,profile.top_h,palette);[small[k].append(t[k]) for k in range(3)];[pred[k].append(p[k]) for k in range(3)];classes.append(c)
        small=tuple(np.stack(x) for x in small);pred=tuple(np.stack(x) for x in pred);intrinsic,imaps=score(hmod,small,pred)
        expanded=tuple(expand(x) for x in pred);full,fmaps=score(hmod,truth,expanded);report[str(scale)]={"intrinsic128":intrinsic,"fullDownsampleAgainst256":full}
        (out/f"sigma{scale}-classes-u4.bin").write_bytes(np.stack(classes).astype(np.uint8).tobytes());qa(out/"qa"/f"sigma{scale}-intrinsic.png",f"J3 intrinsic sigma={scale} GREEN={intrinsic['green']}",imaps);qa(out/"qa"/f"sigma{scale}-full.png",f"J3 full sigma={scale} GREEN={full['green']}",fmaps)
    intrinsic_green=all(report[str(s)]["intrinsic128"]["green"] for s in SCALES);full_green=all(report[str(s)]["fullDownsampleAgainst256"]["green"] for s in SCALES);result={"schema":VERSION,"verdict":"GREEN" if full_green else "RED","intrinsic128Green":intrinsic_green,"fullAgainst256Green":full_green,"palette":palette.tolist(),"results":report,"recipe":recipe};(out/"report.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"output":str(out),"verdict":result["verdict"],"intrinsic128Green":intrinsic_green,"fullAgainst256Green":full_green,"results":report},indent=2))
if __name__=="__main__":main()
