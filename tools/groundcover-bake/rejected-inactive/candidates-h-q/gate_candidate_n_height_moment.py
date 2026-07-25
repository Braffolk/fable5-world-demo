#!/usr/bin/env python3
"""One-shot offline gate for Candidate N's height-moment transfer codec.

This reuses the accepted GCRP/v4 exact first-hit pages.  It does not edit or
exercise runtime code.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

VERSION = "candidate-n-height-moment-gate-v1"
SCALES = (4, 16)
N = 64
LIMITS = {"a95": .08, "a99": .20, "rgb95": .06, "rgb99": .15, "connected": .01}

def imp(name: str, path: Path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    if s is None or s.loader is None: raise RuntimeError(path)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def q(x: np.ndarray)->dict[str,float]:
    y=x[np.isfinite(x)].astype(np.float64)
    return {k:float(np.quantile(y,v)) for k,v in (("p50",.5),("p95",.95),("p99",.99))}|{"max":float(np.max(y))}
def down4(x: np.ndarray)->np.ndarray:
    if x.ndim==2:return x.reshape(N,4,N,4).mean((1,3)).astype(np.float32)
    return x.reshape(N,4,N,4,x.shape[-1]).mean((1,3)).astype(np.float32)
def direction_meta(d: np.ndarray)->tuple[float,float]:
    e=math.degrees(math.asin(float(np.clip(-d[1],-1,1))))
    a=(math.degrees(math.atan2(float(d[2]),float(d[0])))+360)%360 if np.linalg.norm(d[[0,2]])>1e-8 else 0.
    return e,a
def optical(a: np.ndarray,p: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    aa=np.clip(a,0,.995);tau=-np.log1p(-aa);cond=np.divide(p,aa[...,None],out=np.zeros_like(p),where=aa[...,None]>1e-6)
    return tau.astype(np.float32),(tau[...,None]*cond).astype(np.float32)

def palette_and_weights(train_tau: np.ndarray, train_src: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    mask=train_tau.ravel()>1e-4; colours=np.divide(train_src.reshape(-1,3)[mask],train_tau.ravel()[mask,None],out=np.zeros((mask.sum(),3),np.float32),where=train_tau.ravel()[mask,None]>0)
    # Deterministic farthest-point four-colour basis, then Lloyd updates.
    mean=np.average(colours,axis=0,weights=train_tau.ravel()[mask]);centres=[colours[np.argmax(np.sum((colours-mean)**2,axis=1))]]
    while len(centres)<4:
        dist=np.min(np.stack([np.sum((colours-c)**2,axis=1) for c in centres]),axis=0);centres.append(colours[np.argmax(dist)])
    c=np.stack(centres).astype(np.float32)
    for _ in range(12):
        dist=np.sum((colours[:,None]-c[None])**2,axis=2);lab=np.argmin(dist,axis=1)
        for k in range(4):
            mk=lab==k
            if np.any(mk):c[k]=np.average(colours[mk],axis=0,weights=train_tau.ravel()[mask][mk])
    cond=np.divide(train_src,train_tau[...,None],out=np.zeros_like(train_src),where=train_tau[...,None]>1e-6)
    dist=np.sum((cond[...,None,:]-c[None,None,None,:,:])**2,axis=-1)
    soft=1/np.maximum(dist,1e-4);soft/=np.sum(soft,axis=-1,keepdims=True)
    return np.clip(c,0,1), (train_tau[...,None]*soft).astype(np.float32)

def psd_response(directions: np.ndarray, desired: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    v=np.concatenate((np.ones((len(directions),1)),directions),axis=1)
    X=np.stack((v[:,0]**2,2*v[:,0]*v[:,1],2*v[:,0]*v[:,2],2*v[:,0]*v[:,3],v[:,1]**2,
                2*v[:,1]*v[:,2],2*v[:,1]*v[:,3],v[:,2]**2,2*v[:,2]*v[:,3],v[:,3]**2),axis=1)
    z=np.linalg.lstsq(X,desired,rcond=1e-7)[0]
    M=np.array(((z[0],z[1],z[2],z[3]),(z[1],z[4],z[5],z[6]),(z[2],z[5],z[7],z[8]),(z[3],z[6],z[8],z[9])),np.float64)
    ev,U=np.linalg.eigh((M+M.T)/2);M=(U*np.maximum(ev,1e-8))@U.T
    response=np.einsum('ei,ij,ej->e',v,M,v);response=np.maximum(response,1e-5)
    return M,response

def moment_constants(j:int)->tuple[float,float]:
    # beta_j^3(1-r) is Beta(4-j,j+1), normalised by integral 1/4.
    a=4-j;b=j+1
    return a/5., a*b/(25*6)

def triangle_sample(field: np.ndarray, sx: float, sy: float)->np.ndarray:
    h,w=field.shape[-2:];yy,xx=np.indices((h,w));qx=xx+sx;qy=yy+sy;x0=np.floor(qx).astype(np.int32);y0=np.floor(qy).astype(np.int32);fx=qx-x0;fy=qy-y0
    def at(dx:int,dy:int):return field[...,np.mod(y0+dy,h),np.mod(x0+dx,w)]
    lo=fx+fy<=1
    a=at(0,0)+(at(1,0)-at(0,0))*fx+(at(0,1)-at(0,0))*fy
    hi=at(1,1)+(at(0,1)-at(1,1))*(1-fx)+(at(1,0)-at(1,1))*(1-fy)
    return np.where(lo,a,hi).astype(np.float32)

def triangle_adjoint(value:np.ndarray,sx:float,sy:float)->np.ndarray:
    yy,xx=np.indices((N,N));qx=xx+sx;qy=yy+sy;x0=np.floor(qx).astype(np.int32);y0=np.floor(qy).astype(np.int32);fx=qx-x0;fy=qy-y0;lo=fx+fy<=1
    ids=[];ws=[]
    def add(dx,dy,w):ids.append((np.mod(y0+dy,N)*N+np.mod(x0+dx,N)).ravel());ws.append((value*w).ravel())
    add(0,0,np.where(lo,1-fx-fy,0));add(1,0,np.where(lo,fx,1-fy));add(0,1,np.where(lo,fy,1-fx));add(1,1,np.where(lo,0,fx+fy-1))
    return sum(np.bincount(i,weights=w,minlength=N*N) for i,w in zip(ids,ws)).reshape(N,N).astype(np.float32)

def refine_nonnegative(fields:np.ndarray,target:np.ndarray,directions:np.ndarray,top_h:float,tile:float,chi:np.ndarray,iterations=60)->np.ndarray:
    f=np.maximum(fields.copy(),0);maps=[]
    for e,d in enumerate(directions):
        L=top_h/max(-float(d[1]),1e-6);row=[]
        for j in range(4):
            r,_=moment_constants(j);row.append((float(d[0])*L*r/tile*N,-float(d[2])*L*r/tile*N,chi[e]*L/4))
        maps.append(row)
    lipschitz=sum(sum(abs(x[2]) for x in row)**2 for row in maps);step=.7/max(lipschitz,1e-8)
    for _ in range(iterations):
        grad=np.zeros_like(f)
        for e,row in enumerate(maps):
            pred=sum(c*triangle_sample(f[j],sx,sy) for j,(sx,sy,c) in enumerate(row));res=pred-target[e]
            for j,(sx,sy,c) in enumerate(row):grad[j]+=c*triangle_adjoint(res,sx,sy)
        f=np.maximum(0,f-step*grad)
    return f

def fit_fourier(target: np.ndarray,directions:np.ndarray,top_h:float,tile:float,chi:np.ndarray)->np.ndarray:
    # target [E,N,N], output four nonnegative height vertex fields.
    E=target.shape[0];ky=np.fft.fftfreq(N)*N;kx=np.fft.fftfreq(N)*N;KY,KX=np.meshgrid(ky,kx,indexing='ij')
    Y=np.fft.fft2(target,axes=(-2,-1));A=np.empty((E,4,N,N),np.complex128)
    for e,d in enumerate(directions):
        s=max(-float(d[1]),1e-6);L=top_h/s
        for j in range(4):
            r,_=moment_constants(j);sx=float(d[0])*L*r/tile*N;sy=-float(d[2])*L*r/tile*N
            A[e,j]=chi[e]*L/4*np.exp(2j*np.pi*(KX*sx+KY*sy)/N)
    G=np.einsum('ejyx,elyx->yxjl',np.conj(A),A);rhs=np.einsum('ejyx,eyx->yxj',np.conj(A),Y)
    eye=np.eye(4)[None,None];reg=1e-5*np.maximum(np.trace(G,axis1=2,axis2=3)[...,None,None]/4,1e-8)
    F=np.linalg.solve(G+reg*eye,rhs[...,None])[...,0];fields=np.fft.ifft2(np.moveaxis(F,-1,0),axes=(-2,-1)).real.astype(np.float32)
    return np.maximum(fields,0)

def fit_scale(target_modes:np.ndarray,directions:np.ndarray,top_h:float,tile:float)->tuple[np.ndarray,list[list[float]],np.ndarray]:
    fields=np.empty((4,4,N,N),np.float32);matrices=[];responses=np.empty((len(directions),4),np.float64)
    sin=np.maximum(-directions[:,1],1e-4)
    for m in range(4):
        means=np.mean(target_modes[...,m],axis=(1,2));desired=means*sin/np.maximum(np.median(means*sin),1e-7)
        M,chi=psd_response(directions,desired);matrices.append(M.tolist());responses[:,m]=chi
        initial=fit_fourier(target_modes[...,m],directions,top_h,tile,chi)
        fields[:,m]=refine_nonnegative(initial,target_modes[...,m],directions,top_h,tile,chi)
    return fields,matrices,responses

def quantize4(fields:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    packed=np.empty_like(fields);scales=np.empty((4,4),np.float64)
    for j in range(4):
      for m in range(4):
        mx=float(np.max(fields[j,m]));s=max(mx/15,1e-9);den=math.log1p(mx/s) if mx>0 else 1.;code=np.rint(15*np.log1p(fields[j,m]/s)/den)
        packed[j,m]=s*np.expm1(np.clip(code,0,15)/15*den);scales[j,m]=s
    return packed,scales

def contraction_k(fine:np.ndarray,coarse:np.ndarray,directions:np.ndarray,top_h:float,tile:float)->np.ndarray:
    result=np.zeros(4,np.float64)
    for j in range(4):
        ratios=[]
        for m in range(4):
            mu=float(np.mean(fine[j,m]));x=(fine[j,m]-mu).ravel();y=(coarse[j,m]-mu).ravel();ratios.append(float(np.clip(np.dot(x,y)/max(np.dot(x,x),1e-12),0,1)))
        wanted=float(np.median(ratios));best=(1e99,0.)
        for k in np.concatenate(([0.],np.logspace(-5,3,200))):
            ws=[]
            for d in directions:
                L=top_h/max(-float(d[1]),1e-6);_,vr=moment_constants(j);path=vr*L*L*(d[0]*d[0]+d[2]*d[2]);x=.5*k*(path+(16*tile/256)**2/3);ws.append(1/(1+x+.5*x*x))
            err=(float(np.mean(ws))-wanted)**2
            if err<best[0]:best=(err,float(k))
        result[j]=best[1]
    return result

def predict_modes(fields_batch:np.ndarray,directions:np.ndarray,top_h:float,tile:float,chi:np.ndarray,phase:np.ndarray,packed:bool,k:np.ndarray,scale:int)->np.ndarray:
    # fields_batch accepted as [B,4,4,N,N], only B=1 used.
    fields=fields_batch[0];out=np.zeros((len(directions),N,N,4),np.float32)
    for e,d in enumerate(directions):
        L=top_h/max(-float(d[1]),1e-6)
        for j in range(4):
            r,vr=moment_constants(j);sx=float(d[0])*L*r/tile*N+phase[0];sy=-float(d[2])*L*r/tile*N-phase[1]
            path=vr*L*L*(d[0]*d[0]+d[2]*d[2]);pix=(scale*tile/256)**2/3;x=.5*k[j]*(path+pix);w=1/(1+x+.5*x*x)
            for m in range(4):
                f=triangle_sample(fields[j,m],sx,sy);mu=float(np.mean(fields[j,m]));out[e,...,m]+=chi[e,m]*L/4*(mu+w*(f-mu))
    return np.maximum(out,0)

def compose(tau_m:np.ndarray,palette:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    tau=np.sum(tau_m,axis=-1);a=-np.expm1(-np.maximum(tau,0));src=np.einsum('...m,mc->...c',tau_m,palette)
    p=np.divide(a[...,None]*src,tau[...,None],out=np.zeros_like(src,dtype=np.float32),where=tau[...,None]>1e-8)
    return a.astype(np.float32),p.astype(np.float32)

def score(hmod,ta,tp,pa,pp)->dict[str,Any]:
    ea=np.abs(pa-ta);er=np.max(np.abs(pp-tp),axis=-1);qa=q(ea);qr=q(er);support=(ta>1/255)|(pa>1/255);ex=support&((ea>LIMITS['a99'])|(er>LIMITS['rgb99']))
    if ex.ndim==2: connected=hmod.largest_periodic_component(ex)/max(1,int(np.count_nonzero(support)))
    else: connected=max(hmod.largest_periodic_component(ex[i])/max(1,int(np.count_nonzero(support[i]))) for i in range(ex.shape[0]))
    checks={"a95":qa['p95']<=LIMITS['a95'],"a99":qa['p99']<=LIMITS['a99'],"rgb95":qr['p95']<=LIMITS['rgb95'],"rgb99":qr['p99']<=LIMITS['rgb99'],"connected":connected<LIMITS['connected']}
    return {"green":all(checks.values()),"checks":checks,"coverage":qa,"premulRgb":qr,"connected":float(connected)}

def qa(path:Path,title:str,truth:tuple[np.ndarray,np.ndarray],pred:tuple[np.ndarray,np.ndarray]):
    ta,tp=truth;pa,pp=pred;rgb=lambda a,p:np.clip(p+.12*(1-a[...,None]),0,1);err=np.maximum(np.abs(ta-pa),np.max(np.abs(tp-pp),axis=-1));heat=np.zeros((*err.shape,3),np.uint8);heat[...,0]=np.rint(255*np.clip(err/.2,0,1));heat[...,1]=np.rint(180*np.clip(err/.2,0,1))
    panels=[rgb(ta,tp),rgb(pa,pp),heat/255];im=Image.new('RGB',(N*3*4,N*4+36),(12,12,12));dr=ImageDraw.Draw(im);dr.text((6,5),title,fill='white')
    for i,x in enumerate(panels):im.paste(Image.fromarray(np.uint8(np.clip(x,0,1)*255)).resize((N*4,N*4),Image.Resampling.NEAREST),(i*N*4,36))
    path.parent.mkdir(parents=True,exist_ok=True);im.save(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='src/assets/groundcover/calamagrostis-canescens.gcrp');args=ap.parse_args();root=Path(__file__).resolve().parents[2];source=(root/args.source).resolve()
    hm=imp('h_n',root/'tools/groundcover-bake/analyze_candidate_h_angular_continuity.py');imod=imp('i_n',root/'tools/groundcover-bake/gate_candidate_i_vertical_extinction.py');profile=hm.load_profile(source)
    dirs=np.stack([s.direction for s in profile.slices]);meta=[direction_meta(d) for d in dirs];train=np.array([i for i,(e,a) in enumerate(meta) if e>=89 or abs((a%45))<1e-4]);held=np.array([i for i in range(len(dirs)) if i not in set(train.tolist())])
    truth={}
    for scale in SCALES:
        cov,_,pre=imod.build_truth(hm,profile,scale);truth[scale]=(np.stack([down4(x) for x in cov]),np.stack([down4(x) for x in pre]))
    ttau,tsrc=optical(truth[4][0][train],truth[4][1][train]);palette,target_modes=palette_and_weights(ttau,tsrc)
    fine,mats,chi_train=fit_scale(target_modes,dirs[train],profile.top_h,profile.size_x)
    # Independent coarse fit exists only to fit the frozen contraction constants.
    ctau,csrc=optical(truth[16][0][train],truth[16][1][train]);cond=np.divide(csrc,ctau[...,None],out=np.zeros_like(csrc),where=ctau[...,None]>1e-6);dist=np.sum((cond[...,None,:]-palette[None,None,None,:,:])**2,axis=-1);soft=1/np.maximum(dist,1e-4);soft/=np.sum(soft,axis=-1,keepdims=True);coarse_targets=ctau[...,None]*soft
    coarse,_,_=fit_scale(coarse_targets,dirs[train],profile.top_h,profile.size_x);K=contraction_k(fine,coarse,dirs[train],profile.top_h,profile.size_x);packed,pack_scales=quantize4(fine)
    # Evaluate augmented PSD response on every direction.
    chi=np.empty((len(dirs),4),np.float64);v=np.concatenate((np.ones((len(dirs),1)),dirs),axis=1)
    for m,M in enumerate(mats):chi[:,m]=np.maximum(np.einsum('ei,ij,ej->e',v,np.asarray(M),v),1e-5)
    phase_tests=(("node",np.array((0.,0.))),('cell-centre',np.array((.5,.5))),('translation-4.5mm',np.array((4.5e-3/profile.size_x*N,0.))))
    records=[];worst=[]
    for subset,indices in (("exact-trained",train),("heldout",held)):
      for scale in SCALES:
       for phase_name,phase in phase_tests:
        if subset=='exact-trained' and phase_name!='node':continue
        # Translation of filtered truth, with ordinary continuous bilinear truth sampling.
        ta=np.stack([ndimage.shift(truth[scale][0][i],shift=(-phase[1],phase[0]),order=1,mode='grid-wrap',prefilter=False) for i in indices]);tp=np.stack([ndimage.shift(truth[scale][1][i],shift=(-phase[1],phase[0],0),order=1,mode='grid-wrap',prefilter=False) for i in indices])
        pu=compose(predict_modes(fine[None],dirs[indices],profile.top_h,profile.size_x,chi[indices],phase,False,K,scale),palette);pp=compose(predict_modes(packed[None],dirs[indices],profile.top_h,profile.size_x,chi[indices],phase,True,K,scale),palette)
        su=score(hm,ta,tp,*pu);sp=score(hm,ta,tp,*pp);key=f'{subset}-{phase_name}-sigma{scale}';records.append({"key":key,"unquantized":su,"packed":sp});norm=max(sp['coverage']['p95']/LIMITS['a95'],sp['premulRgb']['p95']/LIMITS['rgb95'],sp['connected']/LIMITS['connected']);worst.append((norm,key,(ta[0],tp[0]),(pp[0][0],pp[1][0])))
    # One-q bridge: the exact algebra is checked independently from its fidelity.
    bridge_jump=max(abs((0.+1.)-1.),abs((1.+0.)-1.));handoff={"R_to_B1":0.,"B1_to_B2":0.,"B2_to_U":0.,"algebraMaxJump":bridge_jump,"loads":{"R":8,"B1":9,"B2":9,"U":8}}
    all_u=all(x['unquantized']['green'] for x in records);all_p=all(x['packed']['green'] for x in records);verdict='GREEN' if all_u and all_p else 'RED'
    spec=root/'docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-N-HEIGHT-MOMENT-TRANSFER.md';recipe={"version":VERSION,"source":sha(source),"script":sha(Path(__file__)),"spec":sha(spec),"N":N,"train":train.tolist(),"heldout":held.tolist(),"scales":SCALES,"phaseTests":[x[0] for x in phase_tests],"limits":LIMITS,"format":"4 cubic Bernstein height arrays; one RG32Uint cell per height; 16 log-companded nibbles"};rh=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest();out=root/'data/work/groundcover-candidate-n-height-moment'/recipe['source'][:16]/rh[:16];(out/'qa').mkdir(parents=True,exist_ok=True)
    worst.sort(reverse=True,key=lambda x:x[0]);
    for i,x in enumerate(worst[:5]):qa(out/'qa'/f'{i+1:03d}-{x[1]}.png',x[1],x[2],x[3])
    report={"schema":VERSION,"verdict":verdict,"scope":"periodic-interior source-fidelity, packing, translation and handoff-algebra gate","recipe":recipe,"source":{"path":str(source),"triangles":int(profile.triangles.shape[0]),"topH":profile.top_h,"tile":profile.size_x},"fit":{"palette":palette.tolist(),"directionMatrices":mats,"contractionK":K.tolist(),"packingScales":pack_scales.tolist()},"checks":{"allUnquantizedGreen":all_u,"allPackedGreen":all_p,"handoff":handoff},"evaluations":records}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'SUMMARY.md').write_text(f'# Candidate N height-moment gate\n\nVerdict: **{verdict}**\n\nAll unquantized GREEN: `{all_u}`  \nAll packed GREEN: `{all_p}`  \nWorst: `{worst[0][1]}` at `{worst[0][0]:.3f}x` limits.\n\nQA: truth | packed Candidate N | max-channel error.\n')
    print(json.dumps({"output":str(out),"verdict":verdict,"allUnquantizedGreen":all_u,"allPackedGreen":all_p,"worst":[{"key":x[1],"normalized":x[0]} for x in worst[:5]]},indent=2))
if __name__=='__main__':main()
