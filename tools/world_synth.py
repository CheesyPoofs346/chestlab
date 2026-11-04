import sys,json,time; import os; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from chestlab.live import crop_minimap
from chestlab import worldmap as W
full=cv2.imread('assets/gg/full_z4.png'); wm=W.WorldMap(); tr=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'world_truth_z4.json')))
S=W.Z_REAL*16; T=np.array([-19281.0,-7376.0]); BIAS=np.array(W.WorldFitter.BIAS_PX)
def to_mm(px,py,jx=0,jy=0):
    """minimap (231) rendered from the z4 map, player at (px,py)"""
    c=115.0; M=np.float32([[1/S,0,c-px/S],[0,1/S,c-py/S]])
    return cv2.warpAffine(full,M,(231,231),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT)
# ---- fit a colour grade gg -> real minimap from the aligned real frames (poly deg 2 in Lab)
X=[];Y=[]
for k,z in sorted(tr.items()):
    n=int(k.split('shot')[1].split('.')[0])
    if n in (23,25,26,27): continue
    mm=crop_minimap(cv2.imread(k)); g=to_mm(z[0]+BIAS[0],z[1]+BIAS[1])
    gl=cv2.cvtColor(cv2.GaussianBlur(g,(0,0),1.0),cv2.COLOR_BGR2Lab).reshape(-1,3).astype(np.float32); ml=cv2.cvtColor(mm,cv2.COLOR_BGR2Lab).reshape(-1,3).astype(np.float32)
    hsv=cv2.cvtColor(mm,cv2.COLOR_BGR2HSV).reshape(-1,3); ok=~(((hsv[:,1]<45)&(hsv[:,2]>190))|((hsv[:,0]>=158)&(hsv[:,1]>110)))
    ok&=(np.arange(231*231)//231>50)&(g.reshape(-1,3).max(1)>6)
    X.append(gl[ok][::7]);Y.append(ml[ok][::7])
X=np.vstack(X);Y=np.vstack(Y)
def feats(l): a=l/255.0; return np.column_stack([np.ones(len(a)),a,a**2,a[:,0]*a[:,1],a[:,0]*a[:,2],a[:,1]*a[:,2]])
Wt=np.linalg.lstsq(feats(X).T@feats(X)+1e-3*np.eye(10),feats(X).T@Y,rcond=None)[0]
print("grade fit rms Lab",np.sqrt(((feats(X)@Wt-Y)**2).mean(0)).round(1))
def grade(bgr):
    l=cv2.cvtColor(bgr,cv2.COLOR_BGR2Lab).reshape(-1,3).astype(np.float32); o=np.clip(feats(l)@Wt,0,255).astype(np.uint8).reshape(bgr.shape)
    return cv2.cvtColor(o,cv2.COLOR_Lab2BGR)
rng=np.random.default_rng(7); H4=full.shape[0]
fit=W.WorldFitter(wm); fit.BIAS_PX=(0.0,0.0)
res=dict(ok=0,refused=0,wrong=0,n=0); errs=[]; wrong=[]
t0=time.time()
for i in range(160):
    while True:
        px,py=rng.uniform(150,H4-150,2)
        if wm.valid4[int(py),int(px)] and wm.valid4[int(py)-120:int(py)+120:20,int(px)-120:int(px)+120:20].all(): break
    mm=grade(to_mm(px,py)); mm=cv2.GaussianBlur(mm,(0,0),rng.uniform(0.3,0.9))
    mm=np.clip(mm.astype(np.float32)+rng.normal(0,rng.uniform(1,4),mm.shape),0,255).astype(np.uint8)
    # HUD: route line, pin, arrow, frame, top icons
    a=rng.uniform(0,np.pi); cv2.line(mm,(115,115),(int(115+200*np.cos(a)),int(115+200*np.sin(a))),(255,255,255),2)
    cv2.circle(mm,(int(rng.integers(30,200)),int(rng.integers(60,200))),9,(255,255,255),-1)
    cv2.rectangle(mm,(0,0),(230,26),(60,60,60),-1); cv2.rectangle(mm,(0,0),(230,230),(220,100,30),4)
    if rng.random()<0.3:            # storm over a random half plane (or all)
        ang=rng.uniform(0,2*np.pi); yy,xx=np.mgrid[0:231,0:231]; side=((xx-115)*np.cos(ang)+(yy-115)*np.sin(ang))>rng.uniform(-90,200)
        st=0.4*mm.astype(np.float32)+np.array([150,30,120],np.float32); mm=np.where(side[...,None],st,mm).astype(np.uint8)
    f=fit.acquire(mm); res['n']+=1
    if f is None: res['refused']+=1; continue
    tru=W.px_to_uv([px,py],4); e=np.hypot(*(np.array([f.u,f.v])-tru))*0.0875
    if e<4: res['ok']+=1; errs.append(e)
    else: res['wrong']+=1; wrong.append((round(px),round(py),round(e),round(f.score,2)))
print(res,"in %.0fs"%(time.time()-t0)); print("err m median %.2f p90 %.2f max %.2f"%(np.median(errs),np.percentile(errs,90),max(errs)))
print("wrong:",wrong[:12])
