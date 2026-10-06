"""
陳誼緁 T（5cm）的墨鏡加厚（學員確認的做法）：
  - 鏡框：背面（朝臉）整片沿鏡片法線往後拉出 FRAME mm
  - 鏡腳：內側（朝頭）整片沿左右方向往內拉出 TEMPLE mm
  - 鏡片：正面整片往外拉 0.2mm、背面整片往內拉 0.3mm（保留原本彎度）
用法（在 repo 根目錄）:  python tools/glasses/chen_T_5cm.py 0.4 0.45
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh, pymeshfix
from extrude_faces import parts, extrude_patch
from thicken_offset import crossing_edges
from thicken_details import in_box
from trimesh.ray.ray_pyembree import RayMeshIntersector
FRAME, TEMPLE = float(sys.argv[1]), float(sys.argv[2])
src='submissions/LA1900226_陳誼緁/LA1900226_陳誼緁_T_5cm_flat.stl'; dst=src[:-4]+'_thick.stl'
FS=[-0.14,-9.36,31.45]; LS=[[2.07,-10.3,31.81],[-4.16,-8.61,29.42]]; BOX=(-12,12,26,40)
o=trimesh.load(src); frame,temple,Ls=parts(o,FS,LS,BOX)
back=-np.mean([L['nrm'] for L in Ls],axis=0); back/=np.linalg.norm(back)
head=o.vertices[o.vertices[:,2]>28].mean(0)
solids=[extrude_patch(o,frame[o.face_normals[frame]@back>0.3],back,FRAME)]; dirs=[]
for side in (-1,1):
    tt=temple[np.sign(o.triangles_center[temple][:,0])==side]
    inward=np.array([-side,0,0.0]); dirs.append((tt,inward))
    solids.append(extrude_patch(o,tt[o.face_normals[tt]@inward>0.3],inward,TEMPLE))
out=o
for s in solids: out=trimesh.boolean.union([out,s],engine='manifold')
out=max(out.split(only_watertight=False),key=lambda b:len(b.faces))
# 鏡片：正面整片沿鏡片法線往外拉 0.2mm、背面整片往內拉 0.3mm（保留原本的彎度）
from shapely.geometry import Point
from extrude_faces import lens_info
cc=o.triangles_center
for sd in LS:
    L=lens_info(o,sd); front=L['faces']
    q=cc-L['o']; cand=np.where((np.abs(q@L['nrm'])<1.0)&(o.face_normals@L['nrm']<-0.7))[0]
    xy=np.c_[q[cand]@L['u'],q[cand]@L['v']]; inner=L['poly'].buffer(-0.05)
    back_f=cand[np.array([inner.contains(Point(p)) for p in xy])]
    out=trimesh.boolean.union([out,extrude_patch(o,front,L['nrm'],0.2)],engine='manifold')
    out=trimesh.boolean.union([out,extrude_patch(o,back_f,-L['nrm'],0.3)],engine='manifold')
out=max(out.split(only_watertight=False),key=lambda b:len(b.faces)); out.export(dst); m=trimesh.load(dst)
if not m.is_watertight:
    v,f=pymeshfix.clean_from_arrays(np.ascontiguousarray(m.vertices,dtype=np.float64),np.ascontiguousarray(m.faces,dtype=np.int32)); m=trimesh.Trimesh(v,f); m.export(dst); m=trimesh.load(dst)
def depth(mm,pts,d):
    r=RayMeshIntersector(mm); orig=pts-d*0.05
    loc,ri,_=r.intersects_location(orig,np.tile(d,(len(orig),1)),multiple_hits=True); out=np.full(len(orig),np.nan)
    for i in range(len(orig)):
        dd=np.sort(np.linalg.norm(loc[ri==i]-orig[i],axis=1))
        if len(dd)>=2: out[i]=dd[1]-dd[0]
    return out
pts=o.triangles_center[frame[o.face_normals[frame]@back<-0.3]]
b=depth(m,pts,back); print(f'frame pull {FRAME}: depth after med {np.nanmedian(b):.2f} p10 {np.nanpercentile(b,10):.2f}')
for side,(tt,inward) in zip((-1,1),dirs):
    b=depth(m,o.triangles_center[tt[o.face_normals[tt]@inward<-0.3]],inward); print(f'temple {side} pull {TEMPLE}: after med {np.nanmedian(b):.2f}')
print('wt',m.is_watertight,'bodies',len(m.split(only_watertight=False)),'faces',len(o.faces),'->',len(m.faces),'crossings orig',crossing_edges(o,in_box(o,BOX,1)),'new',crossing_edges(m,in_box(m,BOX,1)))
