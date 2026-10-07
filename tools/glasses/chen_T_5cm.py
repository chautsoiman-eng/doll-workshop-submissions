"""
陳誼緁 T（5cm）的墨鏡加厚（學員確認的做法）：
  - 鏡框：背面（朝臉）整片沿鏡片法線往後拉出 FRAME mm；只拉鏡框圈，不拉鉸鏈/鏡腳根部
  - 鏡腳：內側（朝頭）整片沿左右方向往內拉出 TEMPLE mm
  - 鏡片：正面不動，背面往內拉 LENS mm，不超過鏡框背面、不碰到眼球
用法（在 repo 根目錄）:  python tools/glasses/chen_T_5cm.py 0.4 0 0.5   # 鏡框 鏡腳 鏡片背面（學員最後決定鏡腳不動 → 0）
（鏡腳是整根桿子往內掃，不只拉一部分）
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
from shapely.geometry import Point
FS=[-0.14,-9.36,31.45]; LS=[[2.07,-10.3,31.81],[-4.16,-8.61,29.42]]; BOX=(-12,12,26,40)
o=trimesh.load(src); frame,temple,Ls=parts(o,FS,LS,BOX)
back=-np.mean([L['nrm'] for L in Ls],axis=0); back/=np.linalg.norm(back)
head=o.vertices[o.vertices[:,2]>28].mean(0)
# 鏡框背面：只取「真正的鏡框圈」—— 離鏡片邊緣 0.6mm 內、在鏡片平面後方 0.6mm 內的背面；
# 鉸鏈、鏡腳根部比較後面，不要拉（拉了會在鏡框和鏡腳之間長出一塊斜面）。碰到臉沒關係，不限制。
fb=frame[o.face_normals[frame]@back>0.3]; fc=o.triangles_center[fb]
rim=np.zeros(len(fb),bool)
for L in Ls:
    q=fc-L['o']; xy=np.c_[q@L['u'],q@L['v']]
    near=np.array([L['poly'].exterior.distance(Point(p))<0.6 for p in xy])
    rim|=near&((q@L['nrm'])>-0.6)
print(f'frame back faces {len(fb)}  rim {rim.sum()}  skipped (hinge/temple) {np.sum(~rim)}')
solids=[extrude_patch(o,fb[rim],back,FRAME)]; dirs=[]
for side in (-1,1):
    tt=temple[np.sign(o.triangles_center[temple][:,0])==side]
    inward=np.array([-side,0,0.0])
    if TEMPLE > 0:
        # 整支鏡腳（整根桿子）沿左右方向往內掃過 TEMPLE mm：每個面都擠出小柱體再聯集，
        # 等於整個內側面平移出去，上下兩邊的牆剛好沿著鏡腳的輪廓，從上面看不會有鋸齒。
        # 只取鏡腳桿子本身（離桿子中心線 0.6mm 內），不含鉸鏈和末端黏在一起的頭髮。
        p=o.triangles_center[tt]
        core=p[(np.abs(p[:,0])>4)&((p[:,1]>-11)&(p[:,1]<-8) if side>0 else (p[:,1]>-7)&(p[:,1]<-3))]
        mu=core.mean(0); ax=np.linalg.svd(core-mu)[2][0]
        q=p-mu; r=np.linalg.norm(q-np.outer(q@ax,ax),axis=1)
        bar=tt[(r<0.6)&(np.abs(p[:,0])>4)]
        # 掃過的體積 = 原本 + 「朝內那半邊」每個面擠出的柱體
        bar_all=bar; bar=bar[o.face_normals[bar]@inward>0.15]   # 太貼近側面的柱體太扁，存檔時會壞
        print(f'temple {side}: faces {len(tt)}  inner half of bar {len(bar)}')
        solids.append(extrude_patch(o,bar,inward,TEMPLE)); dirs.append((bar_all,inward))
out=o
for s in solids: out=trimesh.boolean.union([out,s],engine='manifold')
out=max(out.split(only_watertight=False),key=lambda b:len(b.faces))
# 鏡片：正面不動，只把背面往內拉 LENS mm；不超過鏡框背面，離眼球近的地方停在眼球前 0.05mm
from shapely.geometry import Point
from extrude_faces import lens_info, free_space
LENS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
cc=o.triangles_center
for sd in (LS if LENS > 0 else []):
    L=lens_info(o,sd); q=cc-L['o']; nrm=L['nrm']
    cand=np.where((np.abs(q@nrm)<1.0)&(o.face_normals@nrm<-0.7))[0]
    inner=L['poly'].buffer(-0.05)
    bf=cand[np.array([inner.contains(Point(p)) for p in np.c_[q[cand]@L['u'],q[cand]@L['v']]])]
    gap=free_space(o,bf,-nrm)
    dist=np.clip(np.minimum(LENS,gap-0.05),0.0,None)
    bf,dist=bf[dist>0.01],dist[dist>0.01]
    out=trimesh.boolean.union([out,extrude_patch(o,bf,-nrm,dist)],engine='manifold')
    print(f'lens back pull: faces {len(bf)}  full {np.mean(dist>=LENS-1e-6)*100:.0f}%  min {dist.min():.2f}')
out=max(out.split(only_watertight=False),key=lambda b:len(b.faces))
# 布林聯集會留下極小的碎邊，STL 存成 float32 時會黏在一起變破洞 → 先用 manifold 把 < 0.002mm 的邊合併掉
import manifold3d
mf=manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(out.vertices,np.float32),tri_verts=np.asarray(out.faces,np.uint32))).simplify(0.002).to_mesh()
out=trimesh.Trimesh(mf.vert_properties[:,:3],mf.tri_verts); out.export(dst); m=trimesh.load(dst)
print('saved watertight', m.is_watertight)
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
