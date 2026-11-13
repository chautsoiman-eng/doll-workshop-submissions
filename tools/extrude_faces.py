"""
「擠出面」加厚：選一側的整片面（例如鏡框背面、鏡腳內側），沿固定方向平移拉出一段距離。
原本的面都不動，正面看形狀完全不變，只在那一側變厚。

每個三角形各自擠出一個小柱體再聯集（彎曲的面整片擠出會自己交叉），
柱體從表面往內 start 開始，避免面重疊。

陳誼緁 T（5cm）的實際用法見 tools/glasses/chen_T_5cm.py。
"""
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from thicken import face_thickness, connected_to_seeds, thin_faces
from trimesh.ray.ray_pyembree import RayMeshIntersector

def lens_info(m, seed):
    c=m.triangles_center; d0=np.linalg.norm(c-np.array(seed),axis=1); st=int(np.argmin(d0))
    near=np.where(d0<8)[0]; t=face_thickness(m,near); n0=m.face_normals[st]
    ok=near[np.isfinite(t)&(t<0.5)&(m.face_normals[near]@n0>0.5)]
    g=connected_to_seeds(m,np.append(ok,st),[c[st]],reach=0.05)
    nrm=(m.face_normals[g]*m.area_faces[g][:,None]).sum(0); nrm/=np.linalg.norm(nrm)
    u=np.cross(nrm,[0,0,1.0]); u/=np.linalg.norm(u); v=np.cross(nrm,u); o=m.vertices[np.unique(m.faces[g])].mean(0)
    poly=unary_union([Polygon([((p-o)@u,(p-o)@v) for p in tri]) for tri in m.triangles[g]]).buffer(0.02).buffer(-0.02)
    return dict(faces=g,nrm=nrm,u=u,v=v,o=o,poly=poly)

def parts(m, frame_seed, lens_seeds, box, rim_reach=0.8):
    """回傳 (鏡框面, 鏡腳面, 鏡片資訊)；鏡片的正反兩面都排除。"""
    Ls=[lens_info(m,s) for s in lens_seeds]
    # 鏡片背面：用正面種子往內打找到的面也要排除 → 直接排除「在鏡片輪廓內、且很薄」的面
    idx,t=thin_faces(m,box); ok=np.isfinite(t)&(t<1.6)
    g=connected_to_seeds(m,idx[ok],[frame_seed])
    c=m.triangles_center
    lensmask=np.zeros(len(g),bool); nearlens=np.zeros(len(g),bool)
    tm=dict(zip(idx.tolist(),t.tolist())); gt=np.array([tm[i] for i in g])
    for L in Ls:
        q=c[g]-L['o']; xy=np.c_[q@L['u'],q@L['v']]
        inside=np.array([L['poly'].buffer(-0.05).contains(Point(p)) for p in xy])
        lensmask|=inside&(gt<0.35)&(np.abs(m.face_normals[g]@L['nrm'])>0.7)
        nearlens|=np.array([L['poly'].buffer(rim_reach).contains(Point(p)) for p in xy])
    keep=~lensmask
    frame=g[keep&nearlens]; temple=g[keep&~nearlens]
    return frame, temple, Ls

def directional_thickness(m, faces, direction):
    """沿 -direction 往內打（從要拉出的那面量到對面），回傳每個面的厚度。"""
    d=np.asarray(direction,float); d/=np.linalg.norm(d)
    r=RayMeshIntersector(m); o=m.triangles_center[faces]-d*1e-4
    loc,ri,_=r.intersects_location(o,np.tile(-d,(len(faces),1)),multiple_hits=False)
    t=np.full(len(faces),np.inf); t[ri]=np.linalg.norm(loc-o[ri],axis=1); return t

def free_space(m, faces, direction):
    d=np.asarray(direction,float); d/=np.linalg.norm(d)
    r=RayMeshIntersector(m); o=m.triangles_center[faces]+d*1e-4
    loc,ri,_=r.intersects_location(o,np.tile(d,(len(faces),1)),multiple_hits=False)
    f=np.full(len(faces),np.inf); f[ri]=np.linalg.norm(loc-o[ri],axis=1); return f


def extrude_patch(m, faces, direction, dist, start=0.02):
    """把 faces 這片面沿 direction 平移擠出 dist：每個三角形各擠出一個小柱體，再全部聯集。
    彎曲的面整片擠出時會自己交叉，逐個三角形做就不會。"""
    d=np.asarray(direction,float); d/=np.linalg.norm(d)
    T=m.triangles[faces]
    dist=np.broadcast_to(np.asarray(dist,float),(len(faces),))[:,None,None]   # 可以每個面不同距離
    lo=T-d*start; hi=T+d*dist
    V=np.concatenate([lo,hi],axis=1).reshape(-1,3)          # 每個柱體 6 個點
    pf=np.array([[0,2,1],[3,4,5],[0,1,4],[0,4,3],[1,2,5],[1,5,4],[2,0,3],[2,3,5]])
    parts=[trimesh.Trimesh(V[i*6:(i+1)*6],pf,process=False) for i in range(len(faces))]
    for q in parts:
        if q.volume<0: q.invert()
    return trimesh.boolean.union(parts,engine='manifold')
