"""
「直接改面」版的加厚：把眼鏡上太薄地方的頂點沿法線往外推，不新增任何幾何。

做法:
  1. 跟 thicken.py 一樣，從 seeds 找出眼鏡（連在一起的薄面），量出每個面的厚度 t。
  2. 每個頂點需要往外推 d = (MIN_MM + MARGIN - t) / 2（兩側各推 d，總厚度補足）；
     夠厚的頂點 d = 0。
  3. d 在眼鏡的頂點之間平滑擴散，避免出現階梯；眼鏡和臉/頭髮相接的頂點固定 d = 0。
  4. 推之前沿法線打射線，如果路徑上會撞到其他表面，就只推到撞點距離的一半。
  5. 頂點數、面數、連接關係都不變，原本的平滑表面保留。
"""
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from thicken import MIN_MM, MARGIN, connected_to_seeds, thin_faces

SMOOTH_ITERS = 15
MAX_FIX_ITERS = 8


def offset_thicken(mesh, box, seeds, ignore_below=0.0, min_mm=MIN_MM):
    idx, t = thin_faces(mesh, box)
    cand = np.isfinite(t) & (t < 1.6) & (t >= ignore_below)
    tmap = dict(zip(idx[cand].tolist(), t[cand].tolist()))
    glasses = connected_to_seeds(mesh, idx[cand], seeds)
    gt = np.array([tmap[i] for i in glasses])

    # 每個頂點取相鄰眼鏡面裡最薄的厚度
    nv = len(mesh.vertices)
    vt = np.full(nv, np.inf)
    for k in range(3):
        np.minimum.at(vt, mesh.faces[glasses, k], gt)
    gv = np.where(np.isfinite(vt))[0]
    need = np.zeros(nv)
    need[gv] = np.maximum(0.0, (min_mm + MARGIN - vt[gv]) / 2)

    # 和非眼鏡的面共用的頂點 = 邊界，固定不動
    in_g = np.zeros(len(mesh.faces), bool)
    in_g[glasses] = True
    touches_other = np.zeros(nv, bool)
    for k in range(3):
        touches_other[mesh.faces[~in_g, k]] = True
    border = touches_other & np.isfinite(vt)

    # 在眼鏡頂點之間平滑擴散：保留需要的最小值，往周圍漸漸變小
    edges = mesh.edges_unique
    ge = edges[np.isfinite(vt[edges]).all(1)]
    deg = np.bincount(ge.ravel(), minlength=nv).astype(float)
    d = need.copy()
    for _ in range(SMOOTH_ITERS):
        s = np.zeros(nv)
        np.add.at(s, ge[:, 0], d[ge[:, 1]])
        np.add.at(s, ge[:, 1], d[ge[:, 0]])
        avg = np.divide(s, deg, out=np.zeros(nv), where=deg > 0)
        d = np.maximum(need, 0.5 * (d + avg))
        d[border] = 0.0

    move = np.where(d > 1e-6)[0]
    n = mesh.vertex_normals[move]
    # 推之前檢查會不會撞到別的表面
    rmi = RayMeshIntersector(mesh)
    o = mesh.vertices[move] + n * 1e-4
    loc, ri, _ = rmi.intersects_location(o, n, multiple_hits=False)
    hit = np.full(len(move), np.inf)
    hit[ri] = np.linalg.norm(loc - o[ri], axis=1)
    step = np.minimum(d[move], hit * 0.5)

    # 推完如果還是有面互相穿插，就把那附近的推動量減半，重複到沒有為止
    moved_faces = np.where(np.isin(mesh.faces, move).any(1))[0]
    fixes = 0
    for _ in range(MAX_FIX_ITERS):
        out = mesh.copy()
        out.vertices[move] += n * step[:, None]
        bad_v = crossing_vertices(out, moved_faces)
        if not len(bad_v):
            break
        fixes += 1
        ring = np.unique(mesh.edges_unique[np.isin(mesh.edges_unique, bad_v).any(1)])
        step[np.isin(move, ring)] *= 0.5
    report = {
        "fix_rounds": fixes,
        "crossing_vertices_left": int(len(crossing_vertices(out, moved_faces))),
        "glasses_faces": int(len(glasses)),
        "moved_vertices": int(len(move)),
        "max_push_mm": round(float(step.max()), 3) if len(step) else 0.0,
        "limited_by_collision": int((hit * 0.5 < d[move]).sum()),
        "watertight": bool(out.is_watertight),
    }
    return out, report


def crossing_vertices(mesh, faces):
    """faces 範圍內，穿過別的三角形的邊（= 面互相穿插）的端點。"""
    e = np.unique(np.sort(mesh.faces[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    a, b = mesh.vertices[e[:, 0]], mesh.vertices[e[:, 1]]
    v = b - a
    L = np.linalg.norm(v, axis=1)
    ok = L > 1e-9
    e, a, v, L = e[ok], a[ok], v[ok], L[ok]
    loc, ri, ti = RayMeshIntersector(mesh).intersects_location(a, v / L[:, None], multiple_hits=True)
    dist = np.linalg.norm(loc - a[ri], axis=1)
    inside = (dist > 1e-5) & (dist < L[ri] - 1e-5)
    # 撞到的三角形如果包含這條邊的端點，是相鄰面，不算
    tf = mesh.faces[ti]
    shares = (tf == e[ri, 0][:, None]).any(1) | (tf == e[ri, 1][:, None]).any(1)
    bad = inside & ~shares
    return np.unique(e[np.unique(ri[bad])].ravel())


def crossing_edges(mesh, faces):
    return len(crossing_vertices(mesh, faces))
