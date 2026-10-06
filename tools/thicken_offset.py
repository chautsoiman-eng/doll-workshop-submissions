"""
「直接改面」版的加厚：把眼鏡上太薄地方的頂點沿法線往外推，不新增任何幾何。

做法:
  1. 跟 thicken.py 一樣，從 seeds 找出眼鏡（連在一起的薄面），量出每個面的厚度 t。
  2. 每個頂點需要往外推 d = (MIN_MM + MARGIN - t) / 2（兩側各推 d，總厚度補足）；
     夠厚的頂點 d = 0。
  3. 推動量沿著表面距離平滑漸變：從需要加厚的地方往外 FALLOFF_MM 內降到 0
     （可以延伸到鏡腳接合方塊、鼻樑架等較厚的部分，不會在接合處留下一圈凹槽）；
     眼鏡和臉/頭髮相接的地方固定不動，附近 BORDER_FALLOFF_MM 內漸變到 0。
  4. 推之前沿法線打射線，如果路徑上會撞到其他表面，就只推到撞點距離的一半。
  5. 頂點數、面數、連接關係都不變，原本的平滑表面保留。
"""
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from trimesh.ray.ray_pyembree import RayMeshIntersector

from thicken import MIN_MM, MARGIN, connected_to_seeds, thin_faces

THIN_MM = 1.6            # 比這個薄的才算「薄眼鏡」
NEAR_MM = 0.6            # 比較厚的面離薄眼鏡多近才算進眼鏡物件
STEP_SMOOTH_ITERS = 20
OBJECT_MAX_MM = 3.0      # 這個厚度以下、跟眼鏡連在一起的都算眼鏡物件
FALLOFF_MM = 1.0         # 推動量往外漸變到 0 的距離
BORDER_FALLOFF_MM = 0.5  # 靠近臉/頭髮交界時漸變到 0 的距離
MAX_FIX_ITERS = 8


def offset_thicken(mesh, box, seeds, ignore_below=0.0, min_mm=MIN_MM, uniform=False):
    idx, t = thin_faces(mesh, box)
    # 眼鏡物件：從 seeds 出發、連在一起、厚度 < OBJECT_MAX_MM 的面
    # （包含鏡框和鏡腳接合的方塊、鼻樑架這些比較厚的部分，它們不用加厚，但推動量可以延伸過去）
    ok = np.isfinite(t) & (t >= ignore_below)
    thin = ok & (t < THIN_MM)
    thin_g = connected_to_seeds(mesh, idx[thin], seeds)
    # 比 THIN_MM 厚的面，只收離薄眼鏡 NEAR_MM 以內的（接合方塊、鼻樑架根部），
    # 不會順著鏡腳一路長到頭髮
    thick = ok & (t >= THIN_MM) & (t < OBJECT_MAX_MM)
    if thick.any() and len(thin_g):
        dd, _ = cKDTree(mesh.triangles_center[thin_g]).query(mesh.triangles_center[idx[thick]])
        thick[np.where(thick)[0][dd > NEAR_MM]] = False
    cand = np.zeros(len(idx), bool)
    cand[np.isin(idx, thin_g)] = True
    cand |= thick
    tmap = dict(zip(idx[cand].tolist(), t[cand].tolist()))
    glasses = connected_to_seeds(mesh, idx[cand], seeds)
    gt = np.array([tmap[i] for i in glasses])

    # 每個頂點取相鄰眼鏡面裡最薄的厚度，算出需要往外推多少
    nv = len(mesh.vertices)
    vt = np.full(nv, np.inf)
    for k in range(3):
        np.minimum.at(vt, mesh.faces[glasses, k], gt)
    gv = np.where(np.isfinite(vt))[0]
    need = np.zeros(nv)
    need[gv] = np.maximum(0.0, (min_mm + MARGIN - vt[gv]) / 2)

    # 眼鏡和臉/頭髮共用的頂點 = 邊界，固定不動
    in_g = np.zeros(len(mesh.faces), bool)
    in_g[glasses] = True
    touches_other = np.zeros(nv, bool)
    for k in range(3):
        touches_other[mesh.faces[~in_g, k]] = True
    border = touches_other & np.isfinite(vt)

    # 只在眼鏡物件上，用「沿著表面的距離」做平滑漸變
    local = {v: i for i, v in enumerate(gv)}
    e = mesh.edges_unique
    e = e[np.isfinite(vt[e]).all(1)]
    w = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    li = np.array([local[v] for v in e[:, 0]])
    lj = np.array([local[v] for v in e[:, 1]])
    graph = coo_matrix((w, (li, lj)), shape=(len(gv), len(gv))).tocsr()

    def geodesic(src_mask):
        src = np.where(src_mask)[0]
        if not len(src):
            return np.full(len(gv), np.inf), np.zeros(len(gv), int)
        dist, _, origin = dijkstra(graph, directed=False, indices=src, min_only=True,
                                   return_predecessors=True)
        return dist, origin

    def smooth01(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    ln = need[gv]
    # 從需要加厚的頂點往外，在 FALLOFF_MM 內平滑降到 0
    dist, origin = geodesic(ln > 1e-6)
    has = np.isfinite(dist) & (origin >= 0)
    ld = np.zeros(len(gv))
    ld[has] = ln[origin[has]] * smooth01(1 - dist[has] / FALLOFF_MM)
    ld = np.maximum(ld, ln)
    if uniform:
        # 整副眼鏡一起往外推同樣的距離（取大部分薄處需要的量），個別更薄的地方再額外多推
        ld = np.maximum(ld, np.percentile(ln[ln > 1e-6], 95) if (ln > 1e-6).any() else 0.0)
    # 靠近臉/頭髮交界的地方，在 BORDER_FALLOFF_MM 內平滑降到 0
    bdist, _ = geodesic(border[gv])
    ld *= smooth01(bdist / BORDER_FALLOFF_MM)
    d = np.zeros(nv)
    d[gv] = ld

    move = np.where(d > 1e-6)[0]
    n = mesh.vertex_normals[move]
    # 推之前檢查會不會撞到別的表面
    rmi = RayMeshIntersector(mesh)
    o = mesh.vertices[move] + n * 1e-4
    loc, ri, _ = rmi.intersects_location(o, n, multiple_hits=False)
    hit = np.full(len(move), np.inf)
    hit[ri] = np.linalg.norm(loc - o[ri], axis=1)
    # 撞到其他表面的頂點有上限；把推動量在眼鏡上做平滑，凹點變成很淺的過渡而不是小坑
    limit = np.minimum(d[move], hit * 0.5)
    pos = np.full(nv, -1)
    pos[move] = np.arange(len(move))
    me = mesh.edges_unique[(pos[mesh.edges_unique] >= 0).all(1)]
    a, b = pos[me[:, 0]], pos[me[:, 1]]
    deg = np.bincount(np.r_[a, b], minlength=len(move)).astype(float)
    step = limit.copy()
    for _ in range(STEP_SMOOTH_ITERS):
        acc = np.zeros(len(move))
        np.add.at(acc, a, step[b])
        np.add.at(acc, b, step[a])
        avg = np.divide(acc, deg, out=step.copy(), where=deg > 0)
        step = np.minimum(limit, 0.5 * (step + avg))

    # 推完如果還是有面互相穿插，就把那附近的推動量減半，重複到沒有為止
    # 檢查範圍：被推動的頂點周圍 0.5mm 內的所有面（包含旁邊沒動的面）
    lo = mesh.vertices[move].min(0) - 0.5
    hi = mesh.vertices[move].max(0) + 0.5
    fc = mesh.triangles_center
    moved_faces = np.where(((fc > lo) & (fc < hi)).all(1))[0]
    fixes = 0
    for _ in range(MAX_FIX_ITERS):
        out = mesh.copy()
        out.vertices[move] += n * step[:, None]
        # STL 存成 float32，用存檔後的精度檢查，避免存檔四捨五入後才穿插
        out.vertices = out.vertices.astype(np.float32).astype(np.float64)
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
