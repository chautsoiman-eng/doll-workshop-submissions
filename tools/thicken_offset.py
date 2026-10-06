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
  push_dir（例如 [0, -1, 0] = 往正面）：單面推。法線朝這個方向的面推足整個不足量，
  背對的面不動，朝兩側的面照常兩面各推一半。用在背面貼著其他東西、推不動的薄板
  （例如貼在眼睛前面的墨鏡鏡片）。
  push_dir = "auto"：自動判斷。每個薄處看兩面前方的空間：兩面都很充足就各推一半，
  否則空間比較小的那面不動、由另一面推足。方向不固定（例如歪著的墨鏡）時用這個。
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.sparse.csgraph import dijkstra
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

from thicken import MIN_MM, MARGIN, connected_to_seeds, face_thickness, thin_faces

THIN_MM = 1.6            # 比這個薄的才算「薄眼鏡」
NEAR_MM = 0.6            # 比較厚的面離薄眼鏡多近才算進眼鏡物件
STEP_SMOOTH_ITERS = 20
OBJECT_MAX_MM = 3.0      # 這個厚度以下、跟眼鏡連在一起的都算眼鏡物件
FALLOFF_MM = 1.0         # 推動量往外漸變到 0 的距離
BORDER_FALLOFF_MM = 0.5  # 靠近臉/頭髮交界時漸變到 0 的距離
MAX_FIX_ITERS = 20
ROOMY_MM = 2.0           # push_dir="auto"：兩面前方都超過這個空間才兩面各推一半


def offset_thicken(mesh, box, seeds, ignore_below=0.0, min_mm=MIN_MM, uniform=False, push_dir=None, debug=None):
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
    if isinstance(push_dir, str) and push_dir == "auto":
        # 自動單面推：看每個薄處兩面前方的空間，推不動的那面不推，由另一面推足
        vneed, blocked = auto_side_need(mesh, glasses, gt, min_mm)
        need[gv] = vneed[gv]
    elif push_dir is None:
        # 兩面各推一半
        need[gv] = np.maximum(0.0, (min_mm + MARGIN - vt[gv]) / 2)
    else:
        # 單面推：只推法線朝 push_dir 的那一面，而且推足整個不足量
        # （例如貼在眼睛前面的墨鏡鏡片：背面不動，只把正面往外推）
        pd = np.asarray(push_dir, float)
        pd /= np.linalg.norm(pd)
        facing = mesh.vertex_normals[gv] @ pd
        full = np.maximum(0.0, min_mm + MARGIN - vt[gv])
        w_front = np.clip((facing - 0.2) / 0.5, 0.0, 1.0)    # 朝 push_dir：推足整個不足量
        w_back = np.clip((-facing - 0.2) / 0.5, 0.0, 1.0)    # 背對 push_dir：不動
        # 朝兩側的面（例如鏡腳）照常兩面各推一半
        need[gv] = np.maximum(full / 2 * (1 - w_back), full * w_front)

    # 眼鏡和臉/頭髮共用的頂點 = 邊界，固定不動
    in_g = np.zeros(len(mesh.faces), bool)
    in_g[glasses] = True
    touches_other = np.zeros(nv, bool)
    for k in range(3):
        touches_other[mesh.faces[~in_g, k]] = True
    border = touches_other & np.isfinite(vt)
    if isinstance(push_dir, str) and push_dir == "auto":
        # 不動的那面（例如鏡片對著眼睛的背面）也固定，推動量不會漸變過去
        border |= blocked

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
    if debug is not None:
        debug.update(need=need.copy(), border=border.copy(), glasses=glasses)
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
    if debug is not None:
        debug['d'] = d.copy()

    move = np.where(d > 1e-6)[0]
    n = mesh.vertex_normals[move]
    # 推之前檢查會不會撞到別的表面
    rmi = RayMeshIntersector(mesh)
    o = mesh.vertices[move] + n * 1e-4
    loc, ri, _ = rmi.intersects_location(o, n, multiple_hits=False)
    hit = np.full(len(move), np.inf)
    hit[ri] = np.linalg.norm(loc - o[ri], axis=1)
    # 撞到其他表面的頂點有上限。只平滑「被撞到而減少的量」：
    # 減少量往周圍擴散、漸漸變小，凹點變成很淺的過渡；沒撞到的地方照樣推足
    limit = np.minimum(d[move], hit * 0.5)
    if debug is not None:
        debug.update(move=move, hit=hit.copy(), limit=limit.copy())
    pos = np.full(nv, -1)
    pos[move] = np.arange(len(move))
    me = mesh.edges_unique[(pos[mesh.edges_unique] >= 0).all(1)]
    a, b = pos[me[:, 0]], pos[me[:, 1]]
    deg = np.bincount(np.r_[a, b], minlength=len(move)).astype(float)
    red = d[move] - limit
    for _ in range(STEP_SMOOTH_ITERS):
        acc = np.zeros(len(move))
        np.add.at(acc, a, red[b])
        np.add.at(acc, b, red[a])
        avg = np.divide(acc, deg, out=red.copy(), where=deg > 0)
        red = np.maximum(red, 0.5 * (red + avg))
    step = np.clip(d[move] - red, 0.0, limit)

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
        ring = np.unique(mesh.edges_unique[np.isin(mesh.edges_unique, ring).any(1)])  # 兩圈鄰居
        step[np.isin(move, ring)] *= 0.5
    if debug is not None:
        debug['step'] = step.copy()
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


def auto_side_need(mesh, faces, t, min_mm):
    """每個面需要往外推多少：兩面前方的空間不夠時，由空間夠的那一面多推。

    D = 不足量。兩面前方空間都超過 2D：各推 D/2；
    否則空間比較小的那面不推，空間大的那面推 D（最多推到前方空間的一半）。
    回傳每個頂點的推動量（取相鄰面的最大值），以及要固定不動的頂點（不推的那面）。
    """
    rmi = RayMeshIntersector(mesh)
    c = mesh.triangles_center[faces]
    n = mesh.face_normals[faces]
    D = np.maximum(0.0, min_mm + MARGIN - t)
    # 往內打：找到對面的面
    loc, ri, ti = rmi.intersects_location(c - n * 1e-4, -n, multiple_hits=False)
    opp = np.full(len(faces), -1)
    opp[ri] = ti

    def free(fids):
        cc = mesh.triangles_center[fids]
        nn = mesh.face_normals[fids]
        o = cc + nn * 1e-4
        lc, r, _ = rmi.intersects_location(o, nn, multiple_hits=False)
        f = np.full(len(fids), np.inf)
        f[r] = np.linalg.norm(lc - o[r], axis=1)
        return f

    free_self = free(faces)
    free_opp = np.full(len(faces), np.inf)
    has = opp >= 0
    free_opp[has] = free(opp[has])
    # 兩面前方空間都充足（超過不足量的兩倍、也超過 ROOMY_MM）：各推一半
    # 否則：空間比較小的那面不動（例如鏡片對著眼睛的背面），空間大的那面推足
    roomy = (free_self > np.maximum(2 * D, ROOMY_MM)) & (free_opp > np.maximum(2 * D, ROOMY_MM))
    mine = np.where(roomy, D / 2, np.where(free_self >= free_opp, D, 0.0))
    mine = np.minimum(mine, 0.5 * free_self)
    vneed = np.zeros(len(mesh.vertices))
    for k in range(3):
        np.maximum.at(vneed, mesh.faces[faces, k], mine)
    # 不動的面：需要加厚、但被分配到 0（空間比較小的那面）；它的頂點如果沒有別的面要推，就固定
    tight = np.zeros(len(mesh.vertices), bool)
    for k in range(3):
        tight[mesh.faces[faces[(D > 0) & (mine == 0)], k]] = True
    blocked = tight & (vneed == 0)
    # 單面推的薄板（例如鏡片正面）和夠厚的部分（鏡框）相接的那一圈也固定，
    # 讓薄板中間往外凸、邊緣收回到鏡框，不會穿過鏡框內壁
    one_sided = np.zeros(len(mesh.vertices), bool)
    thick = np.zeros(len(mesh.vertices), bool)
    for k in range(3):
        one_sided[mesh.faces[faces[(mine > 0) & ~roomy], k]] = True
        thick[mesh.faces[faces[D == 0], k]] = True
    blocked |= one_sided & thick
    return vneed, blocked


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


def refine_faces(mesh, faces):
    """把指定的面切成 4 塊；共用到被切開邊的鄰面也跟著切成 2~3 塊，所以不會產生裂縫。

    回傳新的 mesh（原本的頂點編號不變，新頂點接在後面）。
    """
    faces = np.asarray(faces)
    F = mesh.faces
    split_edges = np.sort(F[faces][:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    split_edges = np.unique(split_edges, axis=0)
    nv = len(mesh.vertices)
    mid_index = {tuple(e): nv + i for i, e in enumerate(split_edges)}
    new_v = (mesh.vertices[split_edges[:, 0]] + mesh.vertices[split_edges[:, 1]]) / 2

    def mid(a, b):
        return mid_index.get((a, b) if a < b else (b, a))

    out = []
    for tri in F:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ms = [mid(a, b), mid(b, c), mid(c, a)]
        k = sum(m is not None for m in ms)
        if k == 0:
            out.append((a, b, c))
            continue
        # 轉到固定的樣式（保持原本的方向）
        for _ in range(3):
            if k == 3 or (k == 1 and ms[0] is not None) or (k == 2 and ms[0] is not None and ms[1] is not None):
                break
            a, b, c = b, c, a
            ms = ms[1:] + ms[:1]
        mab, mbc, mca = ms
        if k == 3:
            out += [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]
        elif k == 1:
            out += [(a, mab, c), (mab, b, c)]
        else:  # k == 2：ab、bc 被切
            out += [(mab, b, mbc), (a, mab, mbc), (a, mbc, c)]
    return trimesh.Trimesh(np.vstack([mesh.vertices, new_v]), np.array(out), process=False)


def refine_thin(mesh, box, seeds, ignore_below=0.0, min_mm=MIN_MM, max_edge=0.3, rounds=3):
    """部件上比 min_mm 薄、而且邊長超過 max_edge 的粗三角形切細（例如很粗的鏡片網格），
    讓中間有頂點可以推。回傳 (新 mesh, 新增的面數)。"""
    added = 0
    for _ in range(rounds):
        idx, t = thin_faces(mesh, box)
        ok = np.isfinite(t) & (t >= ignore_below)
        g = connected_to_seeds(mesh, idx[ok & (t < THIN_MM)], seeds)
        tm = dict(zip(idx.tolist(), t.tolist()))
        gt = np.array([tm[i] for i in g])
        tri = mesh.vertices[mesh.faces[g]]
        longest = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(1)
        coarse = g[(gt < min_mm) & (longest > max_edge)]
        if not len(coarse):
            break
        n0 = len(mesh.faces)
        mesh = refine_faces(mesh, coarse)
        added += len(mesh.faces) - n0
    return mesh, added



def plate_thicken(mesh, seed, min_mm=MIN_MM, plate_max=0.5, inset_mm=0.12, gap_mm=0.03):
    """「平板模式」：替換太薄的平板（例如貼在眼睛前面的墨鏡鏡片），表面保證平整。

    直接推頂點的話，粗網格的鏡片會被推得皺皺的，所以改成：
      1. 從 seed（點在鏡片「正面」）找出正面：連在一起、比 plate_max 薄、朝向差不多的面。
      2. 把正面投影到鏡片平面，得到鏡片輪廓，往內縮 inset_mm（讓鏡框露出來）。
      3. 做一片厚 min_mm + MARGIN、兩面平整的新板子，背面貼在原本鏡片背面往前 gap_mm 的地方
         （避免兩個面完全重疊），再跟模型布林聯集。原本的面都不動。
    回傳 (新 mesh, 報告)。
    """
    c = mesh.triangles_center
    d0 = np.linalg.norm(c - np.asarray(seed), axis=1)
    start = int(np.argmin(d0))
    near = np.where(d0 < 8.0)[0]
    t = face_thickness(mesh, near)
    n0 = mesh.face_normals[start]
    ok = near[np.isfinite(t) & (t < plate_max) & (mesh.face_normals[near] @ n0 > 0.5)]
    g = connected_to_seeds(mesh, np.append(ok, start), [c[start]], reach=0.05)
    tm = dict(zip(near.tolist(), t.tolist()))
    tmed = float(np.nanmedian([tm.get(i, np.nan) for i in g]))

    nrm = (mesh.face_normals[g] * mesh.area_faces[g][:, None]).sum(0)
    nrm /= np.linalg.norm(nrm)
    u = np.cross(nrm, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(nrm, u)
    pts = mesh.vertices[np.unique(mesh.faces[g])]
    o = pts.mean(0)
    outline = unary_union([Polygon([((p - o) @ u, (p - o) @ v) for p in tri]) for tri in mesh.triangles[g]])
    outline = outline.buffer(0.02).buffer(-0.02 - inset_mm)
    if outline.geom_type == "MultiPolygon":
        outline = max(outline.geoms, key=lambda q: q.area)
    back = float(np.median((pts - o) @ nrm)) - tmed + gap_mm
    thick = min_mm + MARGIN
    plate = trimesh.creation.extrude_polygon(outline, thick)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = u, v, nrm, o + nrm * back
    plate.apply_transform(T)
    out = trimesh.boolean.union([mesh, plate], engine="manifold")
    report = {"plate_area_mm2": round(float(outline.area), 1), "thickness_before_mm": round(tmed, 2),
              "push_mm": round(thick + gap_mm - tmed, 3), "watertight": bool(out.is_watertight)}
    return out, report
