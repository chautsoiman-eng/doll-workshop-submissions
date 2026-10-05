"""
把指定範圍內（例如眼鏡）太薄的地方加厚到 MIN_MM 以上，其他地方不動。

做法:
  1. 在範圍內的每個三角面，沿著表面往內打射線，量出局部壁厚 t。
  2. 對 t < MIN_MM 的面，在表面撒點，每個點給一個半徑 r = (MIN_MM + MARGIN - t) / 2。
     （兩側各往外長 r，總厚度就補到 MIN_MM + MARGIN）
  3. 用體素格子算出「離這些點不超過 r」的區域，marching cubes 轉成封閉網格，
     再跟原模型布林聯集成一個實體。已經夠厚的地方沒有撒點，所以不會變。

  只處理「從 seeds 出發、連在一起的薄面」（例如從鼻樑架和兩邊鏡腳出發），
  所以眉毛、髮尾、眼皮這些剛好在範圍內但跟眼鏡沒連在一起的薄東西不會被改到。
  比 ignore_below 還薄的面（例如模型本身的眼皮薄膜）也不處理。

用法（給 Python 呼叫）:
    thicken(mesh, box=(x0, x1, z0, z1), seeds=[(x, y, z), ...], ignore_below=0.0)
"""
import networkx as nx
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from trimesh.ray.ray_pyembree import RayMeshIntersector

MIN_MM = 0.8
MARGIN = 0.1 
PITCH = 0.05      # 體素大小 (mm)
SPACING = 0.06    # 撒點間距 (mm)


def face_thickness(mesh, idx):
    n = mesh.face_normals[idx]
    o = mesh.triangles_center[idx] - n * 1e-4
    loc, ri, _ = RayMeshIntersector(mesh).intersects_location(o, -n, multiple_hits=False)
    t = np.full(len(idx), np.inf)
    t[ri] = np.linalg.norm(loc - o[ri], axis=1)
    return t


def thin_faces(mesh, box):
    x0, x1, z0, z1 = box
    c = mesh.triangles_center
    idx = np.where((c[:, 0] > x0) & (c[:, 0] < x1) & (c[:, 2] > z0) & (c[:, 2] < z1))[0]
    return idx, face_thickness(mesh, idx)


def connected_to_seeds(mesh, fid, seeds, reach=2.0):
    """fid 裡面，跟 seeds 附近的面連在一起的那些面。"""
    fid = np.asarray(fid)
    adj = mesh.face_adjacency
    keep = np.isin(adj[:, 0], fid) & np.isin(adj[:, 1], fid)
    g = nx.Graph()
    g.add_nodes_from(fid.tolist())
    g.add_edges_from(adj[keep].tolist())
    c = mesh.triangles_center[fid]
    start = set()
    for s in seeds:
        d = np.linalg.norm(c - np.asarray(s), axis=1)
        if d.min() <= reach:
            start.add(int(fid[np.argmin(d)]))
    out = set()
    for comp in nx.connected_components(g):
        if comp & start:
            out |= comp
    return np.array(sorted(out), dtype=int)


def thicken(mesh, box, seeds, ignore_below=0.0, min_mm=MIN_MM):
    idx, t = thin_faces(mesh, box)
    cand = np.isfinite(t) & (t < 1.6) & (t >= ignore_below)
    tmap = dict(zip(idx[cand].tolist(), t[cand].tolist()))
    glasses = connected_to_seeds(mesh, idx[cand], seeds)
    gt = np.array([tmap[i] for i in glasses])
    sel = gt < min_mm
    fid, ft = glasses[sel], gt[sel]
    report = {"glasses_faces": int(len(glasses)), "thin_faces": int(sel.sum()), "thin_area_mm2": round(float(mesh.area_faces[fid].sum()), 1),
              "thinnest_before_mm": round(float(np.percentile(ft, 1)), 2) if len(ft) else None}
    if not len(fid):
        return mesh, report

    # 在薄的面上撒點，每點帶自己的半徑
    sub = mesh.submesh([fid], append=True)
    count = int(sub.area / SPACING ** 2) + 1
    pts, fi = trimesh.sample.sample_surface(sub, count)
    rad = (min_mm + MARGIN - ft[fi]) / 2
    pts = np.vstack([pts, sub.triangles_center])
    rad = np.concatenate([rad, (min_mm + MARGIN - ft) / 2])

    rmax = rad.max()
    lo = pts.min(0) - rmax - 2 * PITCH
    hi = pts.max(0) + rmax + 2 * PITCH
    shape = tuple(np.ceil((hi - lo) / PITCH).astype(int) + 1)
    field = np.full(shape, -1.0, dtype=np.float32)   # >0 = 在加厚層裡
    base = np.floor((pts - lo) / PITCH).astype(np.int32)
    k = int(np.ceil(rmax / PITCH)) + 1
    off = np.stack(np.meshgrid(*[np.arange(-k, k + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off, axis=1) <= k + 1]
    # 每個點在格子上蓋一個球：體素值 = 半徑 - 到點的距離
    for o in off:
        ijk = base + o
        d = np.linalg.norm(lo + ijk * PITCH - pts, axis=1)
        val = (rad - d).astype(np.float32)
        ok = val > -PITCH
        np.maximum.at(field, tuple(ijk[ok].T), val[ok])

    v, f, _, _ = marching_cubes(field, level=0.0, spacing=(PITCH,) * 3)
    shell = trimesh.Trimesh(v + lo, f, process=False)
    shell.fix_normals()
    report["added_shell_watertight"] = bool(shell.is_watertight)
    out = trimesh.boolean.union([mesh, shell], engine="manifold")
    out = max(out.split(only_watertight=False), key=lambda b: len(b.faces))

    report["watertight"] = bool(out.is_watertight)
    return out, report
