"""
眼鏡加厚（獨立版，凍結在 2026-10 確認過效果的版本）。

這個檔案自帶全部演算法，不 import tools/ 底下的其他加厚程式，
所以之後改通用版（tools/thicken_details.py）不會影響眼鏡的結果。

做法：「直接改面 + 平滑漸變」
  1. 量厚度：每個三角面沿表面往內打射線，量到另一側的距離。
  2. 找眼鏡：從 seeds（鏡框、鏡腳上的點）出發，找連在一起、比 THIN_MM 薄的面；
     比較厚但離薄眼鏡 NEAR_MM 以內的面（鏡腳接合方塊、鼻樑架根部）也算進來。
  3. 每個頂點需要往外推 (MIN_MM + MARGIN - 厚度) / 2；
     推動量沿表面距離在 FALLOFF_MM 內平滑降到 0，靠近臉/頭髮交界 BORDER_FALLOFF_MM 內降到 0。
  4. 推之前檢查會不會撞到其他表面（只推一半距離），推動量再整體平滑一次。
  5. 推完檢查面有沒有互相穿插（用 STL 的 float32 精度），有就把那附近的推動量減半重來。
  頂點數、面數不變，表面維持原本的平滑度。

設定檔 tools/glasses/glasses_config.json：
    "模型名稱": {"file": "submissions/.../xxx_flat.stl",
                 "parts": [{"name": "眼鏡", "box": [x0, x1, z0, z1],
                            "seeds": [[x, y, z], ...], "ignore_below": 0.3}]}

用法:
    python tools/glasses/thicken_glasses.py inspect 模型名稱 --box x0 x1 z0 z1 --png 編號圖.png
    python tools/glasses/thicken_glasses.py run 模型名稱 [--png 預覽.png]
    python tools/glasses/thicken_glasses.py run --all
輸出: 原檔名後加 _glasses.stl
"""
import argparse
import json
import os
import warnings

import networkx as nx
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector

warnings.filterwarnings("ignore", message="Glyph")  # 圖上中文檔名缺字型，不影響結果
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glasses_config.json")

MIN_MM = 0.8
MARGIN = 0.1 
THIN_MM = 1.6            # 比這個薄的才算「薄眼鏡」
NEAR_MM = 0.6            # 比較厚的面離薄眼鏡多近才算進眼鏡物件
STEP_SMOOTH_ITERS = 20
OBJECT_MAX_MM = 3.0      # 這個厚度以下、跟眼鏡連在一起的都算眼鏡物件
FALLOFF_MM = 1.0         # 推動量往外漸變到 0 的距離
BORDER_FALLOFF_MM = 0.5  # 靠近臉/頭髮交界時漸變到 0 的距離
MAX_FIX_ITERS = 8


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


# ---------- 指令 ----------



def load_config():
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def out_path(path):
    return path[:-4] + "_glasses.stl"


def wpct(t, a, q):
    o = np.argsort(t)
    cum = np.cumsum(a[o]) / a.sum()
    return float(t[o][min(np.searchsorted(cum, q), len(o) - 1)])


def in_box(mesh, box, pad=0.0):
    x0, x1, z0, z1 = box
    c = mesh.triangles_center
    return np.where((c[:, 0] > x0 - pad) & (c[:, 0] < x1 + pad) & (c[:, 2] > z0 - pad) & (c[:, 2] < z1 + pad))[0]


def thin_components(mesh, idx, t, max_t, min_t=0.0, top=10):
    """idx/t 裡比 max_t 薄的面，依連通分成幾塊，回傳（由大到小）面的陣列。"""
    cand = np.isfinite(t) & (t < max_t) & (t >= min_t)
    fid = idx[cand]
    adj = mesh.face_adjacency
    keep = np.isin(adj[:, 0], fid) & np.isin(adj[:, 1], fid)
    g = nx.Graph()
    g.add_nodes_from(fid.tolist())
    g.add_edges_from(adj[keep].tolist())
    comps = sorted(nx.connected_components(g), key=lambda s: -mesh.area_faces[list(s)].sum())
    return [np.array(sorted(s)) for s in comps[:top]]


def part_stats(mesh, part):
    """用同一組 seeds 選出部件，回傳面積加權的厚度統計。"""
    idx = in_box(mesh, part["box"])
    t = face_thickness(mesh, idx)
    ok = np.isfinite(t) & (t < 2.5) & (t >= part.get("ignore_below", 0.0))
    g = connected_to_seeds(mesh, idx[ok], part["seeds"], reach=0.5)
    if not len(g):
        return {"error": "no faces found near seeds"}
    tm = dict(zip(idx[ok].tolist(), t[ok].tolist()))
    gt = np.array([tm[i] for i in g])
    a = mesh.area_faces[g]
    return {
        "p1_mm": round(wpct(gt, a, .01), 2),
        "p5_mm": round(wpct(gt, a, .05), 2),
        "median_mm": round(wpct(gt, a, .5), 2),
        f"under_{MIN_MM}_mm2": round(float(a[gt < MIN_MM].sum()), 1),
        "area_mm2": round(float(a.sum()), 1),
    }


# ---------- 繪圖 ----------


def _draw(ax, mesh, faces, proj, dep, sgn, colors_fn, title):
    from matplotlib.collections import PolyCollection
    c = mesh.triangles_center
    n = mesh.face_normals[faces]
    f = faces[n[:, dep] * sgn > 0]
    f = f[np.argsort(c[f][:, dep] * sgn)]
    sh = 0.25 + 0.75 * np.abs(mesh.face_normals[f][:, dep])
    col = colors_fn(f, sh)
    ax.add_collection(PolyCollection(mesh.triangles[f][:, :, proj], facecolors=col, edgecolors=col,
                                     linewidths=0.15, antialiased=False))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.grid(alpha=.3)
    ax.set_title(title)


VIEWS = [("front", [0, 2], 1, -1), ("back", [0, 2], 1, 1), ("right (+X)", [1, 2], 0, 1), ("left (-X)", [1, 2], 0, -1)]


def render_labels(png, mesh, faces, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = plt.cm.tab10(np.arange(10))[:, :3]

    def colors(f, sh):
        lab = labels[f]
        return np.where((lab >= 0)[:, None], pal[np.maximum(lab, 0) % 10] * sh[:, None],
                        np.c_[sh * 0.9, sh * 0.9, sh * 0.9])

    fig, axs = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    c = mesh.triangles_center
    for ax, (name, proj, dep, sgn) in zip(axs, [VIEWS[0], VIEWS[2], VIEWS[3]]):
        _draw(ax, mesh, faces, proj, dep, sgn, colors, name)
        for i in range(labels.max() + 1):
            q = c[labels == i].mean(0)
            ax.text(q[proj[0]], q[proj[1]], str(i), fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(png, dpi=55)
    plt.close(fig)


def render_pushed(png, mesh, ref, faces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dv = np.linalg.norm(mesh.vertices - ref.vertices, axis=1)

    def colors(f, sh):
        w = np.clip(dv[mesh.faces[f]].max(1) / 0.2, 0, 1)[:, None]
        return np.c_[sh * 0.9, sh * 0.9, sh * 0.9] * (1 - w) + np.c_[sh, sh * 0.2, sh * 0.15] * w

    fig, axs = plt.subplots(1, 3, figsize=(24, 8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    for ax, (name, proj, dep, sgn) in zip(axs, [VIEWS[0], VIEWS[2], VIEWS[3]]):
        _draw(ax, mesh, faces, proj, dep, sgn, colors, f"{name}  (red = pushed out, full red = 0.2mm)")
    plt.tight_layout()
    plt.savefig(png, dpi=55)
    plt.close(fig)


# ---------- 指令 ----------


def inspect(path, box, ignore_below, png=None):
    mesh = trimesh.load(path)
    idx, t = thin_faces(mesh, box)
    comps = thin_components(mesh, idx, t, 1.6, min_t=ignore_below)
    labels = np.full(len(mesh.faces), -1)
    c = mesh.triangles_center
    print(f"== {os.path.basename(path)}  thin faces (< 1.6mm) inside box {box}, largest first")
    for i, s in enumerate(comps):
        labels[s] = i
        b = c[s]
        seed = c[s[np.argmin(np.linalg.norm(b - b.mean(0), axis=1))]]
        print(f"  [{i}] area {mesh.area_faces[s].sum():6.1f} mm2  seed {np.round(seed, 2).tolist()}  "
              f"x {b[:, 0].min():.1f}..{b[:, 0].max():.1f}  y {b[:, 1].min():.1f}..{b[:, 1].max():.1f}  "
              f"z {b[:, 2].min():.1f}..{b[:, 2].max():.1f}")
    if png:
        render_labels(png, mesh, in_box(mesh, box, pad=1.5), labels)
        print("  image:", png)


def run(name, cfg, png=None):
    src = os.path.join(ROOT, cfg["file"])
    orig = trimesh.load(src)
    mesh = orig
    parts = []
    for part in cfg["parts"]:
        before = part_stats(mesh, part)
        mesh, rep = offset_thicken(mesh, part["box"], part["seeds"], ignore_below=part.get("ignore_below", 0.0))
        parts.append({"part": part["name"], "before": before, "max_push_mm": rep["max_push_mm"]})
    dst = out_path(src)
    mesh.export(dst)
    saved = trimesh.load(dst)
    region = np.unique(np.concatenate([in_box(saved, p["box"], pad=1.0) for p in cfg["parts"]]))
    for p, part in zip(parts, cfg["parts"]):
        p["after"] = part_stats(saved, part)
    result = {
        "model": name,
        "output": os.path.relpath(dst, ROOT),
        "parts": parts,
        "crossing_edges": crossing_edges(saved, region),
        "watertight": bool(saved.is_watertight),
        "bodies": len(saved.split(only_watertight=False)),
        "faces_unchanged": len(saved.faces) == len(orig.faces),
        "extents_mm": np.round(saved.extents, 2).tolist(),
    }
    print(json.dumps(result, ensure_ascii=False))
    if png:
        render_pushed(png, saved, orig, region)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "run"])
    ap.add_argument("name", nargs="?", help="glasses_config.json 裡的模型名稱")
    ap.add_argument("--file", help="直接指定 STL（inspect 用）")
    ap.add_argument("--box", nargs=4, type=float, metavar=("X0", "X1", "Z0", "Z1"))
    ap.add_argument("--ignore-below", type=float, default=0.3)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--png")
    a = ap.parse_args()
    conf = load_config()
    if a.cmd == "inspect":
        path = a.file or os.path.join(ROOT, conf[a.name]["file"])
        inspect(path, a.box, a.ignore_below, a.png)
    else:
        for n in (list(conf) if a.all else [a.name]):
            run(n, conf[n], a.png if not a.all else None)


if __name__ == "__main__":
    main()
