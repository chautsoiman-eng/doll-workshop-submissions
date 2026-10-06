"""
小細節加厚的指令工具（眼鏡、髮尾、手指、配件……），用「直接改面 + 平滑漸變」。
光固化建議最小壁厚 0.8mm（thicken.MIN_MM）。

每個模型的設定放在 tools/details_config.json：
    "模型名稱": {
        "file": "submissions/.../xxx_flat.stl",
        "parts": [
            {"name": "眼鏡",
             "box": [x0, x1, z0, z1],        # 這個部件大概所在的範圍（mm；正面看的左右 x、上下 z）
             "seeds": [[x, y, z], ...],      # 部件上的起點（從 inspect 的編號圖挑）
             "ignore_below": 0.3}            # 比這個還薄的面不處理（例如模型本身的眼皮薄膜）
        ]
    }

用法:
    # 0. 全身掃描：列出所有比 0.8mm 薄的區域，存一張熱圖，決定要處理哪些部件
    python tools/thicken_details.py scan 模型名稱 --png 熱圖.png
    python tools/thicken_details.py scan --file submissions/.../xxx.stl --png 熱圖.png
    # 1. 找起點：列出範圍內連在一起的薄面並編號，挑出屬於部件的編號，把 seed 填進設定檔
    python tools/thicken_details.py inspect 模型名稱 --box x0 x1 z0 z1 --png 編號圖.png
    # 2. 加厚：依序處理模型的所有部件，輸出 xxx_thick.stl，並印出前後厚度與檢查結果
    python tools/thicken_details.py run 模型名稱 [--png 預覽.png]
    python tools/thicken_details.py run --all
"""
import argparse
import json
import os
import warnings
import sys

import networkx as nx
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thicken import MIN_MM, connected_to_seeds, face_thickness, thin_faces  # noqa: E402
from thicken_offset import crossing_edges, offset_thicken  # noqa: E402

warnings.filterwarnings("ignore", message="Glyph")  # 圖上中文檔名缺字型，不影響結果
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "tools", "details_config.json")


def load_config():
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def out_path(path):
    return path[:-4] + "_thick.stl"


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


def render_heat(png, mesh, faces, t, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt = np.full(len(mesh.faces), np.inf)
    tt[faces] = t

    def colors(f, sh):
        v = tt[f]
        base = np.c_[sh * 0.85, sh * 0.85, sh * 0.85]
        hot = plt.cm.turbo_r(np.clip(v, 0, MIN_MM) / MIN_MM)[:, :3] * sh[:, None]
        return np.where((v < MIN_MM)[:, None], hot, base)

    fig, axs = plt.subplots(1, 4, figsize=(28, 10))
    allf = np.arange(len(mesh.faces))
    for ax, (name, proj, dep, sgn) in zip(axs, VIEWS):
        _draw(ax, mesh, allf, proj, dep, sgn, colors, f"{title} - {name}")
    fig.colorbar(plt.cm.ScalarMappable(cmap="turbo_r", norm=plt.Normalize(0, MIN_MM)), ax=axs, shrink=0.5,
                 label=f"thickness mm (only < {MIN_MM} colored)")
    plt.savefig(png, dpi=50)
    plt.close(fig)


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

def scan(path, png=None, top=20):
    mesh = trimesh.load(path)
    idx = np.arange(len(mesh.faces))
    t = face_thickness(mesh, idx)
    comps = thin_components(mesh, idx, t, MIN_MM, min_t=0.0, top=top)
    c = mesh.triangles_center
    print(f"== {os.path.basename(path)}  faces thinner than {MIN_MM}mm, grouped (largest first)")
    print(f"   total thin area {mesh.area_faces[np.isfinite(t) & (t < MIN_MM)].sum():.1f} mm2")
    for i, s in enumerate(comps):
        b = c[s]
        ts = t[s]
        print(f"  [{i:2d}] area {mesh.area_faces[s].sum():6.1f} mm2  thinnest {ts.min():.2f}mm  median {np.median(ts):.2f}mm  "
              f"box x {b[:, 0].min():.1f}..{b[:, 0].max():.1f}  z {b[:, 2].min():.1f}..{b[:, 2].max():.1f}  "
              f"(y {b[:, 1].min():.1f}..{b[:, 1].max():.1f})")
    if png:
        render_heat(png, mesh, idx, t, os.path.basename(path))
        print("  image:", png)


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
    ap.add_argument("cmd", choices=["scan", "inspect", "run"])
    ap.add_argument("name", nargs="?", help="details_config.json 裡的模型名稱")
    ap.add_argument("--file", help="直接指定 STL（不用設定檔）")
    ap.add_argument("--box", nargs=4, type=float, metavar=("X0", "X1", "Z0", "Z1"))
    ap.add_argument("--ignore-below", type=float, default=0.0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--png")
    a = ap.parse_args()
    conf = load_config() if os.path.exists(CONFIG) else {}
    path = a.file or (os.path.join(ROOT, conf[a.name]["file"]) if a.name else None)
    if a.cmd == "scan":
        scan(path, a.png)
    elif a.cmd == "inspect":
        inspect(path, a.box, a.ignore_below, a.png)
    else:
        for n in (list(conf) if a.all else [a.name]):
            run(n, conf[n], a.png if not a.all else None)


if __name__ == "__main__":
    main()
