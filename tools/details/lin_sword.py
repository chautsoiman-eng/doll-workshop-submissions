"""
林羽倫：劍身前後兩面整片沿劍身的垂直方向往外拉，刀刃從 0.1～0.2mm 變成約 1mm，從正面看輪廓不變。
靠近護手（劍格）那 5mm 拉出量漸變到 0，接到護手的地方不會有段差。
做法是直接移動頂點（不用布林聯集），面數不變，不會產生碎面。
用法（在 repo 根目錄）:  python tools/details/lin_sword.py 0.5      # 每一面拉出 mm
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
from thicken_offset import crossing_edges
from thicken import face_thickness
D = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
src = 'submissions/LA1900936_林羽倫/LA1900936_林羽倫_v1_10cm_flat.stl'; dst = src[:-4] + '_thick.stl'
o = trimesh.load(src)
# 劍身座標：MU 中心、L 沿劍身（往護手）、W 寬度方向、N 劍面的法線（量自原始模型）
MU = np.array([-29.27, -15.66, 72.06])
c = o.triangles_center
s = np.where((c[:, 0] < -24) & (c[:, 2] > 57) & (c[:, 1] < -10))[0]
p = c[s] - c[s].mean(0); L, W, N = np.linalg.svd(p, full_matrices=False)[2]
if L[2] > 0: L = -L
MU = c[s].mean(0)
q = c - MU; a, w, n = q @ L, q @ W, q @ N
# 直接移動頂點：劍面這一側的頂點沿 +N 移 D、另一側沿 -N 移 D（整片平移，不用布林聯集，面數不變）。
# 刀刃上的三角形會被拉開，變成一條約 2D 寬的平整側面。
v = o.vertices - MU; va, vw, vn = v @ L, v @ W, v @ N
vnorm = o.vertex_normals @ N
sel = (va > -17.5) & (va < 18) & (np.abs(vw) < 3.8) & (np.abs(vn) < 0.95)
side = np.where(np.abs(vn) > 0.02, np.sign(vn), np.sign(vnorm))
ramp = np.clip((17 - va) / 5, 0, 1)                    # 護手上方 5mm 內漸變到 0（劍身中心往護手方向 17mm 之後完全不動）
V = o.vertices.copy()
V[sel] += (side[sel] * D * ramp[sel])[:, None] * N
m = trimesh.Trimesh(V, o.faces, process=False)
print(f'moved vertices {sel.sum()}')
m.export(dst); m = trimesh.load(dst)
for tag, mm in (('before', o), ('after', m)):
    cc = mm.triangles_center - MU; aa = cc @ L
    f = np.where((aa > -16) & (aa < 15) & (np.abs(cc @ W) < 3.7) & (np.abs(cc @ N) < 1.6))[0]
    t = face_thickness(mm, f)
    print(f'{tag}: blade thickness p1 {np.nanpercentile(t[np.isfinite(t)], 1):.2f}  area < 0.8mm {mm.area_faces[f][np.isfinite(t) & (t < 0.8)].sum():.1f} mm2  area < 1.0mm {mm.area_faces[f][np.isfinite(t) & (t < 1.0)].sum():.1f} mm2')
print('crossing edges near blade: before', crossing_edges(o, np.where(np.linalg.norm(o.triangles_center - MU, axis=1) < 25)[0]), 'after', crossing_edges(m, np.where(np.linalg.norm(m.triangles_center - MU, axis=1) < 25)[0]))
print('wt', m.is_watertight, 'bodies', len(m.split(only_watertight=False)), 'faces', len(o.faces), '->', len(m.faces))
