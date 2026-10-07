"""
林羽倫：劍身只放大「厚度方向」的比例（長度、寬度不變），刀刃一樣是尖的，形狀只是等比例變厚。
整支劍身（含靠近護手、有花紋的那段）都放大，進到護手的 1.5mm 內倍率漸變回 1，不會有段差。
做法是直接移動頂點，面數不變。
用法（在 repo 根目錄）:  python tools/details/lin_sword.py 1.5      # 厚度倍率
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
from thicken_offset import crossing_edges
K = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
src = 'submissions/LA1900936_林羽倫/LA1900936_林羽倫_v1_10cm_flat.stl'; dst = src[:-4] + '_thick.stl'
o = trimesh.load(src)
# 劍身座標：MU 中心、L 沿劍身（往護手）、W 寬度方向、N 劍面的法線
c = o.triangles_center
s = np.where((c[:, 0] < -24) & (c[:, 2] > 57) & (c[:, 1] < -10))[0]
MU = c[s].mean(0); L, W, N = np.linalg.svd(c[s] - MU, full_matrices=False)[2]
if L[2] > 0: L = -L
v = o.vertices - MU; va, vw, vn = v @ L, v @ W, v @ N
# 劍身（含護手上方有花紋的那段，到 23mm）＋護手開頭 1.5mm 的漸變區
sel = (va > -17.5) & (((va < 23) & (np.abs(vw) < 5) & (np.abs(vn) < 1.2)) | ((va >= 23) & (va < 24.5) & (np.abs(vw) < 10.5) & (np.abs(vn) < 2.5)))
# 每一段劍身的中心面（厚度方向的中點），沿長度平滑
bins = np.arange(-17.5, 24.5, 1.0); mid = []
for b0 in bins:
    k = sel & (va >= b0) & (va < b0 + 1)
    mid.append((vn[k].max() + vn[k].min()) / 2 if k.any() else 0)
center = np.interp(va, bins + 0.5, mid)
ramp = np.clip((24.5 - va) / 1.5, 0, 1)                # 整支劍身都放大，進到護手的 1.5mm 內倍率漸變回 1
k = 1 + (K - 1) * ramp
V = o.vertices.copy()
V[sel] += (((vn - center) * (k - 1))[sel])[:, None] * N
m = trimesh.Trimesh(V, o.faces, process=False); m.export(dst); m = trimesh.load(dst)
print(f'scale thickness x{K}: moved vertices {sel.sum()}')
for t in (-14, -12, -8, -4, 0, 4, 8, 12, 15, 17, 19.5, 22, 24):
    row = []
    for mm in (o, m):
        P = np.vstack(mm.section(L, MU + L * t).discrete) - MU; pw, pn = P @ W, P @ N
        b = (np.abs(pw) < 5) & (np.abs(pn) < 2); pw, pn = pw[b], pn[b]
        if mm is o: wid = pw.max() - pw.min(); cen = (pw.max() + pw.min()) / 2
        q = np.abs(pw - cen) < 0.3; row.append(pn[q].max() - pn[q].min())
        wl = pw.max() - pw.min()
    print(f'{t:+5.1f}mm: width {wid:.2f} -> {wl:.2f}  center thickness {row[0]:.2f} -> {row[1]:.2f}')
near = lambda mm: np.where(np.linalg.norm(mm.triangles_center - MU, axis=1) < 25)[0]
print('crossing edges near blade: before', crossing_edges(o, near(o)), 'after', crossing_edges(m, near(m)))
print('wt', m.is_watertight, 'bodies', len(m.split(only_watertight=False)), 'faces', len(o.faces), '->', len(m.faces))
