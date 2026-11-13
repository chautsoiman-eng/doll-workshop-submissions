"""
Phoebe_Wang：荷葉裙擺是單層薄片（約 0.65mm），只把內側（朝腿那面）的頂點沿法線往內推，
推到總厚度約 TARGET，外側不動，從外面看形狀不變。面數不變，不用布林聯集。
推動量依各點原本的厚度決定（越薄推越多），再沿網格平滑，不會有段差。
用法（在 repo 根目錄）:  python tools/details/phoebe_skirt.py 0.85
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
from thicken import face_thickness
from thicken_inner import inner_side
from thicken_offset import crossing_edges
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
src = 'submissions/LA2000262_Phoebe_Wang/LA2000262_Phoebe_Wang_v1_90mm_flat.stl'; dst = src[:-4] + '_thick.stl'
o = trimesh.load(src)
c = o.triangles_center
Z0, Z1 = 13.0, 24.5
cand = np.where((c[:, 2] > Z0) & (c[:, 2] < Z1))[0]
t = face_thickness(o, cand)
ok = np.isfinite(t) & (t < 0.8)
axis = np.array([-1.3, -5.0])
r = np.linalg.norm(c[cand][:, :2] - axis, axis=1)
ok &= r > 7.5                                           # 排除腿
sheet, ts = cand[ok], t[ok]
inner = inner_side(o, sheet, axis)
# 波浪的側面容易判斷錯（內外側交錯），用相鄰面多數決把標記弄平順，再排除朝下的裙擺邊緣
lab = np.zeros(len(o.faces), np.int8); lab[sheet] = -1; lab[inner] = 1
adj = o.face_adjacency; adj = adj[(lab[adj[:, 0]] != 0) & (lab[adj[:, 1]] != 0)]
for _ in range(8):
    acc = np.zeros(len(o.faces))
    np.add.at(acc, adj[:, 0], lab[adj[:, 1]]); np.add.at(acc, adj[:, 1], lab[adj[:, 0]])
    flip = (lab != 0) & (np.sign(acc) == -lab) & (np.abs(acc) >= 2)
    lab[flip] = -lab[flip]
inner = np.where((lab == 1) & (o.face_normals[:, 2] > -0.7))[0]
outer_v = np.zeros(len(o.vertices), bool); outer_v[np.unique(o.faces[lab == -1])] = True
tmap = dict(zip(sheet.tolist(), ts.tolist()))
print(f'skirt thin faces {len(sheet)}  inner {len(inner)}  thickness median {np.median(ts):.2f}')
# 每個內側頂點的推動量 = TARGET - 相鄰內側面的最小厚度
need = np.zeros(len(o.vertices))
for k in range(3):
    np.maximum.at(need, o.faces[inner, k], np.clip(TARGET - np.array([tmap[f] for f in inner]), 0, 0.4))
# 沿網格平滑推動量（只在內側頂點之間），避免段差
vid = np.unique(o.faces[inner]); mask = np.zeros(len(o.vertices), bool); mask[vid] = True
mask &= ~outer_v                                         # 同時屬於外側面的點（邊緣）不動，外側形狀不變
vid = np.where(mask)[0]
e = o.edges_unique; e = e[mask[e[:, 0]] & mask[e[:, 1]]]
for _ in range(6):
    acc = np.zeros(len(o.vertices)); cnt = np.zeros(len(o.vertices))
    np.add.at(acc, e[:, 0], need[e[:, 1]]); np.add.at(acc, e[:, 1], need[e[:, 0]])
    np.add.at(cnt, e[:, 0], 1); np.add.at(cnt, e[:, 1], 1)
    sm = np.where(cnt > 0, acc / np.maximum(cnt, 1), need)
    need[mask] = np.maximum(need[mask] * 0.5 + sm[mask] * 0.5, 0)
vn = np.zeros((len(o.vertices), 3))
for k in range(3):
    np.add.at(vn, o.faces[inner, k], o.face_normals[inner] * o.area_faces[inner, None])
vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-12)[:, None]
# 前方空間不夠（荷葉邊的波浪摺得很近）的地方少推一點，留 0.1mm，不撞到隔壁那一層
from thicken_inner import free_space
gap = np.full(len(o.vertices), np.inf); gap[vid] = free_space(o, o.vertices[vid], vn[vid])
need[mask] = np.minimum(need[mask], np.clip(gap[mask] / 2 - 0.05, 0, None))
for _ in range(3):                                      # 限制後再平滑一次（只會變小，不會撞）
    acc = np.zeros(len(o.vertices)); cnt = np.zeros(len(o.vertices))
    np.add.at(acc, e[:, 0], need[e[:, 1]]); np.add.at(acc, e[:, 1], need[e[:, 0]])
    np.add.at(cnt, e[:, 0], 1); np.add.at(cnt, e[:, 1], 1)
    sm = np.where(cnt > 0, acc / np.maximum(cnt, 1), need)
    need[mask] = np.minimum(need[mask], sm[mask])
# 窄凹槽：兩側往內推會在中間撞在一起。推完後量新位置離原本表面的距離，太近（< 推動量的 80%）就縮小，重複幾次
pq = trimesh.proximity.ProximityQuery(o)
for it in range(12):
    P = o.vertices[vid] + vn[vid] * need[vid, None]
    _, dd, _ = pq.on_surface(P)
    bad = dd < need[vid] * 0.8 - 1e-4
    if not bad.any(): break
    need[vid[bad]] = np.clip(dd[bad] * 0.8, 0, None)
    print(f'  pass {it}: reduced {bad.sum()} vertices')
    for _ in range(2):                                  # 縮小的地方跟鄰居平滑（只往小的方向），不留凹點
        acc = np.zeros(len(o.vertices)); cnt = np.zeros(len(o.vertices))
        np.add.at(acc, e[:, 0], need[e[:, 1]]); np.add.at(acc, e[:, 1], need[e[:, 0]])
        np.add.at(cnt, e[:, 0], 1); np.add.at(cnt, e[:, 1], 1)
        sm = np.where(cnt > 0, acc / np.maximum(cnt, 1), need)
        need[mask] = np.minimum(need[mask], sm[mask] * 0.5 + need[mask] * 0.5)
V = o.vertices.copy(); V[mask] += vn[mask] * need[mask, None]     # 內側面的法線朝內（朝腿），往那邊推
m = trimesh.Trimesh(V, o.faces, process=False); m.export(dst); m = trimesh.load(dst)
print(f'moved vertices {mask.sum()}  push median {np.median(need[mask]):.2f} max {need[mask].max():.2f}')
for tag, mm in (('before', o), ('after', m)):
    cc = mm.triangles_center; f = np.where((cc[:, 2] > Z0) & (cc[:, 2] < Z1) & (np.linalg.norm(cc[:, :2] - axis, axis=1) > 7.5))[0]
    tt = face_thickness(mm, f)
    print(f'{tag}: skirt area < 0.8mm {mm.area_faces[f][np.isfinite(tt) & (tt < 0.8)].sum():.0f} mm2')
box = np.where((c[:, 2] > Z0) & (c[:, 2] < Z1))[0]
print('crossing edges: before', crossing_edges(o, box), 'after', crossing_edges(m, box))
print('wt', m.is_watertight, 'bodies', len(m.split(only_watertight=False)), 'faces', len(o.faces), '->', len(m.faces))
