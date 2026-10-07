"""
謝雅慧：百褶裙的每一褶是空心的（外牆、內牆各約 0.38mm），把空心一層一層填滿，外面形狀不變。
用法（在 repo 根目錄）:  python tools/details/xie_skirt.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, trimesh
from fill_cavities import fill
from thicken_inner import save_clean
from thicken import face_thickness
src = 'submissions/LA1301182_謝雅慧/LA1301182_謝雅慧_v1_10cm_flat.stl'; dst = src[:-4] + '_thick.stl'
AXIS = (3.47, -8.89)          # 裙子中心軸（x, y）
o = trimesh.load(src)
out, stats = fill(o, 25.5, 37.5, AXIS)
print(f'slices filled {len(stats)}  z {stats[0][0]:.2f}..{stats[-1][0]:.2f}  holes/slice max {max(s[1] for s in stats)}')
m = save_clean(out, dst)

def skirt(mm):
    c = mm.triangles_center
    return np.where((c[:, 2] > 25.5) & (c[:, 2] < 37.5) & (np.linalg.norm(c[:, :2] - AXIS, axis=1) < 14))[0]
for tag, mm in (('before', o), ('after', m)):
    f = skirt(mm); t = face_thickness(mm, f)
    print(f'{tag}: skirt area < 0.8mm {mm.area_faces[f][np.isfinite(t) & (t < 0.8)].sum():.0f} mm2')
_, d, _ = trimesh.proximity.closest_point(o, m.vertices[np.unique(m.faces[skirt(m)])])
print(f'new surface farther than 0.01mm from the original: {np.mean(d > 0.01) * 100:.1f}% of skirt vertices')
print('wt', m.is_watertight, 'bodies', len(m.split(only_watertight=False)), 'faces', len(o.faces), '->', len(m.faces))
