"""
「填空心」加厚：像百褶裙這種，每一褶是空心的盒子（外牆、內牆各只有約 0.4mm，中間是空的），
把中間的空心一層一層填滿，外牆、內牆都不動，從外面看形狀不變，牆就變成一整塊實心。

做法：
  1. 每隔 dz（預設 0.05mm，跟光固化層高一樣）水平切一刀，找出切面上「封閉的小洞」
     （面積 < max_area、離中心軸 min_r 以上，這樣不會填到身體和裙子之間的大空間）。
  2. 小洞往外擴 grow（讓填充物稍微壓進牆裡，不會留縫），擠出成 dz 高的薄片。
  3. 全部薄片跟原模型聯集。空心在下襬打開的那一段（切面不是封閉的）不會填。

用法（給 Python 呼叫，實例見 tools/details/xie_skirt.py）。
"""
import numpy as np
import trimesh
import manifold3d
from shapely.geometry import Polygon


def cavity_slabs(m, z0, z1, axis, max_area=12.0, min_r=6.0, dz=0.05, grow=0.03):
    zs = np.arange(z0 + dz / 2, z1, dz)
    secs = m.section_multiplane([0, 0, 0], [0, 0, 1], zs)
    slabs, stats = [], []
    for z, s in zip(zs, secs):
        if s is None:
            continue
        T = np.eye(3)
        if hasattr(s, "metadata") and "to_3D" in s.metadata:
            T = s.metadata["to_3D"]
        holes = []
        for p in s.polygons_full:
            for h in p.interiors:
                hp = Polygon(h)
                if not hp.is_valid or hp.area < 1e-4 or hp.area > max_area:
                    continue
                c3 = (T @ np.r_[hp.centroid.x, hp.centroid.y, 0, 1])[:3]
                if np.linalg.norm(c3[:2] - axis) < min_r:
                    continue
                g = hp.buffer(grow, join_style=2)
                pts = np.asarray(g.exterior.coords)[:-1]
                # 切面座標轉回世界座標（section_multiplane 的平面是水平的，只有平移）
                w = (T @ np.c_[pts, np.zeros(len(pts)), np.ones(len(pts))].T).T[:, :2]
                if Polygon(w).exterior.is_ccw is False:
                    w = w[::-1]
                holes.append(w)
        if not holes:
            continue
        cs = manifold3d.CrossSection([h.astype(np.float64) for h in holes], manifold3d.FillRule.NonZero)
        slabs.append(manifold3d.Manifold.extrude(cs, dz + 0.01).translate([0, 0, z - dz / 2 - 0.005]))
        stats.append((z, len(holes)))
    return slabs, stats


def fill(m, z0, z1, axis, **kw):
    slabs, stats = cavity_slabs(m, z0, z1, np.asarray(axis, float), **kw)
    base = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                                               tri_verts=np.asarray(m.faces, np.uint32)))
    out = manifold3d.Manifold.batch_boolean([base] + slabs, manifold3d.OpType.Add).to_mesh()
    return trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts), stats
