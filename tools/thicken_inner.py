"""
「內側整片拉出」加厚：像裙子這種一圈薄片，只把內側（朝身體那面）整片沿著各點的法線往內拉，
外側的面完全不動，從外面看形狀不變，薄片的下緣會一起變厚。

做法：
  1. 選出薄片的面（厚度 < thin_below，在 box 範圍內、離軸 max_r 以內）。
  2. 用射線找每個面對面的點，離中心軸比較近的那面就是內側。
  3. 內側每個三角形各做一個小柱體：底面是原本的三角形（往材料裡退 start），
     頂面是三個頂點各自沿「頂點法線」移動 dist；相鄰柱體共用邊，拉出來的面連續、光滑。
     dist 會被往內的剩餘空間限制（不撞到腿或身體）。
  4. 全部聯集。

用法（給 Python 呼叫，實例見檔尾 __main__）。
"""
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from thicken import face_thickness

MIN_MM = 0.85


def sheet_faces(m, zr, thin_below=0.7, axis=None, max_r=np.inf):
    c = m.triangles_center
    idx = np.where((c[:, 2] > zr[0]) & (c[:, 2] < zr[1]))[0]
    t = face_thickness(m, idx)
    ok = np.isfinite(t) & (t < thin_below)
    idx, t = idx[ok], t[ok]
    if axis is None:
        axis = c[idx][:, :2].mean(0)
    keep = np.linalg.norm(c[idx][:, :2] - axis, axis=1) < max_r
    return idx[keep], t[keep], np.asarray(axis, float)


def inner_side(m, faces, axis):
    """離中心軸比較近的那面 = 內側。"""
    c = m.triangles_center[faces]; n = m.face_normals[faces]
    loc, ri, _ = RayMeshIntersector(m).intersects_location(c - n * 1e-4, -n, multiple_hits=False)
    h = np.full((len(faces), 3), np.nan); h[ri] = loc
    rf = np.linalg.norm(c[:, :2] - axis, axis=1); rh = np.linalg.norm(h[:, :2] - axis, axis=1)
    return faces[rf < rh]


def vertex_normals(m, faces):
    vn = np.zeros((len(m.vertices), 3))
    for k in range(3):
        np.add.at(vn, m.faces[faces, k], m.face_normals[faces] * m.area_faces[faces, None])
    l = np.linalg.norm(vn, axis=1); l[l == 0] = 1
    return vn / l[:, None]


def free_space(m, pts, dirs):
    loc, ri, _ = RayMeshIntersector(m).intersects_location(pts + dirs * 1e-4, dirs, multiple_hits=False)
    f = np.full(len(pts), np.inf); f[ri] = np.linalg.norm(loc - pts[ri], axis=1)
    return f


def pull_inner(m, faces, dist, start=0.02, clearance=0.1):
    """faces（內側）沿頂點法線拉出 dist mm，回傳拉出的實體。"""
    vn = vertex_normals(m, faces)
    vids = np.unique(m.faces[faces])
    d = np.zeros(len(m.vertices))
    d[vids] = np.minimum(dist, free_space(m, m.vertices[vids], vn[vids]) - clearance).clip(0)
    F = m.faces[faces]
    lo = m.vertices[F] - vn[F] * start
    hi = m.vertices[F] + vn[F] * d[F][:, :, None]
    V = np.concatenate([lo, hi], axis=1).reshape(-1, 3)
    pf = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
    parts = [trimesh.Trimesh(V[i * 6:(i + 1) * 6], pf, process=False) for i in range(len(faces))]
    for q in parts:
        if q.volume < 0:
            q.invert()
    solid = trimesh.boolean.union(parts, engine="manifold")
    return solid, d[vids]


def save_clean(out, dst):
    """布林聯集後的極小碎邊在 STL（float32）裡會黏在一起變破洞：先用 manifold 合併，再不行才用 pymeshfix。"""
    import manifold3d, pymeshfix
    out = max(out.split(only_watertight=False), key=lambda b: len(b.faces))
    mf = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.asarray(out.vertices, np.float32),
                                             tri_verts=np.asarray(out.faces, np.uint32))).simplify(0.002).to_mesh()
    trimesh.Trimesh(mf.vert_properties[:, :3], mf.tri_verts).export(dst)
    m = trimesh.load(dst)
    if not m.is_watertight:
        v, f = pymeshfix.clean_from_arrays(np.ascontiguousarray(m.vertices, dtype=np.float64),
                                           np.ascontiguousarray(m.faces, dtype=np.int32))
        trimesh.Trimesh(v, f).export(dst)
        m = trimesh.load(dst)
    return m
