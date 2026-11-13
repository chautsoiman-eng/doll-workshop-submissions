"""
把學員的 STL 等比縮放到 10cm 高，並把底部切平（方便黏底板）。

用法:
    python tools/scale_flatten.py submissions/XXX/XXX_v1.stl [...]

規則:
  1. 只做等比縮放（X/Y/Z 同一倍率），不會變形。Z 軸朝上。
  2. 底部平面檢查：在 10cm 尺度下，從最低點往上掃描水平截面，算面積 A(d) 和周長 P(d)。
     底部邊緣的平均斜率 = (dA/dd) / P（每往上 1mm，輪廓往外擴幾 mm）。
     找最小的切削深度 d，讓斜率 <= MAX_SLOPE（45°），也就是剛好把鞋底/底座的圓弧倒角切掉；
     再往上是鞋面、裙擺等造型本身，不切。用水平面切掉 d 並封口。
     若 Z=0 的真正水平面面積已經 >= 95% 的 A(d)，視為底部已經是平面，不切。
  3. 切完後再等比縮放一次，讓成品高度剛好 TARGET_H。
  4. 尺寸上限：寬(X) <= 90mm、深(Y) <= 50mm、高(Z) <= 120mm。
     10cm 高會超過上限的，改成等比縮小到剛好符合上限（高度會小於 10cm）。
輸出: 同資料夾 <原檔名>_10cm_flat.stl；縮小過的為 <原檔名>_<高度>mm_flat.stl
"""
import json
import sys

import numpy as np
import pymeshfix
import trimesh

TARGET_H = 100.0      # mm
MAX_SLOPE = 1.0       # tan(45°)：底部邊緣比 45° 陡就停止切
SCAN_MM = 4.0         # 往上掃描的範圍 (mm)
STEP_MM = 0.05
LIMITS = {"width_x": 90.0, "depth_y": 50.0, "height_z": 120.0}


def section_area_perimeter(mesh, z):
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return 0.0, 0.0
    polys = sec.to_2D()[0].polygons_full
    return float(sum(p.area for p in polys)), float(sum(p.length for p in polys))


def scale_to_height(mesh, h):
    mesh.apply_scale(h / mesh.extents[2])
    lo, hi = mesh.bounds
    # X/Y 置中，底部貼齊 Z=0
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def repair(mesh, report):
    if not mesh.is_watertight:
        # 丟掉孤立的雜點三角形；局部的非流形邊用 MeshFix 修；多個實體再聯集成一個
        bodies = [b for b in mesh.split(only_watertight=False) if len(b.faces) > 100]
        for i, b in enumerate(bodies):
            if not b.is_volume:
                v, f = pymeshfix.clean_from_arrays(
                    np.ascontiguousarray(b.vertices, dtype=np.float64),
                    np.ascontiguousarray(b.faces, dtype=np.int32))
                bodies[i] = trimesh.Trimesh(v, f)
        mesh = bodies[0] if len(bodies) == 1 else trimesh.boolean.union(bodies, engine="manifold")
        report["repaired_watertight"] = bool(mesh.is_watertight)
    return mesh


def process(path, target_h=TARGET_H, out=None, mesh=None):
    if mesh is None:
        mesh = trimesh.load(path, force="mesh")
    report = {"file": path, "orig_extents": mesh.extents.round(3).tolist(),
              "orig_watertight": bool(mesh.is_watertight)}
    mesh = repair(mesh, report)
    scale_to_height(mesh, target_h)

    depths = np.arange(STEP_MM, SCAN_MM + 1e-9, STEP_MM)
    areas, perims = np.array([section_area_perimeter(mesh, d) for d in depths]).T
    slope = np.gradient(areas, depths) / np.maximum(perims, 1e-9)
    i = int(np.argmax(slope <= MAX_SLOPE))
    cut = float(depths[i])
    report["cut_section_area_mm2"] = round(float(areas[i]), 1)

    down = (mesh.face_normals[:, 2] < -0.999) & (mesh.triangles_center[:, 2] < 1e-3)
    if mesh.area_faces[down].sum() >= 0.95 * areas[i]:
        report["already_flat"] = True
        report["cut_mm"] = 0.0
    else:
        report["already_flat"] = False
        report["cut_mm"] = round(cut, 2)
        mesh = mesh.slice_plane(plane_origin=[0, 0, cut], plane_normal=[0, 0, 1], cap=True)
        scale_to_height(mesh, target_h)

    w, d, h = mesh.extents
    fit = min(1.0, LIMITS["width_x"] / w, LIMITS["depth_y"] / d, LIMITS["height_z"] / h)
    report["fit_scale"] = round(fit, 4)
    if fit < 1.0:
        scale_to_height(mesh, h * fit)

    down = (mesh.face_normals[:, 2] < -0.999) & (mesh.triangles_center[:, 2] < 1e-3)
    report["base_area_mm2"] = round(float(mesh.area_faces[down].sum()), 1)
    report["watertight"] = bool(mesh.is_watertight)
    w, d, h = mesh.extents
    report["final_mm"] = {"width_x": round(w, 2), "depth_y": round(d, 2), "height_z": round(h, 2)}
    report["over_limit"] = [k for k, v in report["final_mm"].items() if v > LIMITS[k] + 1e-6]

    if out is None:
        h = mesh.extents[2]
        suffix = "_10cm_flat.stl" if abs(h - TARGET_H) < 1e-6 else f"_{h:.0f}mm_flat.stl"
        out = path[:-4] + suffix
    mesh.export(out)
    report["output"] = out
    return report


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(process(p), ensure_ascii=False, indent=2))
