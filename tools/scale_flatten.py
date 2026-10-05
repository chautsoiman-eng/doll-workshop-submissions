"""
把學員的 STL 等比縮放到 10cm 高，並把底部切平（方便黏底板）。

用法:
    python tools/scale_flatten.py submissions/XXX/XXX_v1.stl [...]

規則:
  1. 只做等比縮放（X/Y/Z 同一倍率），不會變形。Z 軸朝上。
  2. 底部平面檢查：在 10cm 尺度下，從最低點往上掃描水平截面面積 A(d)。
     A_max = 底部 SCAN_MM 範圍內最大的截面面積（約等於整個鞋底/底座的面積）。
     若 A(FLAT_TOL_MM) 已經 >= COVERAGE * A_max，視為底部已經是平面，不切。
     否則找最小的切削深度 d，使 A(d) >= COVERAGE * A_max，用水平面切掉 d 並封口。
  3. 切完後再等比縮放一次，讓成品高度剛好 TARGET_H。
  4. 檢查尺寸上限：寬(X) <= 90mm、深(Y) <= 50mm、高(Z) <= 120mm。
輸出: 同資料夾 <原檔名>_10cm_flat.stl
"""
import json
import os
import sys

import numpy as np
import trimesh

TARGET_H = 100.0      # mm
COVERAGE = 0.95       # 切面面積要達到鞋底最大截面的比例
SCAN_MM = 4.0         # 往上掃描的範圍 (mm)
STEP_MM = 0.05
FLAT_TOL_MM = 0.05    # 這個深度內已達標 = 底部本來就是平的
LIMITS = {"width_x": 90.0, "depth_y": 50.0, "height_z": 120.0}


def section_area(mesh, z):
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return 0.0
    planar, _ = sec.to_2D()
    return float(sum(p.area for p in planar.polygons_full))


def scale_to_height(mesh, h):
    mesh.apply_scale(h / mesh.extents[2])
    lo, hi = mesh.bounds
    # X/Y 置中，底部貼齊 Z=0
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def process(path):
    mesh = trimesh.load(path, force="mesh")
    report = {"file": path, "orig_extents": mesh.extents.round(3).tolist(),
              "orig_watertight": bool(mesh.is_watertight)}
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    scale_to_height(mesh, TARGET_H)

    depths = np.arange(STEP_MM, SCAN_MM + 1e-9, STEP_MM)
    areas = np.array([section_area(mesh, d) for d in depths])
    a_max = float(areas.max())
    need = COVERAGE * a_max
    cut = float(depths[np.argmax(areas >= need)])
    report["bottom_max_area_mm2"] = round(a_max, 1)
    report["area_at_0.05mm_mm2"] = round(float(areas[0]), 1)

    if cut <= FLAT_TOL_MM:
        report["already_flat"] = True
        report["cut_mm"] = 0.0
    else:
        report["already_flat"] = False
        report["cut_mm"] = round(cut, 2)
        mesh = mesh.slice_plane(plane_origin=[0, 0, cut], plane_normal=[0, 0, 1], cap=True)
        scale_to_height(mesh, TARGET_H)

    down = (mesh.face_normals[:, 2] < -0.999) & (mesh.triangles_center[:, 2] < 1e-3)
    report["base_area_mm2"] = round(float(mesh.area_faces[down].sum()), 1)
    report["watertight"] = bool(mesh.is_watertight)
    w, d, h = mesh.extents
    report["final_mm"] = {"width_x": round(w, 2), "depth_y": round(d, 2), "height_z": round(h, 2)}
    report["over_limit"] = [k for k, v in report["final_mm"].items() if v > LIMITS[k] + 1e-6]

    out = path[:-4] + "_10cm_flat.stl"
    mesh.export(out)
    report["output"] = out
    return report


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(process(p), ensure_ascii=False, indent=2))
