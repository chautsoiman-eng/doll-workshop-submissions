"""
把幾個公仔由下往上疊起來（上面的公仔踩在下面公仔的頭上），合成一個實體，
最底部切平，再等比縮放到不超過 90x50x120mm 的最大尺寸。

用法:
    python tools/stack_dolls.py 輸出.stl 最底.stl 中間.stl 最上.stl

規則:
  1. 所有公仔用同一個比例（原始檔都是同一套工具產生，尺度一致），不變形。
  2. 上面公仔「腳底」的中心對準下面公仔「頭頂」的中心（XY），
     再往下放到剛好碰到頭頂，然後多壓 SINK_MM 進去，讓接觸面夠大、能黏成一體。
  3. 用 manifold 布林聯集成一個封閉實體。
  4. 交給 scale_flatten.process：最底部用 45° 規則切平，縮放到 120mm 高，
     超過 90x50 的話再等比縮小。
"""
import json
import sys

import numpy as np
import trimesh

import scale_flatten as sf

TARGET_H = 120.0   # 疊起來的總高度上限
SINK_MM = 1.5      # 上面公仔的腳壓進下面公仔頭頂的深度（以成品尺寸計）
FEET_BAND = 0.03   # 最低的這段高度（原始單位）當作腳底
HEAD_BAND = 0.05   # 最高的這段高度（原始單位）當作頭頂
GRID = 0.004       # 高度圖格子大小（原始單位）


def height_map(v, origin, shape, top):
    ij = np.floor((v[:, :2] - origin) / GRID).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    hm = np.full(shape, -np.inf if top else np.inf)
    (np.maximum if top else np.minimum).at(hm, (ij[ok, 0], ij[ok, 1]), v[ok, 2])
    return hm


def place_on(lower, upper, sink):
    """移動 upper，讓它的腳踩在 lower 的頭頂上，回傳實際接觸格數。"""
    lv, uv = lower.vertices, upper.vertices
    head = lv[lv[:, 2] > lv[:, 2].max() - HEAD_BAND][:, :2].mean(0)
    feet = uv[uv[:, 2] < uv[:, 2].min() + FEET_BAND][:, :2].mean(0)
    upper.apply_translation([*(head - feet), 0])
    uv = upper.vertices

    lo = np.minimum(lv[:, :2].min(0), uv[:, :2].min(0))
    hi = np.maximum(lv[:, :2].max(0), uv[:, :2].max(0))
    shape = tuple(np.ceil((hi - lo) / GRID).astype(int) + 1)
    top = height_map(lv, lo, shape, top=True)
    bot = height_map(uv, lo, shape, top=False)
    both = np.isfinite(top) & np.isfinite(bot)
    if not both.any():
        raise ValueError("上面公仔的腳和下面公仔的頭沒有重疊")
    dz = float((top - bot)[both].max())   # 剛好碰到
    upper.apply_translation([0, 0, dz - sink])
    return int(both.sum())


def stack(paths, out):
    report = {"order_bottom_to_top": paths, "joints": []}
    meshes = [sf.repair(trimesh.load(p, force="mesh"), {}) for p in paths]
    # 先粗估成品比例，把 SINK_MM 換算成原始單位
    unit_per_mm = sum(m.extents[2] for m in meshes) / TARGET_H
    sink = SINK_MM * unit_per_mm
    for m in meshes:
        m.apply_translation(-m.bounds[0] * [0, 0, 1])
    for lower, upper in zip(meshes, meshes[1:]):
        place_on(lower, upper, sink)
        joint = trimesh.boolean.intersection([lower, upper], engine="manifold")
        report["joints"].append({
            "overlap_xy_mm": (joint.extents[:2] / unit_per_mm).round(1).tolist()
            if len(joint.faces) else [0, 0],
            "overlap_volume_mm3": round(float(joint.volume) / unit_per_mm ** 3, 1)
            if len(joint.faces) else 0.0,
        })
    combined = trimesh.boolean.union(meshes, engine="manifold")
    report.update(sf.process(out, target_h=TARGET_H, out=out, mesh=combined))
    return report


if __name__ == "__main__":
    print(json.dumps(stack(sys.argv[2:], sys.argv[1]), ensure_ascii=False, indent=2))
