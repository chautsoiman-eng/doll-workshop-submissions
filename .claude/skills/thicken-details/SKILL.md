---
name: thicken-details
description: 掃描公仔 STL 上所有比 0.8mm 薄的小細節（髮尾、手指、配件、飾品、眼鏡……），挑出要處理的部件，用「直接改面 + 平滑漸變」加厚到光固化可列印的厚度。眼鏡請優先用 thicken-glasses。
---

# 小細節加厚（通用版）

程式：`tools/thicken_details.py`（演算法在 `tools/thicken_offset.py`、`tools/thicken.py`）
設定：`tools/details_config.json`
輸出：原檔名加 `_thick.stl`

> 眼鏡已經有凍結、確認過效果的獨立版 `thicken-glasses`（`tools/glasses/`）。
> 這個通用版會繼續改進，改動不會影響眼鏡版。

## 準備

```bash
pip install trimesh numpy scipy networkx embreex shapely matplotlib manifold3d pymeshfix
```

模型要先經過 `tools/scale_flatten.py`，Z 軸朝上、正面朝 -Y、單位 mm。

## 步驟

1. **全身掃描**：找出所有比 0.8mm 薄的地方。

   ```bash
   python tools/thicken_details.py scan --file submissions/XXX/XXX_v1_10cm_flat.stl --png /tmp/scan.png
   ```

   會依連通分組列出（面積、最薄、中位數、位置 x/z 範圍），並存一張四個方向的熱圖
   （只有 < 0.8mm 的地方上色，紅 = 最薄）。
   用熱圖和參考圖決定要處理哪些部件。不是每一塊都需要處理，例如模型內部的瑕疵薄膜、
   埋在頭髮裡看不到的地方。

2. **每個部件找起點**：給一個框住部件的範圍。

   ```bash
   python tools/thicken_details.py inspect --file submissions/XXX/XXX_v1_10cm_flat.stl \
       --box x0 x1 z0 z1 --png /tmp/inspect.png
   ```

   看編號圖，挑出屬於這個部件的編號，記下 `seed` 座標。相鄰但不想動的東西不要挑
   （例如處理髮尾時別挑到耳朵、處理手指時別挑到拿著的道具）。

3. **寫進設定檔** `tools/details_config.json`，一個模型可以有多個部件，會依序處理：

   ```json
   "LAxxxxxxx_姓名": {
     "file": "submissions/LAxxxxxxx_姓名/LAxxxxxxx_姓名_v1_10cm_flat.stl",
     "parts": [
       {"name": "髮尾", "box": [x0, x1, z0, z1], "ignore_below": 0.0, "seeds": [[x, y, z]]},
       {"name": "劍",   "box": [x0, x1, z0, z1], "ignore_below": 0.0, "seeds": [[x, y, z]]}
     ]
   }
   ```

4. **執行**：

   ```bash
   python tools/thicken_details.py run LAxxxxxxx_姓名 --png /tmp/pushed.png
   ```

## 檢查結果

跟 thicken-glasses 一樣：每個部件的 `after.p5_mm` ≥ 0.8 左右、`crossing_edges` 不比原始模型多、
`watertight` 和 `faces_unchanged` 為 `true`、`bodies` 跟原始一樣；
再看紅色標記圖，確認只推了想推的部件，交接處是漸變、沒有段差。

原始模型的穿插數可以這樣量（同一個範圍）：

```python
from thicken_offset import crossing_edges
from thicken_details import in_box
crossing_edges(mesh, in_box(mesh, box, pad=1.0))
```

## 參數（`tools/thicken_offset.py`、`tools/thicken.py`）

| 參數 | 預設 | 意思 |
|---|---|---|
| `MIN_MM` | 0.8 | 目標最小厚度（光固化建議值） |
| `MARGIN` | 0.1 | 額外餘量 |
| `THIN_MM` | 1.6 | 比這個薄才算「薄部件」 |
| `NEAR_MM` | 0.6 | 比較厚的面離薄部件多近才一起算（讓推動量可以延伸到接合處） |
| `FALLOFF_MM` | 1.0 | 推動量沿表面距離漸變到 0 的距離 |
| `BORDER_FALLOFF_MM` | 0.5 | 靠近其他部位（臉、身體）交界時漸變到 0 的距離 |

## 已知限制

- **貼著其他表面的薄板**（例如貼在眼睛前面的墨鏡鏡片）：兩面都往外推，背面會撞到後面的
  東西而推不動，結果加不夠厚。需要改成「背面推不動時由正面多推」，還沒做。
- **尖端**（髮尾尖、指尖）往外推會變得比較鈍，這是加厚必然的代價。
- 比 `ignore_below` 還薄的面不處理。
