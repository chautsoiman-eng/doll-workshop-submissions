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

## 部件選項

| 選項 | 用途 |
|---|---|
| `"plate": true` | **平板模式**（例如貼在眼睛前面、背面推不動的墨鏡鏡片）。seeds 點在鏡片**正面**，每片一個點。照原本鏡片的輪廓做一片 0.9mm、兩面平整的新鏡片，背面貼齊原本的背面（往前 0.03mm 避免面重疊），輪廓往內縮 0.12mm 讓鏡框露出來，再布林聯集。原本的面不動，表面保證平整。 |
| `"plate_inward": true` | 搭配 `plate`：新鏡片的正面貼齊原本的正面，整片往內（眼睛方向）長，從正面看完全不變。鏡框、鏡腳不動。先量鏡片後面到眼睛的空間，要大於加厚量。 |
| `"plate_front_out": 0.2, "plate_back_in": 0.3` | 搭配 `plate`：直接指定正面往外、背面往內各加多少 mm（總厚度 = 原本 + 兩者）。陳誼緁 T 用這個（學員指定）。 |
| `"plate_inset": 0.0` | 搭配 `plate`：新鏡片輪廓往內縮多少 mm（預設 0.12）；往內長時用 0，讓鏡片填滿到鏡框。 |
| `"push_dir": "auto"` | 自動單面推：看每個薄處兩面前方的空間，空間比較小的那面不動、另一面推足。**實測在粗網格上表面會不平整，平板請改用 `plate`。** |
| `"refine": true` | 先把很粗的薄三角形切細（不會產生裂縫），讓中間有頂點可以推。 |

## 擠出面（不改形狀的加厚）

`tools/extrude_faces.py`：選一側的整片面（例如鏡框背面、鏡腳內側），沿固定方向平移拉出一段距離。
原本的面都不動，從正面看形狀完全不變，只在那一側變厚。每個三角形各擠出一個小柱體再聯集，
彎曲的面也不會自己交叉。學員偏好這個做法勝過推頂點或重做零件。
實例：`tools/glasses/chen_T_5cm.py`（陳誼緁 T）。學員確認的結果：
- 鏡框背面拉出 0.4mm：OK。只拉「鏡框圈」（離鏡片邊緣 0.6mm 內、鏡片平面後方 0.6mm 內），鉸鏈和鏡腳根部不要拉，否則鏡框和鏡腳之間會長出一塊原本沒有的斜面。眼鏡碰到臉沒關係。
- 鏡腳內側：沿左右方向往內拉 0.3mm OK。只拉法線明確朝內（> 0.5）、離鏡框 1mm 以外的面；之前拉 0.45mm、方向指向頭中心、沒排除鉸鏈時會壞。
- 鏡片：只能往內（背面）加厚，不能往外；鏡片背面不能超過鏡框背面；離眼球近的地方要停在眼球前，不能撞進去（撞進去模型會壞）。
- 不要用平板替換鏡片（彎的鏡片會露出分界線），也不要在加厚鏡框之後才找鏡片輪廓（會抓到新的面，輪廓變形）。

## 已知限制

- **很小、網格很粗的細部**（例如疊放方案裡只有 37～57mm 高的公仔的眼鏡框）：推頂點會讓表面變皺，
  目前不處理。
- **尖端**（髮尾尖、指尖）往外推會變得比較鈍，這是加厚必然的代價。
- 比 `ignore_below` 還薄的面不處理。
- `plate` 模式在鏡片和鏡框交接處可能多出幾處面穿插，要用切片軟體確認。
