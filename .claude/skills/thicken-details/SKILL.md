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
- 鏡腳：**學員最後決定不動**（用 `0.4 0 0.5` 執行）。試過的做法都覺得怪：
  整支桿子沿左右方向往內掃 0.3mm（取離桿子中心線 0.6mm 內、朝內那半邊的所有面）。
  只拉一部分（例如只拉法線 > 0.5 的面）學員不接受：只有一小塊突出很怪，而且邊界沿著三角形，從上面看是鋸齒。
  整半邊都拉，上下兩邊的牆剛好沿著鏡腳的輪廓，就不會有鋸齒。法線太貼近側面（< 0.15）的面不要拉，柱體太扁，存檔時會壞。
  方向指向頭中心、拉 0.45mm 時會壞。
- 鏡片：只能往內（背面）加厚，不能往外；鏡片背面不能超過鏡框背面；離眼球近的地方要停在眼球前，不能撞進去（撞進去模型會壞）。
- 不要用平板替換鏡片（彎的鏡片會露出分界線），也不要在加厚鏡框之後才找鏡片輪廓（會抓到新的面，輪廓變形）。

## 填空心（百褶裙等「空心的盒子」）

`tools/fill_cavities.py`：有些衣服看起來是厚的，其實是空心的，例如謝雅慧的百褶裙。
每一褶都是外牆、內牆各約 0.38mm 的空心盒子。這種情況不要往外推，也不要從內側拉，否則會把褶子弄亂。
改成每隔 0.05mm 水平切一刀，把切面上封閉的小洞（面積 < 12mm²、離中心軸 6mm 以上）擠成薄片填進去。
牆不動，外面形狀完全不變（實測 99.8% 的頂點位置沒變）。
下襬最底下那一小段，空心是開口的（切面沒有封閉），不會填。
實例：`tools/details/xie_skirt.py`。要判斷是不是空心，先看幾個高度的水平橫切面。

`tools/thicken_inner.py`：真正的單層薄片，只從內側沿頂點法線整片往內拉，外側不動。
`save_clean()` 是存 STL 前的清理（manifold simplify，再不行才用 pymeshfix）。

## 整片平移頂點（劍身這種兩面平的配件）

`tools/details/lin_sword.py`：林羽倫的劍。劍身的橫切面是六角形，中間 1.45mm，刀刃只有 0.1～0.2mm。
做法：沿劍面的法線，正面的頂點往外移 0.5mm，背面的頂點往另一邊移 0.5mm。
刀刃上的三角形會被拉開，變成約 1mm 寬的平整側面；從正面看，劍的輪廓完全不變，面數也不變。
靠近護手的 5mm 內，移動量漸變到 0（漸變區太靠近護手的花紋，會留下一道接縫）。
- **學員不接受**：劍的刀刃不能動、劍不能變鈍，刀刃被拉開成平面看起來像階梯，很奇怪。這支劍已經撤回，維持原樣。
  劍身中間原本就有 1.4～1.6mm，已經夠厚。
- **改用厚度比例縮放**（學員提議）：只把劍身沿厚度方向（以每段劍身的中心面為準）放大 K 倍，長度、寬度不變，
  刀刃一樣是尖的，形狀只是等比例變厚。學員要整支劍身（含護手上方有花紋的那段）一起放大，進到護手的 1.5mm 內倍率才漸變回 1。`python tools/details/lin_sword.py 1.5`
- 不要用擠出小柱體再聯集：碎面太多，存檔修補時 pymeshfix 會把整支劍刪掉。

## 已知限制

- **很小、網格很粗的細部**（例如疊放方案裡只有 37～57mm 高的公仔的眼鏡框）：推頂點會讓表面變皺，
  目前不處理。
- **尖端**（髮尾尖、指尖）往外推會變得比較鈍，這是加厚必然的代價。
- 比 `ignore_below` 還薄的面不處理。
- `plate` 模式在鏡片和鏡框交接處可能多出幾處面穿插，要用切片軟體確認。
- 布林聯集後存成 STL（float32）時，極小的碎邊會黏在一起變成破洞。先用 manifold 的 `simplify(0.002)` 合併，剩下的再交給 pymeshfix。
