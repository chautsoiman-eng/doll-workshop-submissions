---
name: thicken-glasses
description: 把公仔 STL 上太細的眼鏡（鏡框、鏡腳、鼻樑架）加厚到光固化可列印的 0.8mm 以上，用「直接改面 + 平滑漸變」，表面保持光滑、面數不變。學員模型戴眼鏡、要列印前使用。
---

# 眼鏡加厚（光固化，最小壁厚 0.8mm）

程式：`tools/glasses/thicken_glasses.py`（自帶全部演算法，不受通用版 `tools/thicken_details.py` 修改影響）
設定：`tools/glasses/glasses_config.json`
輸出：原檔名加 `_glasses.stl`，放在同一個資料夾

## 準備

```bash
pip install trimesh numpy scipy networkx embreex shapely matplotlib manifold3d pymeshfix
```

模型要先經過 `tools/scale_flatten.py`（等比縮放、底部切平），Z 軸朝上、正面朝 -Y。

## 步驟

1. **找出眼鏡的位置**：先量厚度、看熱圖，或直接用 `inspect` 加一個大概的範圍。
   範圍 `--box x0 x1 z0 z1` 是正面看的左右（x）和上下（z），單位 mm，框住整副眼鏡即可。

   ```bash
   python tools/glasses/thicken_glasses.py inspect --file submissions/XXX/XXX_v1_10cm_flat.stl \
       --box -20 20 63 78 --png /tmp/inspect.png
   ```

   會列出範圍內「連在一起、比 1.6mm 薄」的每一塊（由大到小編號），並存一張編號圖。

2. **挑起點（seeds）**：看編號圖，挑出屬於眼鏡的編號，用它們印出來的 `seed` 座標。
   - 通常 0 號是整副鏡框；鏡腳常被頭髮擋住，會是另外幾個小塊，也要挑。
   - **不要挑**：頭髮、瀏海、眉毛、睫毛、眼皮。挑錯的話那些地方也會被加粗。
   - 鏡框上方另外有橫桿（例如飛行員墨鏡），也要另外挑一個點。

3. **寫進設定檔** `tools/glasses/glasses_config.json`：

   ```json
   "LAxxxxxxx_姓名": {
     "file": "submissions/LAxxxxxxx_姓名/LAxxxxxxx_姓名_v1_10cm_flat.stl",
     "parts": [{"name": "眼鏡", "box": [-20, 20, 63, 78], "ignore_below": 0.3,
                "seeds": [[x, y, z], [x, y, z]]}]
   }
   ```

   `ignore_below: 0.3`：比 0.3mm 還薄的面不處理。有些模型在眼睛前面有一層 0.03～0.3mm 的
   「眼皮薄膜」，那是模型瑕疵，不是眼鏡，加厚它會讓眼睛變形。

4. **執行加厚**：

   ```bash
   python tools/glasses/thicken_glasses.py run LAxxxxxxx_姓名 --png /tmp/pushed.png
   ```

## 檢查結果（每一個都要看）

`run` 會印出一行 JSON：

| 欄位 | 合格標準 |
|---|---|
| `parts[].after.p5_mm` | 眼鏡最薄 5% 面積的厚度，≥ 0.8 左右 |
| `parts[].after.under_0.8_mm2` | 低於 0.8mm 的面積，應該只剩眼鏡面積的幾 % |
| `crossing_edges` | 面互相穿插的數量，**不能比原始模型多**（原始模型有時本來就有幾處） |
| `watertight` | 必須是 `true` |
| `faces_unchanged` | 必須是 `true`（直接改面不新增幾何） |
| `bodies` | 跟原始模型一樣（兩個人分開的模型本來就是 2） |

再看 `--png` 存的圖：紅色是被往外推的地方（全紅 = 0.2mm）。確認：
- 紅色只在眼鏡上，**沒有跑到頭髮、臉、眉毛**。
- 鏡框和鏡腳接合方塊、鼻樑架的交接處是**漸變**，沒有一圈灰色（沒推）的段差。

## 已知限制

- **貼在眼睛前面的鏡片（墨鏡鏡片）**：鏡片是一片薄板，背面往內推會撞到眼睛，
  只能推很少。例如陳誼緁 T 的墨鏡（鏡片 0.13～0.2mm）用這個版本加不到 0.8mm。
  這種情況先不要用這個 skill，改用通用版或手動處理。
- 推之前若會撞到其他表面，只會推一半距離，所以少數貼著方塊、頭髮的地方可能停在 0.7mm 多。
- 加厚量不大（通常 0.05～0.25mm），外觀圖上不明顯是正常的，要看厚度數字和紅色標記圖。

## 不要用的做法

`tools/thicken.py` 的 `thicken()`（撒點 + 體素外殼 + 布林聯集）：表面會有細顆粒感，
學員已經確認不能接受。只保留它的量厚度、找連通面等工具函式。
