# 2D 到 3D 公仔製作 — 學員檔案收集

這是 private repo,收集學員上傳的 STL 檔案、2D 參考圖、卡片 PDF。
檔案會依 `submissions/工號_姓名/` 分好資料夾。

不要在這個 repo 開 GitHub Pages —— 上傳網頁放在 `pages-portal` 分支,
兩個分支分開才能保持這裡的學員資料不對外公開。

## 上傳網頁網址(不含權杖)

```
https://chautsoiman-eng.github.io/doll-workshop-submissions/
```

**這串網址單獨開沒用**,一定要在後面接上 `#t=你的 fine-grained token` 才能上傳。
完整、能用的連結**不要存進這個 repo 或任何 git 裡**——GitHub 的 Secret Scanning
會直接擋下含 token 的 commit(就算是 private repo 也一樣),而且就算之後刪掉,
舊的 commit 歷史還是找得到,等於永遠留底。

**完整連結請存在 git 以外的地方**:手機備忘錄、密碼管理工具都可以。

- 產生新 token / 組出完整連結的步驟看 `doll-workshop` repo 裡的 `upload-portal/README.md`
- 上課前記得重新產生一組乾淨的 token,舊的直接去
  <https://github.com/settings/personal-access-tokens> 刪掉
