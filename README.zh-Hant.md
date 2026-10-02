# Laya Snake on DGX Spark (GB10)

把 Laya 貪食蛇(從 Apple 專用的 laya-mlx 移植成 PyTorch 版)跑在 NVIDIA DGX Spark GB10 上。
**已在此機實測通過**:CUDA 推論 OK、蛇全速跑、GATE PASSED。

本機無 sudo,所以全部走**原生 Python venv**(不用 Docker)。

> 繁體中文版說明文件見 [USER_GUIDE.html](USER_GUIDE.html) 與本檔。
> English version: see [README.md](README.md)。

---

## License & Attribution

本專案以 **Apache License 2.0** 授權(見 [LICENSE](LICENSE)、[NOTICE](NOTICE)),
為下列上游的衍生作品:

- **laya-mlx** — https://github.com/mizorewww/laya-mlx (Apache-2.0), snake 遊戲與展示框架取自於此
- **Laya** — https://github.com/NandhaKishorM/laya (Apache-2.0), 模型 runtime / DecisionModel 取自於此

模型權重由 Convai Innovations 於 Hugging Face 另行下載(`convaiinnovations/laya`、
`convaiinnovations/laya-multilingual`),**不包含在本 repo**。

---

## 即時畫面(322M · 全速)

![Laya Snake FULL on GB10](docs/images/snake_322m_full.png)

> 上圖為 **Laya Snake FULL** 在 NVIDIA GB10 上全速執行的即時畫面(322M 模型)。
> 右側顯示決策機率、死路風險、食物可達性、推論時間與決策速度。

---

## 已在 ASUS DGX Spark GB10 (GX10) 上實測驗證

本套件是在一台**真正的 NVIDIA DGX Spark GB10**(機身 `gx10-0000`,aarch64)上建置並驗證的。
貪食蛇在 **CUDA** 上端到端跑通,CUDA 閘門測試通過。

**驗證環境**

| 項目 | 版本 / 值 |
|---|---|
| 主機 | DGX Spark GB10 · `gx10-0000` · aarch64 · Ubuntu |
| GPU | NVIDIA GB10 |
| Python | 3.12(venv 於 `~/laya-snake-dgx-work/venv`) |
| PyTorch | 2.14.0+cu130 (aarch64 CUDA build) |
| laya runtime | 0.3.22(來自 `NandhaKishorM/laya`) |
| 模型 | `convaiinnovations/laya-multilingual`(322M,預設)· `convaiinnovations/laya`(421M) |
| CUDA | `torch.cuda.is_available()` = True |

**驗證結果(322M · 全速)**

| 檢查 | 結果 |
|---|---|
| CUDA 閘門(`gate_check.py`) | PASSED |
| Live CUDA 載入 | OK(冷啟動 ~753 ms,154 tokens) |
| Snake · 全速 | ~107 steps/s,0 死亡,0 護盾介入(1500 步) |
| Snake · P50 推論 | 9.1 ms |
| GPU 利用(無頭) | ~74%,33.8 W |

在本機重現:

```bash
bash ~/laya-snake-dgx/bootstrap.sh --gate        # 重跑 CUDA 閘門
bash ~/laya-snake-dgx/run_snake.sh --headless --steps 600 --max-speed
```

即時截圖見上方「即時畫面」,完整性能比較見 `performance_card.html`。

---

## 一、檔案總覽

| 路徑 | 用途 |
|---|---|
| `~/laya-snake-dgx/bootstrap.sh` | **一鍵安裝/修復**(全新機或重裝,冪等,可無腦重跑) |
| `~/laya-snake-dgx/after_reboot.sh` | **重開機後一鍵恢復執行**(驗證 + 開視窗,不重裝) |
| `~/laya-snake-dgx/run_snake.sh` | 執行蛇的 launcher(已 export 關鍵 env) |
| `~/laya-snake-dgx/gate_check.py` | CUDA 閘門測試(自包含、離線) |
| `~/laya-snake-dgx/snake/` | 移植的 snake 套件 |
| `~/laya-snake-dgx/USER_GUIDE.html` | 中英雙語使用指南(瀏覽器開啟) |
| `~/laya-snake-dgx/performance_card.html` | 性能對比卡(瀏覽器開啟) |
| `~/laya-snake-dgx-work/venv/` | Python venv(磁碟上,重開機不消失) |
| `~/laya-snake-dgx-work/hf/` | 模型權重 cache(磁碟上,重開機不消失) |

**壓縮包**:`laya-snake-dgx.zip` 已內含以上全部檔案,scp 到 Spark 解壓即可。

---

## 二、從 zip 安裝(全新機,一條龍)

從「拿到 zip」到「蛇跑起來」共 4 步。

### Step 1 + 2 · 傳輸 + 解壓
```bash
# 工作站
scp laya-snake-dgx.zip asus@<SPARK_IP>:~/

# Spark
cd ~
unzip -o laya-snake-dgx.zip      # 生成 ~/laya-snake-dgx/
ls ~/laya-snake-dgx/
```

### Step 3 · 一鍵安裝(核心)
```bash
bash ~/laya-snake-dgx/bootstrap.sh
```
自動依序:**檢查前置 → 建 venv → 裝 torch 2.14 cu130(約 4GB)→ 裝 laya runtime → 裝 snake 套件 → 下載模型權重 → 建 launcher → 跑 CUDA gate**。
首次下載較大,約 10-30 分鐘。**最後必須看到「GATE PASSED」才算完成**。
只想驗證不重裝:`bash ~/laya-snake-dgx/bootstrap.sh --gate`。

### Step 4 · 啟動 venv + 執行
```bash
source ~/laya-snake-dgx-work/venv/bin/activate
python --version          # 應顯示 venv 的 Python 3.12
laya-snake --max-speed    # 或直接用 launcher(免 activate)
```

---

## 三、關於 venv 環境

- venv 位置:`~/laya-snake-dgx-work/venv`
- **每次要用 python 前必先 activate**:`source ~/laya-snake-dgx-work/venv/bin/activate`
- 驗證已進入:`which python` 應指向 venv 內路徑;`torch.cuda.is_available()` 應為 True
- 省事法:直接跑 `run_snake.sh` / `gate_check.py`,內部已解析 venv,免手動 activate

---

## 四、重開機後(最常用!)

venv 和權重在磁碟上,重開機不會消失,所以**不用重裝、不跑 bootstrap**,只需恢復執行。

```bash
bash ~/laya-snake-dgx/after_reboot.sh            # 驗證 + 開 FULL 互動窗
bash ~/laya-snake-dgx/after_reboot.sh --headless # 開無頭 benchmark
```
它會檢查 venv/torch/laya/權重 → 確認 :0 桌面 → 開「Laya Snake FULL」視窗,**並提示如何進 venv**。

> ⚠️ 前提:GNOME 桌面必須已登入(`:0` 起來)。剛開機未就緒就稍等或登入後再跑。

---

## 五、執行蛇的各種方式

```bash
~/laya-snake-dgx/run_snake.sh                          # 互動,12 FPS(看得清楚)
~/laya-snake-dgx/run_snake.sh --max-speed              # 互動,全速(~75 steps/s)
~/laya-snake-dgx/run_snake.sh --headless --steps 600 --max-speed   # 無頭 benchmark
~/laya-snake-dgx/run_snake.sh --max-speed --model convaiinnovations/laya   # 換英文 421M
~/laya-snake-dgx/run_snake.sh benchmark --rates 30 --sweep-steps 100 --seeds 101,102 --output ~/laya-snake-dgx-work/bench.json
```

### 互動控制鍵
| 鍵 | 動作 |
|---|---|
| 空白鍵 | 暫停/繼續 |
| ↑↓ 或 +− | 調整決策率 ±2 |
| R | 換新 seed 開局 |
| Q / Ctrl-C | 離開 |

---

## 六、兩個必知坑(否則會炸)

1. **`TORCH_DISABLE_NATIVE_JIT=1` 必須設** — torch 2.14 首次推論會用 gcc 現編 Triton kernel,要 `Python.h`(python3-dev),但本機無 sudo 裝不了。`run_snake.sh` 和 `gate_check.py` 都已設好;**別繞過 launcher 直接跑 `laya-snake`**,否則碰到 Triton 就報錯。
2. **用 PyTorch 版權重** — `convaiinnovations/laya-multilingual` 或 `convaiinnovations/laya`,不是 MLX 的 `aac6fef/...-mlx`。

---

## 七、常見問題

| 狀況 | 原因 | 解決 |
|---|---|---|
| Triton `Python.h` 錯誤 | 沒設 `TORCH_DISABLE_NATIVE_JIT` | 用 `run_snake.sh`,別直接 `laya-snake` |
| GPU 利用率低(8-16%) | 互動 12 FPS 刻意放慢 | 正常;`--max-speed` 拉到 ~74% |
| `torch.cuda.is_available()` False | NVIDIA 驅動問題 | 檢查驅動 / Container Toolkit |
| 權重找不到 | cache 路徑不對 | 重跑 `bootstrap.sh` 或確認 `~/laya-snake-dgx-work/hf` |
| 視窗沒開出來 | 桌面未登入 / dbus 未就緒 | 登入後重跑 `after_reboot.sh` |
| `python` 找不到套件 | 沒進 venv | `source ~/laya-snake-dgx-work/venv/bin/activate` |

---

## 八、性能摘要

FP16,端到端。GB10 於乾淨環境實測(無 Snake 在跑、GPU 閒置);Apple M3 Max 數據引自官方表。
峰值記憶體:GB10 以 MB 計,M3 Max 以 MiB 計(1 MiB ≈ 1.048 MB)。

| 指標 | GB10 421M | GB10 322M | M3 Max 421M | M3 Max 322M |
|---|---|---|---|---|
| 單問 · P50 延遲 | 15.8 ms | 7.2 ms | 13.42 ms | 7.39 ms |
| 單問 · P95 延遲 | 17.2 ms | 8.6 ms | 13.92 ms | 7.79 ms |
| 50 問批量吞吐 | 641 q/s | 1300 q/s | 146.8 q/s | 395.0 q/s |
| 峰值 GPU 記憶體(單問) | 2499 MB | 1638 MB | 943.6 MiB | 687.6 MiB |
| Snake 全速 steps/s | 48.4 | 107.6 | — | 63.6 |
| Snake 平均推論 | 20.4 ms | 9.1 ms | — | — |

> **怎麼讀這些數字**
> - **單問 · P50/P95** — 對一個簡短問題做一次完整推論的延遲(ms);P50 為中位數(典型),
>   P95 為最慢 5% 的上限(評估穩定性)。
> - **50 問批量吞吐** — 50 個問題一次合批的每秒處理數(q/s);固定開銷攤平後的真正運算上限。
> - **峰值 GPU 記憶體** — 做一個單問的峰值記憶體(權重 + forward 暫存)。
> - **Snake steps/s** — Snake demo 全速時每秒決策數(含推論 + 規劃)。「—」= M3 Max 未報告。
>
> **為什麼數字不同:** 單問延遲取決於每次呼叫的固定開銷(PyTorch eager dispatch、autocast、
> tokenization),所以 batch=1 時 GB10 與 M3 Max 接近;而批量吞吐反映原始矽晶運算力,
> Blackwell(GB10)是 M3 Max 的兩倍(421M:147→641,322M:395→1300 q/s)。
> GB10 峰值記憶體較高是 PyTorch 運行時保留足跡,非模型本身。

互動視覺化:見 [`performance_card.html`](performance_card.html)。
