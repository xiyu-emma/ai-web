# Audio AI Web Platform

這是一個基於 Web 的音訊 AI 分析平台，專為海洋生物（如鯨豚）聲音偵測與標記而設計。系統提供音訊上傳、頻譜圖轉換、AI 自動偵測、人工標記校正以及模型訓練功能。

## 主要功能

### 音訊上傳與處理
- 支援單檔與資料夾批次上傳 (.wav, .mp3)
- 自動將長音訊切割並轉換為頻譜圖 (Spectrogram)
- **平行化處理與 GPU 加速**：長檔音訊支援多核心平行處理解析，Mel 頻譜圖生成具備 VRAM 動態分配及 GPU 加速能力。
- 支援多種頻譜圖類型：Mel、STFT、DEMON、Envelope Spectrum、YAMNet Log Mel
- 可調整頻譜圖參數：FFT 大小、視窗重疊、Mel 濾波器數量、頻率範圍等

### AI 自動偵測
- 整合 YOLOv8 分類模型，自動辨識頻譜圖中的聲音事件
- 支援多種 CNN 模型：YOLOv8n/s、ResNet18、EfficientNet-B0
- 隨需載入模型機制，節省資源

### 標記與驗證系統
- 提供 Web 介面檢視分析結果（全新海洋主題專業極簡風格介面）
- 支援人工標記與修正 AI 預測結果
- **AI 輔助標記保留機制**：若音檔片段已有「人工標記」，執行 AI 輔助標記時會自動保留原始人工標記結果，並將 AI 的預測儲存於背景，方便在介面中比對 AI 的誤判情形。
- 全面擴充至 **10 種標籤類別**，包含多種鯨豚聲紋型態及新增「10. 未知聲音 (Unknown Vocalization)」，並獨立出「0. 無標註 (No Label)」。
- **進階框選標記**：在頻譜圖上繪製矩形框標記特定聲紋區域
  - 即時繪製與刪除框選
  - 鍵盤快捷鍵支援（S=儲存、方向鍵=翻頁、Ctrl+Z=復原）
- **互動式放大檢視**：支援在圖片放大對話框(Modal)中直接進行標記，並可直接導航至前後頻譜圖。

### 模型訓練
- 利用標記好的資料，直接在平台上一鍵啟動模型訓練
- 支援多種訓練參數自訂：epochs、batch size、learning rate、image size、以及 Loss Function (預設 Cross Entropy)
- **進階訓練設定與抽樣**：可選擇特定標籤參與訓練，自訂各標籤的樣本數量，並支援隨機抽樣 (Random Sampling) 機制以平衡資料集。
- **資料品質控管**：自動檢查並阻擋包含「未分類 (標籤 0)」的資料進入訓練集，以確保模型品質。
- 訓練完成後可檢視詳細報告（Confusion Matrix、準確率、F1-score 等）

### 資料集與批次管理
- 支援打包下載「圖片 + 標籤」的資料集 (Zip 格式)
- **資料批次匯出 / 匯入 workflow**：
  - 支援將標記結果批次匯出為標準 CSV/Excel 格式。
  - 支援使用者於 Excel 編輯標籤後，整批匯入 (Batch Import) 回系統，後端會自動比對並更新資料庫紀錄。
- **框選標記匯出**：下載資料集時自動產生 `bbox_annotations.csv` 包含時間 (秒) 與頻率 (Hz) 範圍，方便未來訓練物件偵測模型。

### 背景任務處理與 API 整合
- 使用 Celery + Redis 處理耗時的音訊分析與模型訓練任務，經優化可支援多任務高併發平行處理 (Parallel Processing)。
- 即時進度更新，確保網頁操作順暢。
- 新增對外整合之 Partner API（詳細規格請參考 `partner_api_doc.md`）。

## 技術棧

- **Backend Framework**: Flask (Python)
- **Database**: MySQL 8.0
- **Task Queue**: Celery + Redis
- **AI / ML**:
  - Ultralytics YOLOv8 (Image Classification)
  - PyTorch (CNN Models)
  - Librosa (Audio Processing)
- **Containerization**: Docker & Docker Compose

## 快速開始

### 前置需求

請確保您的電腦已安裝：
- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/install/)

### 安裝與執行

1. **Clone 專案**
   ```bash
   git clone <repository_url>
   cd audio-ai_web
   ```

2. **啟動服務**
   使用 Docker Compose 一鍵啟動所有服務 (Web, Database, Redis, Worker)：
   ```bash
   docker-compose up --build
   ```

3. **訪問應用程式**
   服務啟動後，請打開瀏覽器訪問：
   [http://localhost:5000](http://localhost:5000)

### 預設資料庫帳號
如果在 `docker-compose.yml` 中未修改，預設資料庫連線資訊如下：
- **Port**: 3306
- **User**: user
- **Password**: password
- **Database**: audio_db

## 專案結構

```
.
├── app/
│   ├── static/               # 靜態資源目錄
│   │   ├── css/              # 網頁樣式表 (CSS)
│   │   ├── results/          # 分析結果儲存區 (頻譜圖、切割音訊)
│   │   ├── training_runs/    # 模型訓練產出 (權重檔、訓練圖表)
│   │   └── uploads/          # 原始上傳音訊備份
│   ├── templates/            # Flask HTML 模板
│   │   ├── base.html         # 基礎頁面佈局
│   │   ├── index.html        # 音訊上傳介面
│   │   ├── history.html      # 歷史紀錄列表
│   │   ├── result.html       # 分析結果詳細頁面
│   │   ├── label.html        # 人工標記介面
│   │   ├── label_advanced.html   # 進階框選標記介面
│   │   ├── training_status.html  # 訓練任務狀態
│   │   └── training_report.html  # 訓練詳細報告
│   ├── routers/              # 路由視圖目錄
│   │   ├── __init__.py
│   │   ├── api.py            # API 介面路由
│   │   ├── download.py       # 下載相關路由
│   │   ├── labels.py         # 標記相關路由
│   │   ├── pages.py          # 網頁視圖路由
│   │   ├── status.py         # 狀態檢查路由
│   │   ├── training.py       # 訓練相關路由
│   │   └── upload.py         # 上傳相關路由
│   ├── __init__.py           # Flask App 初始化
│   ├── ai_model.py           # AI 推論核心邏輯
│   ├── audio_utils.py        # 音訊處理工具庫
│   ├── main_router.py        # 註冊所有路由的核心檔案
│   ├── models.py             # 資料庫模型定義
│   └── tasks.py              # Celery 背景任務
├── docker-compose.yml        # Docker 服務編排配置
├── Dockerfile                # Web Service 容器建置檔
├── main.py                   # 應用程式啟動入口
└── requirements.txt          # Python 專案依賴套件
```

## 關於 AI 模型

系統預設會尋找 `app/models/best.pt` 作為推論模型。

1. **初次使用**：如果您還沒有模型，推論功能將暫時無法使用，但您仍可使用標記功能。
2. **訓練模型**：在平台上標記足夠資料後，至「訓練狀態」頁面啟動訓練。
3. **部署新模型**：訓練成功後，請將訓練出的最佳模型檔案（通常位於 `static/training_runs/...`）複製並重新命名為 `app/models/best.pt` 即可生效。

## 進階框選標記使用說明

1. 在標記頁面點擊「進階框選標記」按鈕
2. 在頻譜圖上拖曳滑鼠繪製矩形框
3. 選擇標籤類別（鯨豚聲紋類型或環境噪音）
4. 按 `S` 或點擊「儲存標記」
5. 下載資料集時會自動包含 `bbox_annotations.csv`，記錄每個框的時間（秒）和頻率（Hz）範圍

## 資料庫模型

主要資料表：
- `audio_info`: 上傳音訊基本資訊
- `results`: 頻譜圖檔案路徑
- `cetacean_info`: 鯨豚事件資訊與標籤 (包含 `ai_event_type` 輔助檢驗)
- `bbox_annotations`: 框選標記座標與標籤
- `labels`: 標籤定義表
- `training_runs`: 訓練任務紀錄
