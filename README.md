# AI-Web 音訊分析與機器學習平台

一個基於 Flask 的音訊分析與機器學習平台,整合了音訊處理、頻譜圖生成、自動標記和 YOLOv8 模型訓練功能。

## 專案簡介

本專案提供了一個完整的音訊分析工作流程:
- **音訊上傳與處理**: 支援多種音訊格式的上傳與分段處理
- **頻譜圖生成**: 自動生成梅爾頻譜圖(Mel Spectrogram)或標準頻譜圖
- **手動標記**: 提供友善的網頁介面進行頻譜圖標記
- **自動標記**: 支援上傳已訓練的模型進行批次自動標記
- **模型訓練**: 整合 YOLOv8 圖像分類模型訓練功能
- **訓練報告**: 視覺化訓練結果與性能指標

## 核心技術棧

### 後端框架
- **Flask**: Web 應用框架
- **Flask-SQLAlchemy**: ORM 資料庫管理
- **Celery**: 非同步任務處理
- **Redis**: 訊息佇列與快取

### 資料庫
- **MySQL 8.0**: 主要資料庫

### 機器學習與音訊處理
- **PyTorch**: 深度學習框架
- **Ultralytics**: YOLOv8 模型訓練
- **Librosa**: 音訊特徵提取
- **TensorFlow**: 深度學習支援
- **SoundFile**: 音訊檔案處理

### 容器化
- **Docker & Docker Compose**: 容器化部署

## 系統架構

```
ai-web/
├── app/
│   ├── __init__.py          # Flask 應用初始化與設定
│   ├── main.py              # 主要路由與 API 端點
│   ├── models.py            # 資料庫模型定義
│   ├── tasks.py             # Celery 背景任務
│   ├── ai_model.py          # AI 模型相關功能
│   ├── audio_utils.py       # 音訊處理工具函式
│   └── templates/           # HTML 模板
├── static/
│   ├── uploads/             # 上傳的音訊檔案
│   ├── results/             # 處理後的頻譜圖與音訊片段
│   └── training_runs/       # 模型訓練結果
├── Dockerfile               # Docker 映像檔定義
├── docker-compose.yml       # Docker Compose 設定
└── requirements.txt         # Python 套件依賴
```

## 資料庫模型

### Upload (上傳記錄)
- 儲存音訊檔案上傳資訊與處理參數
- 追蹤處理狀態與進度

### Result (分析結果)
- 儲存每個音訊片段的頻譜圖路徑
- 關聯標籤資訊

### Label (標籤)
- 管理所有可用的標籤類別

### TrainingRun (訓練記錄)
- 記錄模型訓練任務的狀態與結果
- 儲存訓練參數與性能指標

## 快速開始

### 環境需求

- Docker
- Docker Compose

### 安裝與啟動

1. **克隆專案**
```bash
git clone https://github.com/xiyu-emma/ai-web.git
cd ai-web
```

2. **使用 Docker Compose 啟動所有服務**
```bash
docker-compose up --build
```

這將啟動以下服務:
- **web**: Flask 應用 (http://localhost:5000)
- **db**: MySQL 資料庫
- **redis**: Redis 伺服器
- **worker**: Celery 背景任務處理器

3. **初始化資料庫**

在另一個終端機視窗執行:
```bash
docker-compose exec web flask init-db
```

4. **訪問應用**

開啟瀏覽器並訪問: http://localhost:5000

## 功能說明

### 1. 音訊上傳與分析

**操作步驟**:
1. 在首頁選擇音訊檔案
2. 設定分析參數:
   - **頻譜圖類型**: 梅爾頻譜圖 (Mel) 或標準頻譜圖 (Normal)
   - **分段時長**: 每個片段的秒數
   - **重疊百分比**: 相鄰片段的重疊程度
   - **採樣率**: 保持原始採樣率或重新採樣
   - **聲道處理**: 單聲道或立體聲
3. 點擊「上傳並分析」

**處理流程**:
- 系統會將音訊切分為多個片段
- 為每個片段生成頻譜圖
- 儲存音訊片段與頻譜圖到資料庫

### 2. 查看分析歷史

在「分析歷史」頁面可以:
- 查看所有上傳任務的狀態
- 監控處理進度
- 刪除不需要的記錄
- 查看詳細分析結果

### 3. 頻譜圖標記

**手動標記**:
1. 進入「標記頁面」
2. 建立標籤類別
3. 為每個頻譜圖選擇對應標籤

**自動標記**:
1. 上傳已訓練的模型檔案 (.pt 或 .h5)
2. 系統會自動為所有頻譜圖進行預測與標記

### 4. 模型訓練

**訓練步驟**:
1. 在「分析歷史」頁面選擇已標記的資料集
2. 點擊「開始訓練」
3. 選擇 YOLOv8 模型大小 (n/s/m/l/x)
4. 系統會在背景執行訓練

**訓練過程**:
- 自動組織訓練資料集
- 執行 YOLOv8 圖像分類訓練
- 生成訓練報告與視覺化結果

### 5. 查看訓練報告

在「訓練狀態」頁面可以:
- 查看所有訓練任務的狀態
- 檢視訓練進度
- 查看詳細的訓練報告,包括:
  - 訓練曲線圖
  - 混淆矩陣
  - 驗證集預測結果
  - 性能指標

## API 端點

### 任務狀態查詢
- `GET /api/upload/<upload_id>/status` - 查詢分析任務狀態
- `GET /api/training/<run_id>/status` - 查詢訓練任務狀態

### 標籤管理
- `GET /api/labels` - 獲取所有標籤
- `POST /api/labels` - 建立新標籤
- `DELETE /api/labels/<label_id>` - 刪除標籤

### 結果更新
- `POST /api/results/<result_id>/label` - 更新頻譜圖標籤

## 環境變數設定

以下環境變數可在 `docker-compose.yml` 中調整:

```yaml
# Flask 設定
FLASK_APP=app:create_app
FLASK_ENV=development

# 資料庫連線
DATABASE_URL=mysql+pymysql://user:password@db/audio_db

# Celery 設定
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# 時區設定
TZ=Asia/Taipei
```

## 開發說明

### 本地開發模式

如果想在不使用 Docker 的情況下開發:

1. **安裝依賴**
```bash
pip install -r requirements.txt
```

2. **設定環境變數**
```bash
export DATABASE_URL="mysql+pymysql://user:password@localhost/audio_db"
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
export FLASK_APP="app:create_app"
```

3. **啟動 Flask 應用**
```bash
flask run
```

4. **啟動 Celery Worker**
```bash
celery -A app.celery worker --loglevel=info
```

### 資料庫遷移

如果修改了資料庫模型,需要重新初始化:
```bash
docker-compose exec web flask init-db
```

## 常見問題

### Q: 如何修改資料庫密碼?
A: 在 `docker-compose.yml` 中修改 `MYSQL_PASSWORD` 和 `DATABASE_URL` 的密碼部分。

### Q: 訓練任務失敗怎麼辦?
A: 檢查 Celery worker 的日誌: `docker-compose logs worker`

### Q: 如何增加更多的 Celery worker?
A: 在 `docker-compose.yml` 中複製 worker 服務並重新命名。

### Q: 頻譜圖生成速度慢?
A: 可以調整分段時長和重疊百分比,減少生成的頻譜圖數量。

## 效能優化建議

1. **使用 GPU 加速**: 在 `docker-compose.yml` 中為 worker 服務添加 GPU 支援
2. **增加 Celery Workers**: 平行處理多個任務
3. **調整資料庫連線池**: 修改 SQLAlchemy 的 `pool_size` 參數
4. **使用持久化儲存**: 將訓練結果儲存到外部儲存服務

## 授權

本專案為個人專案,請聯繫專案所有者以獲取使用授權。

## 聯絡方式

- GitHub: [@xiyu-emma](https://github.com/xiyu-emma)
- 專案連結: [https://github.com/xiyu-emma/ai-web](https://github.com/xiyu-emma/ai-web)

## 更新日誌

### v1.0.0 (2025)
- 初始版本發布
- 支援音訊上傳與頻譜圖生成
- 整合 YOLOv8 模型訓練
- 提供完整的標記與訓練工作流程