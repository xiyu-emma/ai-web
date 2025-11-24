# Windows 設定指南

本指南專為 Windows 用戶設計,幫助您在本地環境中執行 AI-Web 專案。

## 前置需求

### 1. 安裝 Docker Desktop

如果尚未安裝:
1. 下載 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 執行安裝程式
3. 重新啟動電腦
4. 啟動 Docker Desktop
5. 等待 Docker 引擎啟動完成（右下角圖示顯示綠色）

### 2. 安裝 Git

如果尚未安裝:
1. 下載 [Git for Windows](https://git-scm.com/download/win)
2. 執行安裝程式（使用預設設定即可）

## 快速開始

### 步驟 1: 克隆專案

開啟 **PowerShell** 或 **命令提示字元**:

```powershell
# 導航到您想要存放專案的位置
cd C:\Users\你的使用者名稱\Documents

# 克隆專案
git clone https://github.com/xiyu-emma/ai-web.git
cd ai-web

# 切換到新功能分支
git checkout feature/custom-folder-csv-export
```

### 步驟 2: 設定自訂儲存資料夾（可選）

如果您想使用自訂資料夾儲存音訊檔案:

```powershell
# 建立儲存資料夾
mkdir C:\AudioData

# 或使用其他位置
mkdir D:\MyProjects\AudioAnalysis
```

### 步驟 3: 修改 Docker Compose 設定

編輯 `docker-compose.yml`,在 `web` 和 `worker` 服務中添加 volume 掛載:

```yaml
services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./app:/app/app
      - ./static:/app/static
      # 添加以下行 - 根據您的實際路徑修改
      - C:\AudioData:C:\AudioData
      # 如果使用 D 槽
      - D:\MyProjects\AudioAnalysis:D:\MyProjects\AudioAnalysis
    # ... 其他設定

  worker:
    build: .
    command: celery -A app.celery worker --loglevel=info
    volumes:
      - ./app:/app/app
      - ./static:/app/static
      # 同樣的路徑
      - C:\AudioData:C:\AudioData
      - D:\MyProjects\AudioAnalysis:D:\MyProjects\AudioAnalysis
    # ... 其他設定
```

**重要**: 
- Windows 路徑使用反斜線 `\` 或雙反斜線 `\\`
- 路徑必須是絕對路徑（包含槽符號，如 `C:\` 或 `D:\`）

### 步驟 4: 啟動服務

在專案目錄中執行:

```powershell
# 構建並啟動所有容器
docker-compose up --build

# 或在背景執行
docker-compose up --build -d
```

首次啟動可能需要幾分鐘下載映像檔和安裝套件。

### 步驟 5: 初始化資料庫

開啟**新的 PowerShell 視窗**,執行:

```powershell
# 導航到專案目錄
cd C:\Users\你的使用者名稱\Documents\ai-web

# 初始化資料庫
docker-compose exec web flask init-db
```

### 步驟 6: 訪問應用程式

開啟瀏覽器,訪問:
```
http://localhost:5000
```

## 使用自訂資料夾功能

### 方法 1: 使用網頁介面選擇資料夾

1. 在上傳頁面,選擇「Choose Custom Folder」選項
2. 點擊「Browse Folder」按鈕
3. 瀏覽器會開啟資料夾選擇對話框
4. 選擇您想要的儲存位置
5. 路徑會自動填入

**注意**: 由於瀏覽器安全限制,某些瀏覽器可能無法完整顯示路徑。如果遇到此情況,可以:
- 雙擊路徑輸入框
- 選擇「是」以手動輸入路徑
- 輸入完整路徑,例如: `C:\AudioData`

### 方法 2: 手動輸入路徑

1. 選擇「Choose Custom Folder」選項
2. 雙擊路徑輸入框
3. 在確認對話框中選擇「是」
4. 手動輸入完整路徑:
   ```
   C:\AudioData
   D:\MyProjects\AudioAnalysis
   ```

### 路徑格式範例

✅ **正確格式**:
- `C:\AudioData`
- `C:\Users\張三\Documents\音訊分析`
- `D:\Projects\Audio\Results`

❌ **錯誤格式**:
- `C:/AudioData` (使用了正斜線)
- `AudioData` (缺少槽符號)
- `./AudioData` (相對路徑)

## Docker Desktop 設定

### 允許檔案共享

1. 開啟 Docker Desktop
2. 點擊右上角的齒輪圖示（設定）
3. 選擇「Resources」→「File Sharing」
4. 添加您要使用的資料夾路徑
5. 點擊「Apply & Restart」

### 增加記憶體配置（可選）

如果處理大型音訊檔案:
1. Docker Desktop → 設定 → Resources → Advanced
2. 調整 Memory 到 4GB 或更高
3. 點擊「Apply & Restart」

## 常見問題

### Q1: Docker Desktop 無法啟動

**解決方案**:
1. 確認已啟用 Hyper-V 或 WSL2
2. 以管理員身份執行 PowerShell:
   ```powershell
   # 啟用 Hyper-V
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
   
   # 或安裝 WSL2
   wsl --install
   ```
3. 重新啟動電腦

### Q2: Port 5000 已被佔用

**解決方案**:
修改 `docker-compose.yml`:
```yaml
ports:
  - "8000:5000"  # 改用 port 8000
```

然後訪問 `http://localhost:8000`

### Q3: 無法存取自訂資料夾

**可能原因**:
1. Docker Desktop 未允許該資料夾共享
2. 路徑格式錯誤
3. 資料夾不存在

**解決方案**:
1. 檢查 Docker Desktop 的 File Sharing 設定
2. 確認路徑使用反斜線 `\`
3. 手動建立資料夾:
   ```powershell
   mkdir C:\AudioData
   ```

### Q4: 權限被拒絕

**解決方案**:
1. 以管理員身份執行 PowerShell
2. 或修改資料夾權限:
   - 右鍵資料夾 → 內容 → 安全性
   - 編輯 → 添加 "Everyone"
   - 勾選「完全控制」

### Q5: 中文路徑顯示亂碼

**解決方案**:
1. 建議使用英文路徑名稱
2. 或確保 PowerShell 使用 UTF-8:
   ```powershell
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   ```

## 停止和清理

### 停止服務
```powershell
# 停止所有容器
docker-compose down

# 停止並刪除資料（包括資料庫）
docker-compose down -v
```

### 查看日誌
```powershell
# 查看所有日誌
docker-compose logs

# 查看特定服務
docker-compose logs web
docker-compose logs worker

# 持續追蹤
docker-compose logs -f
```

### 重新啟動
```powershell
# 重新啟動所有服務
docker-compose restart

# 僅重新啟動 worker
docker-compose restart worker
```

## 效能優化建議

### 1. 使用 SSD 儲存
將自訂資料夾設定在 SSD 上以提升處理速度。

### 2. 關閉防毒軟體掃描
將專案資料夾和 Docker 資料夾加入防毒軟體的排除清單。

### 3. 使用 WSL2 後端
Docker Desktop 設定中啟用「Use the WSL 2 based engine」以獲得更好的效能。

## 備份建議

### 備份資料庫
```powershell
# 匯出資料庫
docker-compose exec db mysqldump -u root -p audio_db > backup.sql

# 還原資料庫
Get-Content backup.sql | docker-compose exec -T db mysql -u root -p audio_db
```

### 備份分析結果
定期備份您的自訂資料夾:
```powershell
# 複製到備份位置
Copy-Item -Path "C:\AudioData" -Destination "D:\Backup\AudioData" -Recurse
```

## 下一步

1. ✅ 確認 Docker Desktop 正常運行
2. ✅ 啟動應用程式
3. ✅ 測試上傳功能
4. ✅ 測試自訂資料夾選擇
5. ✅ 測試 CSV 匯出
6. 📖 閱讀 [CSV Export Guide](CSV_EXPORT_GUIDE.md)
7. 📖 閱讀 [Custom Folder Guide](CUSTOM_FOLDER_GUIDE.md)

## 技術支援

如遇到問題:
1. 查看日誌: `docker-compose logs`
2. 檢查 Docker Desktop 狀態
3. 參考本文件的常見問題章節
4. 在 GitHub 上提交 Issue

祝使用愉快! 🚀
