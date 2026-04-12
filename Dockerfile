# 使用 Ultralytics 官方提供的 GPU 映像檔，已內建 CUDA, PyTorch 與常用 ML 依賴
FROM ultralytics/ultralytics:latest-python

# 設定容器內的工作目錄
WORKDIR /code

# 安裝額外的系統依賴 (基礎映像檔已含 ffmpeg 和 opencv 依賴，這裡補上 libsndfile1)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 複製依賴文件並安裝 Python 套件
# 這裡使用 --no-cache-dir 確保安裝的是最新版本且減少映像檔體積
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 將您本地的 app 資料夾，完整複製到容器的 /code/app 路徑下
COPY ./app ./app

# 設定生產環境變數
ENV FLASK_APP=app:create_app
ENV FLASK_ENV=production

# 預設啟動指令
CMD ["flask", "run", "--host=0.0.0.0"]