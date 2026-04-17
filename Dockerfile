# 使用 Ultralytics 官方提供的 GPU 映像檔
FROM ultralytics/ultralytics:latest

# 設定環境變項，確保 GPU 對容器可見 (Windows/Linux 通用宣告)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

# 設定容器內的工作目錄
WORKDIR /code

# 安裝額外的系統依賴 (基礎映像檔已含 ffmpeg 和 opencv 依賴，這裡補上 libsndfile1)
RUN sed -i 's/http:\/\//https:\/\//g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; \
    sed -i 's/http:\/\//https:\/\//g' /etc/apt/sources.list 2>/dev/null || true; \
    apt-get update && apt-get install -y \
    libsndfile1 \
    nvidia-cuda-toolkit \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 複製依賴文件並安裝 Python 套件
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 將您本地的 app 資料夾，完整複製到容器的 /code/app 路徑下
COPY ./app ./app

# 設定生產環境變數
ENV FLASK_APP=app:create_app
ENV FLASK_ENV=production

# 預設啟動指令
CMD ["flask", "run", "--host=0.0.0.0"]