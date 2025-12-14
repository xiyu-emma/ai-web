from . import db
from datetime import datetime
from zoneinfo import ZoneInfo
import json

# 定義時區
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

# --- 1. 基礎資訊表 (Project, Point, Recoder) ---

class ProjectInfo(db.Model):
    """
    表二、Project Info 紀錄計畫資訊 
    """
    __tablename__ = 'project_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    name = db.Column(db.String(255), nullable=False, comment='計畫名稱')
    area = db.Column(db.String(255), nullable=True, comment='執行區域')
    start_time = db.Column(db.DateTime, nullable=True, comment='執行開始時間')
    end_time = db.Column(db.DateTime, nullable=True, comment='執行結束時間')
    
    # 關聯
    points = db.relationship('PointInfo', backref='project', lazy=True)

class RecoderInfo(db.Model):
    """
    表四、Recode Info 紀錄錄音機資訊 
    """
    __tablename__ = 'recoder_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    brand = db.Column(db.String(100), nullable=False, comment='儀器廠牌')
    recoder = db.Column(db.String(100), nullable=False, comment='儀器型號')
    sn = db.Column(db.String(100), nullable=False, comment='儀器序號')
    sen = db.Column(db.Float, nullable=True, comment='靈敏度')
    high_gain = db.Column(db.Float, nullable=True, comment='高增益值')
    low_gain = db.Column(db.Float, nullable=True, comment='低增益值')
    
    # 表五、儀器狀態定義 
    # 1:服役中, 2:停用, 3:維修中, 4:校準中, 5:損壞, 6:報廢, 7:遺失, 8:出借中
    status = db.Column(db.Integer, default=1, comment='儀器狀態')
    
    belong = db.Column(db.String(100), nullable=True, comment='儀器產權')
    
    # 關聯
    points = db.relationship('PointInfo', backref='recoder', lazy=True)

class PointInfo(db.Model):
    """
    表三、Point Info 紀錄計畫內的點位資訊 
    """
    __tablename__ = 'point_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    name = db.Column(db.String(255), nullable=False, comment='點位名稱')
    phase = db.Column(db.Integer, default=0, nullable=False, comment='點位階段')
    start_time = db.Column(db.DateTime, nullable=True, comment='執行開始時間')
    end_time = db.Column(db.DateTime, nullable=True, comment='執行結束時間')
    gps_lat = db.Column(db.Float, nullable=True, comment='點位緯度')
    gps_lon = db.Column(db.Float, nullable=True, comment='點位經度')
    depth = db.Column(db.Float, nullable=True, comment='點位水深')
    fs = db.Column(db.Integer, nullable=True, comment='取樣頻率')
    return_success = db.Column(db.Boolean, nullable=True, comment='回收成功')
    
    # Foreign Keys
    project_id = db.Column(db.Integer, db.ForeignKey('project_info.id'), nullable=True, comment='所屬計畫')
    recoder_id = db.Column(db.Integer, db.ForeignKey('recoder_info.id'), nullable=True, comment='所使用的錄音機')
    
    # 關聯
    audios = db.relationship('AudioInfo', backref='point', lazy=True)

# --- 2. 核心音檔表 (取代原本的 Upload) ---

class AudioInfo(db.Model):
    """
    表六、Audio Info 音檔資訊 
    (此表整合了系統原有的 Upload 功能)
    """
    __tablename__ = 'audio_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    file_name = db.Column(db.String(255), nullable=False, comment='檔案名稱') # 對應原 original_filename
    file_path = db.Column(db.String(255), nullable=False, comment='檔案路徑') # 原始檔存放路徑
    file_type = db.Column(db.String(50), nullable=False, comment='檔案類型')
    record_time = db.Column(db.DateTime, default=lambda: datetime.now(TAIPEI_TZ), comment='錄製時間') # 這裡暫代上傳時間
    record_duration = db.Column(db.Float, nullable=True, comment='錄製長度(s)')
    target = db.Column(db.String(100), nullable=True, comment='目標物')
    target_type = db.Column(db.Integer, nullable=True, comment='目標物類型')
    fs = db.Column(db.Integer, nullable=True, comment='取樣頻率')
    
    # Foreign Keys
    point_id = db.Column(db.Integer, db.ForeignKey('point_info.id'), nullable=True, comment='所屬點位')

    # --- 以下為系統運作所需欄位 (非計畫書定義，但為程式運行必要) ---
    # 用於儲存分析後的結果資料夾路徑 (static/results/...)
    result_path = db.Column(db.String(255), nullable=True) 
    # 儲存參數 (JSON)
    params = db.Column(db.Text, nullable=True)
    # 任務狀態 (PENDING, PROCESSING, COMPLETED, FAILED)
    status = db.Column(db.String(50), default='PENDING', nullable=False)
    # 進度 (0-100)
    progress = db.Column(db.Integer, default=0, nullable=False)

    # 關聯 (對應切片後的結果 Result)
    results = db.relationship('Result', backref='audio_info', lazy=True, cascade="all, delete-orphan")
    
    # 關聯 (對應計畫書的三種辨識資訊)
    cetaceans = db.relationship('CetaceanInfo', backref='audio_info', lazy=True, cascade="all, delete-orphan")
    ships = db.relationship('ShipInfo', backref='audio_info', lazy=True, cascade="all, delete-orphan")
    turbines = db.relationship('TurbineInfo', backref='audio_info', lazy=True, cascade="all, delete-orphan")

    def get_params(self):
        try:
            return json.loads(self.params) if self.params else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    # 為了相容舊程式碼，增加 original_filename 屬性別名
    @property
    def original_filename(self):
        return self.file_name
    
    @property
    def upload_timestamp(self):
        return self.record_time

# --- 3. 分析結果與標記表 (依據計畫書擴充) ---

class CetaceanInfo(db.Model):
    """
    表七、Cetacean Info 鯨豚資訊 
    """
    __tablename__ = 'cetacean_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    audio_id = db.Column(db.Integer, db.ForeignKey('audio_info.id'), nullable=False, comment='所屬音檔')
    
    start_sample = db.Column(db.Integer, nullable=True, comment='音檔中開始位置')
    end_sample = db.Column(db.Integer, nullable=True, comment='音檔中結束位置')
    event_duration = db.Column(db.Integer, nullable=True, comment='持續時間(s)')
    
    # 表八、鯨豚聲紋類型 
    # 0:未知, 1:上升型, 2:下降型, 3:U型, 4:倒U型, 6:sin型, 7:嘎搭聲, 8:突發脈衝聲
    event_type = db.Column(db.Integer, nullable=True, comment='類型')
    
    # 0: 人工, 1: AI
    detect_type = db.Column(db.Integer, default=0, comment='辨識類型')

class ShipInfo(db.Model):
    """
    表九、Ship Info 船舶資訊 
    """
    __tablename__ = 'ship_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    audio_id = db.Column(db.Integer, db.ForeignKey('audio_info.id'), nullable=False, comment='所屬音檔')
    
    start_sample = db.Column(db.Integer, nullable=True, comment='音檔中開始位置')
    end_sample = db.Column(db.Integer, nullable=True, comment='音檔中結束位置')
    event_duration = db.Column(db.Integer, nullable=True, comment='持續時間(s)')
    
    # 表十、船舶類型定義 
    # 1:油輪, 2:散裝船, 3:貨輪, 4:漁船, 5:快艇, 6:舷外機船
    event_type = db.Column(db.Integer, nullable=True, comment='類型')

class TurbineInfo(db.Model):
    """
    表十一、Turbine Info 風機資訊 
    """
    __tablename__ = 'turbine_info'
    
    id = db.Column(db.Integer, primary_key=True, comment='id')
    audio_id = db.Column(db.Integer, db.ForeignKey('audio_info.id'), nullable=False, comment='所屬音檔')
    
    start_sample = db.Column(db.Integer, nullable=True, comment='音檔中開始位置')
    end_sample = db.Column(db.Integer, nullable=True, comment='音檔中結束位置')
    event_duration = db.Column(db.Integer, nullable=True, comment='持續時間(s)')
    
    # 表十二、風機狀態定義 
    # 1:運轉, 2:打樁
    event_type = db.Column(db.Integer, nullable=True, comment='類型')

# --- 4. 系統運作與訓練輔助表 (保留原系統架構以支援 MLOPS 流程) ---

class Result(db.Model):
    """
    儲存切割後的頻譜圖檔案資訊，用於網頁顯示與訓練資料集生成。
    """
    __tablename__ = 'results'
    id = db.Column(db.Integer, primary_key=True)
    
    # 注意：這裡改為關聯到新的 AudioInfo
    upload_id = db.Column(db.Integer, db.ForeignKey('audio_info.id'), nullable=False)
    
    audio_filename = db.Column(db.String(255), nullable=True)
    spectrogram_filename = db.Column(db.String(255), nullable=False)
    spectrogram_training_filename = db.Column(db.String(255), nullable=False)
    
    label_id = db.Column(db.Integer, db.ForeignKey('labels.id'), nullable=True)
    label = db.relationship('Label', backref='results')
    
    @property
    def audio_url(self):
        if self.audio_filename:
            return f"{self.audio_info.result_path}/{self.audio_filename}"
        return None

    @property
    def spectrogram_url(self):
        return f"{self.audio_info.result_path}/{self.spectrogram_filename}"

    @property
    def spectrogram_training_url(self):
        return f"{self.audio_info.result_path}/{self.spectrogram_training_filename}"

class Label(db.Model):
    """
    標籤管理表 (用於 API /api/labels) [cite: 7]
    """
    __tablename__ = 'labels'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.String(255), nullable=True)

class TrainingRun(db.Model):
    """
    紀錄 AI 模型訓練任務狀態與結果 [cite: 7]
    """
    __tablename__ = 'training_runs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(TAIPEI_TZ))
    status = db.Column(db.String(50), default='PENDING')
    results_path = db.Column(db.String(255), nullable=True)
    params = db.Column(db.Text, nullable=True) 
    metrics = db.Column(db.Text, nullable=True)
    progress = db.Column(db.Integer, default=0, nullable=False)

    def get_params(self):
        if self.params:
            try:
                return json.loads(self.params)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    def get_metrics(self):
        if self.metrics:
            try:
                return json.loads(self.metrics)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}