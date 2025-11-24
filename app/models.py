from . import db
from datetime import datetime
from zoneinfo import ZoneInfo
import json

TAIPEI_TZ = ZoneInfo("Asia/Taipei")

class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_timestamp = db.Column(db.DateTime, default=lambda: datetime.now(TAIPEI_TZ))
    result_path = db.Column(db.String(255), nullable=False)
    custom_folder_path = db.Column(db.String(255), nullable=True)
    params = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default='PENDING', nullable=False)
    progress = db.Column(db.Integer, default=0, nullable=False)
    results = db.relationship('Result', backref='upload', lazy=True, cascade="all, delete-orphan")

    def get_params(self):
        try:
            return json.loads(self.params)
        except (json.JSONDecodeError, TypeError):
            return {}

class Result(db.Model):
    __tablename__ = 'results'
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), nullable=False)
    audio_filename = db.Column(db.String(255), nullable=True)
    spectrogram_filename = db.Column(db.String(255), nullable=False)
    spectrogram_training_filename = db.Column(db.String(255), nullable=False)
    label_id = db.Column(db.Integer, db.ForeignKey('labels.id'), nullable=True)
    label = db.relationship('Label', backref='results')
    
    @property
    def audio_url(self):
        if self.audio_filename:
            return f"{self.upload.result_path}/{self.audio_filename}"
        return None

    @property
    def spectrogram_url(self):
        return f"{self.upload.result_path}/{self.spectrogram_filename}"

    @property
    def spectrogram_training_url(self):
        return f"{self.upload.result_path}/{self.spectrogram_training_filename}"

class Label(db.Model):
    __tablename__ = 'labels'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    description = db.Column(db.String(255), nullable=True)

class TrainingRun(db.Model):
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
