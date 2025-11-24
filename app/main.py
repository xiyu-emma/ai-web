import os
import json
import shutil
import csv
from datetime import datetime
from flask import Blueprint, current_app, render_template, request, redirect, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
from io import StringIO, BytesIO

from . import db, celery
from .models import Upload, Result, Label, TrainingRun

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/history')
def history():
    sort_order = request.args.get('sort', 'desc')
    query = Upload.query
    if sort_order == 'asc':
        all_uploads = query.order_by(Upload.upload_timestamp.asc()).all()
    else:
        all_uploads = query.order_by(Upload.upload_timestamp.desc()).all()
    return render_template('history.html', uploads=all_uploads, current_sort=sort_order)

@main_bp.route('/results/<int:upload_id>')
def results(upload_id):
    page = request.args.get('page', 1, type=int)
    upload_record = Upload.query.get_or_404(upload_id)
    params = upload_record.get_params()
    try:
        segment_duration = float(params.get('segment_duration', 2.0))
        overlap_percent = float(params.get('overlap', 50))
        overlap_ratio = overlap_percent / 100.0
        hop_length_seconds = segment_duration * (1 - overlap_ratio)
    except (ValueError, TypeError):
        hop_length_seconds = 1.0
    pagination = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('result.html', upload=upload_record, pagination=pagination, hop_length_seconds=hop_length_seconds)

@main_bp.route('/labeling/<int:upload_id>')
def labeling_page(upload_id):
    page = request.args.get('page', 1, type=int)
    upload_record = Upload.query.get_or_404(upload_id)
    pagination = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).paginate(
        page=page, per_page=50, error_out=False
    )
    return render_template('label.html', upload=upload_record, pagination=pagination)

@main_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    try:
        params_dict = {
            'spec_type': request.form['spec_type'],
            'segment_duration': float(request.form['segment_duration']),
            'overlap': float(request.form['overlap']),
            'sample_rate': request.form.get('sample_rate', 'None'),
            'channels': request.form.get('channels', 'mono')
        }
    except (KeyError, ValueError):
        return "Invalid parameters provided.", 400
    
    if file:
        filename = secure_filename(file.filename)
        params_json = json.dumps(params_dict)
        custom_folder = request.form.get('custom_folder', '').strip()
        
        new_upload = Upload(
            original_filename=filename,
            result_path="pending",
            custom_folder_path=custom_folder if custom_folder else None,
            params=params_json,
            status='PENDING'
        )
        db.session.add(new_upload)
        db.session.commit()
        upload_id = new_upload.id
        
        if custom_folder:
            result_dir_name = os.path.join(custom_folder, str(upload_id))
        else:
            result_dir_name = os.path.join(current_app.root_path, 'static', 'results', str(upload_id))
        
        os.makedirs(result_dir_name, exist_ok=True)
        
        if custom_folder:
            new_upload.result_path = os.path.join(custom_folder, str(upload_id))
        else:
            new_upload.result_path = os.path.join('results', str(upload_id))
        
        db.session.commit()
        
        upload_path = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(upload_path)
        
        celery.send_task('app.tasks.process_audio_task', args=[upload_id])
        return redirect(url_for('main.history', new_upload_id=upload_id))
    
    return redirect(url_for('main.index'))

@main_bp.route('/export_csv/<int:upload_id>')
def export_csv(upload_id):
    upload_record = Upload.query.get_or_404(upload_id)
    results = Result.query.filter_by(upload_id=upload_id).order_by(Result.id.asc()).all()
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Audio Filename', 'Spectrogram Filename', 'Label'])
    
    for result in results:
        label_name = result.label.name if result.label else 'No Label'
        writer.writerow([
            result.id,
            result.audio_filename or 'N/A',
            result.spectrogram_filename,
            label_name
        ])
    
    output.seek(0)
    byte_output = BytesIO()
    byte_output.write(output.getvalue().encode('utf-8-sig'))
    byte_output.seek(0)
    
    filename = f"labels_{upload_record.original_filename}_{upload_id}.csv"
    return send_file(
        byte_output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@main_bp.route('/history/delete_selected', methods=['POST'])
def delete_selected_uploads():
    upload_ids = request.form.getlist('upload_ids')
    if not upload_ids:
        return redirect(url_for('main.history'))
    
    uploads_to_delete = Upload.query.filter(Upload.id.in_(upload_ids)).all()
    for upload in uploads_to_delete:
        if upload.custom_folder_path:
            physical_path = os.path.join(upload.custom_folder_path, str(upload.id))
        else:
            physical_path = os.path.join(current_app.root_path, 'static', upload.result_path)
        
        try:
            if os.path.isdir(physical_path):
                shutil.rmtree(physical_path)
        except OSError as e:
            print(f"Error deleting folder {physical_path}: {e}")
        
        temp_upload_file = os.path.join(current_app.root_path, current_app.config['UPLOAD_FOLDER'], f"{upload.id}_{upload.original_filename}")
        if os.path.exists(temp_upload_file):
            try:
                os.remove(temp_upload_file)
            except OSError as e:
                print(f"Error deleting temp file {temp_upload_file}: {e}")
        
        db.session.delete(upload)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting database records: {e}")
    
    return redirect(url_for('main.history'))

@main_bp.route('/training/status')
def training_status():
    runs = TrainingRun.query.order_by(TrainingRun.timestamp.desc()).all()
    return render_template('training_status.html', runs=runs)

@main_bp.route('/training/report/<int:run_id>')
def training_report(run_id):
    run = TrainingRun.query.get_or_404(run_id)
    if run.status != 'SUCCESS' or not run.results_path:
        return "Training not completed or no results available.", 404
    
    results_base_path = run.results_path
    report_images = {
        'results': os.path.join(results_base_path, 'results.png'),
        'confusion_matrix': os.path.join(results_base_path, 'confusion_matrix.png'),
        'val_batch0_labels': os.path.join(results_base_path, 'val_batch0_labels.jpg'),
        'val_batch0_pred': os.path.join(results_base_path, 'val_batch0_pred.jpg'),
    }
    metrics = run.get_metrics()
    return render_template('training_report.html', run=run, images=report_images, metrics=metrics)

@main_bp.route('/training/start', methods=['POST'])
def start_training():
    upload_ids = request.form.getlist('upload_ids')
    if not upload_ids:
        return redirect(url_for('main.history'))
    
    model_name = request.form.get('model_name', 'yolov8n-cls.pt')
    params = {'model_name': model_name, 'upload_ids': upload_ids}
    new_run = TrainingRun(status='PENDING', params=json.dumps(params))
    db.session.add(new_run)
    db.session.commit()
    
    celery.send_task('app.tasks.train_yolo_model', args=[upload_ids, new_run.id, model_name])
    return redirect(url_for('main.training_status', new_run_id=new_run.id))

@main_bp.route('/labeling/auto_label', methods=['POST'])
def auto_label():
    if 'model_file' not in request.files:
        return "No model file found.", 400
    
    file = request.files['model_file']
    upload_id = request.form.get('upload_id')
    
    if file.filename == '' or not upload_id:
        return "Missing model file or upload ID.", 400
    
    if file and (file.filename.endswith('.pt') or file.filename.endswith('.h5')):
        filename = secure_filename(file.filename)
        temp_model_dir = os.path.join(current_app.root_path, 'static', 'temp_models')
        os.makedirs(temp_model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_model_path = os.path.join(temp_model_dir, f"{timestamp}_{upload_id}_{filename}")
        file.save(saved_model_path)
        celery.send_task('app.tasks.auto_label_task', args=[int(upload_id), saved_model_path])
        return redirect(url_for('main.labeling_page', upload_id=upload_id))
    
    return "Invalid file format. Please upload .pt or .h5 file.", 400

@main_bp.route('/training/delete_selected', methods=['POST'])
def delete_selected_runs():
    run_ids = request.form.getlist('run_ids')
    if not run_ids:
        return redirect(url_for('main.training_status'))
    
    runs_to_delete = TrainingRun.query.filter(TrainingRun.id.in_(run_ids)).all()
    for run in runs_to_delete:
        if run.results_path:
            base_path = os.path.dirname(os.path.dirname(run.results_path))
            physical_path = os.path.join(current_app.root_path, 'static', base_path, str(run.id))
            try:
                if os.path.isdir(physical_path):
                    shutil.rmtree(physical_path)
            except OSError as e:
                print(f"Error deleting training folder {physical_path}: {e}")
        db.session.delete(run)
    db.session.commit()
    return redirect(url_for('main.training_status'))

@main_bp.route('/api/upload/<int:upload_id>/status')
def get_upload_status(upload_id):
    upload = Upload.query.get_or_404(upload_id)
    return jsonify({'id': upload.id, 'status': upload.status, 'progress': upload.progress})

@main_bp.route('/api/training/<int:run_id>/status')
def get_training_run_status(run_id):
    run = TrainingRun.query.get_or_404(run_id)
    return jsonify({'id': run.id, 'status': run.status, 'progress': run.progress})

@main_bp.route('/api/labels', methods=['GET', 'POST'])
def handle_labels():
    if request.method == 'GET':
        labels = Label.query.order_by(Label.name).all()
        return jsonify([{'id': label.id, 'name': label.name} for label in labels])
    if request.method == 'POST':
        data = request.get_json()
        if not data or not data.get('name'):
            return jsonify({'error': 'Label name is required'}), 400
        if Label.query.filter_by(name=data['name']).first():
            return jsonify({'error': 'Label name already exists'}), 409
        new_label = Label(name=data['name'])
        db.session.add(new_label)
        db.session.commit()
        return jsonify({'id': new_label.id, 'name': new_label.name}), 201

@main_bp.route('/api/labels/<int:label_id>', methods=['DELETE'])
def delete_label(label_id):
    label = Label.query.get_or_404(label_id)
    db.session.delete(label)
    db.session.commit()
    return jsonify({'success': True})

@main_bp.route('/api/results/<int:result_id>/label', methods=['POST'])
def update_result_label(result_id):
    result = Result.query.get_or_404(result_id)
    data = request.get_json()
    label_id = data.get('label_id')
    if label_id is not None and not isinstance(label_id, int):
        return jsonify({'error': 'Invalid label_id format'}), 400
    result.label_id = label_id
    db.session.commit()
    return jsonify({'success': True})
