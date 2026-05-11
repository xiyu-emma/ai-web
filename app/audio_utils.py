import os
import librosa
import librosa.display
import matplotlib
# 使用非互動式後端，這在伺服器環境下至關重要
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from scipy.io import wavfile
from datetime import datetime
from .ai_model import run_inference
import soundfile as sf
import torch
import torchaudio
import gc  # 垃圾回收模組
import concurrent.futures

from scipy.signal import butter, sosfiltfilt, decimate, get_window, lfilter, hilbert
from numpy.fft import fft, fftfreq

# --- 偵測 GPU 裝置 ---
_TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[audio_utils] PyTorch 頻譜圖運算裝置: {_TORCH_DEVICE}")

# --- YAMNet 參數設定 ---

class YAMNetParams:
    sample_rate: float = 16000.0
    stft_window_seconds: float = 0.025
    stft_hop_seconds: float = 0.010
    mel_bands: int = 64
    mel_min_hz: float = 125.0
    mel_max_hz: float = 7500.0
    log_offset: float = 0.001
    patch_window_seconds: float = 0.96
    patch_hop_seconds: float = 0.48
    tflite_compatible: bool = False

def waveform_to_log_mel_spectrogram_patches(waveform, params):
    """
    使用 PyTorch (torchaudio) 計算 Log Mel 頻譜圖，支援 GPU 加速。
    取代原本的 TensorFlow 實作，解決 ultralytics Docker 環境下 libdevice 缺失問題。
    """
    window_length_samples = int(round(params.sample_rate * params.stft_window_seconds))
    hop_length_samples = int(round(params.sample_rate * params.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    # 建立 MelSpectrogram 轉換器 (在目標裝置上運行)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(params.sample_rate),
        n_fft=fft_length,
        win_length=window_length_samples,
        hop_length=hop_length_samples,
        n_mels=params.mel_bands,
        f_min=params.mel_min_hz,
        f_max=params.mel_max_hz,
        power=1.0,  # 幅度頻譜 (非功率頻譜)
    ).to(_TORCH_DEVICE)

    # 將 waveform 轉為 PyTorch tensor 並移至 GPU
    if isinstance(waveform, np.ndarray):
        waveform_tensor = torch.from_numpy(waveform).float()
    else:
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, samples]

    waveform_tensor = waveform_tensor.to(_TORCH_DEVICE)

    # GPU 加速運算：STFT + Mel 濾波 + Log
    mel_spec = mel_transform(waveform_tensor)  # [1, n_mels, time]
    log_mel_spec = torch.log(mel_spec + params.log_offset)

    # 轉回 NumPy (移回 CPU)
    result = log_mel_spec.squeeze(0).T.cpu().numpy()  # [time, n_mels] 與原本 TF 版本相同

    # 釋放 GPU 記憶體
    del waveform_tensor, mel_spec, log_mel_spec
    if _TORCH_DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return result

def waveform_to_linear_mel_spectrogram_patches(waveform, params):
    """
    使用 PyTorch (torchaudio) 計算線性 Mel 頻譜圖，支援 GPU 加速。
    不套用 log 計算，保留原始能量值。
    """
    window_length_samples = int(round(params.sample_rate * params.stft_window_seconds))
    hop_length_samples = int(round(params.sample_rate * params.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(params.sample_rate),
        n_fft=fft_length,
        win_length=window_length_samples,
        hop_length=hop_length_samples,
        n_mels=params.mel_bands,
        f_min=params.mel_min_hz,
        f_max=params.mel_max_hz,
        power=1.0,
    ).to(_TORCH_DEVICE)

    if isinstance(waveform, np.ndarray):
        waveform_tensor = torch.from_numpy(waveform).float()
    else:
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)

    if waveform_tensor.dim() == 1:
        waveform_tensor = waveform_tensor.unsqueeze(0)

    waveform_tensor = waveform_tensor.to(_TORCH_DEVICE)

    mel_spec = mel_transform(waveform_tensor)
    
    result = mel_spec.squeeze(0).T.cpu().numpy()

    del waveform_tensor, mel_spec
    if _TORCH_DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return result


# --- DEMON 參數與輔助函式 ---

CLASSIC_DEMON_PARAMS = {
    'BANDPASS_LOW': 2000, 'BANDPASS_HIGH': 7500, 'DOWNSAMPLE_RATE': 2000,
    'WINDOW_SIZE': 2048, 'WINDOW_OVERLAP_RATIO': 0.95, 'WINDOW_TYPE': 'hann',
    'FREQ_YLIM': 200
}

def _bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

def _square_law_demodulate(signal):
    return signal ** 2

# --- 核心繪圖函式 (已加入記憶體保護) ---

def save_spectrogram(y, sr, out_path_display, out_path_training, spec_type='mel', spec_params=None):
    """
    儲存頻譜圖。
    
    參數:
        y: 音訊資料
        sr: 取樣率
        out_path_display: 顯示用頻譜圖路徑
        out_path_training: 訓練用頻譜圖路徑
        spec_type: 頻譜圖類型 ('mel', 'stft', 'classic_demon', 'envelope_spectrum', 'log_mel')
        spec_params: 頻譜圖參數字典，包含:
            - n_fft: FFT window size (預設 1024)
            - hop_length: 步幅 (預設 512)
            - window_type: 窗函數類型 (預設 'hann')
            - n_mels: Mel 濾波器數量 (預設 128)
            - f_min: 最低頻率 (預設 0)
            - f_max: 最高頻率 (預設 sr/2)
            - power: 功率指數 (預設 2.0)
    """
    # 預設參數
    if spec_params is None:
        spec_params = {}
    
    n_fft = spec_params.get('n_fft', 1024)
    hop_length = spec_params.get('hop_length', 512)
    window_type = spec_params.get('window_type', 'hann')
    n_mels = spec_params.get('n_mels', 128)
    f_min = spec_params.get('f_min', 0)
    f_max = spec_params.get('f_max', 0)
    power = spec_params.get('power', 2.0)
    
    # 如果 f_max 為 0，使用 Nyquist 頻率
    if f_max <= 0 or f_max > sr / 2:
        f_max = sr / 2
    
    # 分流處理特殊圖形
    if spec_type == 'classic_demon':
        save_classic_demon_plot(y, sr, out_path_display, out_path_training, spec_params)
        return
    elif spec_type == 'envelope_spectrum':
        save_envelope_spectrum_plot(y, sr, out_path_display, out_path_training, spec_params)
        return
    elif spec_type == 'log_mel':
        save_log_mel_plot(y, sr, out_path_display, out_path_training, spec_params)
        return
    elif spec_type == 'linear_mel':
        save_linear_mel_plot(y, sr, out_path_display, out_path_training, spec_params)
        return

    # 標準 STFT 處理
    fig = None
    try:
        fig = Figure(figsize=(9.69, 3.7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        time_str = ""
        if 'time_start' in spec_params and 'time_end' in spec_params:
            time_str = f" ({spec_params['time_start']:.1f}s - {spec_params['time_end']:.1f}s)"
        
        if spec_type == 'stft':
            # 將 numpy array 轉成 PyTorch tensor 並掛載到 GPU (如果可用)
            tensor_y = torch.from_numpy(y).float().to(_TORCH_DEVICE)
            
            # 設定 window function
            if window_type == 'hann':
                window_tensor = torch.hann_window(n_fft).to(_TORCH_DEVICE)
            elif window_type == 'hamming':
                window_tensor = torch.hamming_window(n_fft).to(_TORCH_DEVICE)
            else:
                window_tensor = torch.hann_window(n_fft).to(_TORCH_DEVICE) # 預設使用 hann
                
            # PyTorch STFT (支援 GPU 加速)
            D_complex = torch.stft(
                tensor_y, 
                n_fft=n_fft, 
                hop_length=hop_length,
                win_length=n_fft,
                window=window_tensor,
                center=True,
                pad_mode='reflect',
                return_complex=True
            )
            
            # 計算 Amplitude 或 Power
            D_mag = torch.abs(D_complex)
            if power == 2.0:
                D_power = D_mag ** 2
                max_val = torch.max(D_power)
                # 轉為 dB 尺度: 10 * log10(power / max_power)
                S_db_tensor = 10.0 * torch.log10(torch.clamp(D_power, min=1e-10) / torch.clamp(max_val, min=1e-10))
            else:
                max_val = torch.max(D_mag)
                # 轉為 dB 尺度: 20 * log10(mag / max_mag)
                S_db_tensor = 20.0 * torch.log10(torch.clamp(D_mag, min=1e-10) / torch.clamp(max_val, min=1e-10))
                
            # 轉回 NumPy 供後續顯示
            display_data = S_db_tensor.cpu().numpy()
            
            # 釋放 GPU 記憶體
            del tensor_y, window_tensor, D_complex, D_mag, S_db_tensor
            if _TORCH_DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            return

        specshow_kwargs = {
            'sr': sr,
            'x_axis': 'time',
            'ax': ax,
            'y_axis': 'hz',
            'cmap': 'viridis'
        }

        # 預防 Matplotlib 繪製極度密集的數據矩陣時引發 OOM (Signal 9 SIGKILL)
        # 對時間軸進行 Max Pooling 降採樣
        if display_data.shape[1] > 2000:
            factor = display_data.shape[1] // 1000
            trunc_len = (display_data.shape[1] // factor) * factor
            display_data = display_data[:, :trunc_len].reshape(display_data.shape[0], -1, factor).max(axis=2)
            hop_length = hop_length * factor

        specshow_kwargs['hop_length'] = hop_length

        # 1. 繪製顯示用圖 (包含座標軸與標題)
        librosa.display.specshow(display_data, **specshow_kwargs)
        if f_max > 0:
            ax.set_ylim([f_min, f_max])
        ax.set_title(f'STFT Spectrogram{time_str}')
        fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')
        fig.tight_layout()
        fig.savefig(out_path_display, dpi=100)
        
        # 2. 快速儲存訓練用圖 (直接存 Numpy Array，避開 Matplotlib 繪圖瓶頸)
        try:
            import cv2
            import matplotlib.cm as cm
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            freq_mask = (freqs >= f_min) & (freqs <= f_max)
            train_data = display_data[freq_mask, :]
            
            d_min = train_data.min()
            d_max = train_data.max()
            norm_data = (train_data - d_min) / (d_max - d_min + 1e-8)
            norm_data = np.flipud(norm_data)
            colored = cm.viridis(norm_data)
            img_bgr = (colored[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(out_path_training, img_bgr)
        except Exception as fast_save_err:
            print(f"快速儲存 STFT 失敗: {fast_save_err}")
    
    except Exception as e:
        print(f"繪圖失敗 ({spec_type}): {e}")
    finally:
        # 強制關閉圖表釋放記憶體
        if fig:
            fig.clf()

def save_log_mel_plot(y, sr, out_path_display, out_path_training, spec_params=None):
    """繪製 YAMNet 格式的 Log Mel 頻譜圖"""
    fig = None
    try:
        params = YAMNetParams()
        params.sample_rate = float(sr)  # 動態調整 sample rate，不再強制寫死 16000
        
        if spec_params:
            if 'window_overlap' in spec_params:
                overlap_ratio = spec_params.get('window_overlap', 0.6)  # Default YAMNet overlap is 60% (15ms / 25ms)
                params.stft_hop_seconds = params.stft_window_seconds * (1.0 - overlap_ratio)
            if 'f_min' in spec_params:
                params.mel_min_hz = float(spec_params['f_min'])
            if 'f_max' in spec_params:
                f_max_val = float(spec_params['f_max'])
                params.mel_max_hz = f_max_val if f_max_val > 0 else params.sample_rate / 2.0
                
        # 確保 mel_max_hz 不會超過 Nyquist
        params.mel_max_hz = min(params.mel_max_hz, params.sample_rate / 2.0)
        
        log_mel_spectrogram = waveform_to_log_mel_spectrogram_patches(y, params)
        data_to_plot = np.array(log_mel_spectrogram).T

        fig = Figure(figsize=(9.69, 3.7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hop_length = int(params.sample_rate * params.stft_hop_seconds)
        
        img = librosa.display.specshow(
            data_to_plot, 
            sr=params.sample_rate, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='mel', 
            fmin=params.mel_min_hz, 
            fmax=params.mel_max_hz,
            ax=ax,
            cmap='viridis'
        )
        
        time_str = ""
        if spec_params and 'time_start' in spec_params and 'time_end' in spec_params:
            time_str = f" ({spec_params['time_start']:.1f}s - {spec_params['time_end']:.1f}s)"
        ax.set_title(f"Log Mel Spectrogram{time_str}")
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        fig.tight_layout()
        fig.savefig(out_path_display, dpi=100)
        
        try:
            import cv2
            import matplotlib.cm as cm
            d_min = data_to_plot.min()
            d_max = data_to_plot.max()
            norm_data = (data_to_plot - d_min) / (d_max - d_min + 1e-8)
            norm_data = np.flipud(norm_data)
            colored = cm.viridis(norm_data)
            img_bgr = (colored[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(out_path_training, img_bgr)
        except Exception as fast_save_err:
            print(f"快速儲存 Log Mel 失敗: {fast_save_err}")
    except Exception as e:
        print(f"繪製 YAMNet Log Mel 頻譜圖時發生錯誤: {e}")
    finally:
        if fig: fig.clf()

def save_linear_mel_plot(y, sr, out_path_display, out_path_training, spec_params=None):
    """繪製線性 Mel 頻譜圖"""
    fig = None
    try:
        params = YAMNetParams()
        params.sample_rate = float(sr)
        
        if spec_params:
            if 'window_overlap' in spec_params:
                overlap_ratio = spec_params.get('window_overlap', 0.6)
                params.stft_hop_seconds = params.stft_window_seconds * (1.0 - overlap_ratio)
            if 'f_min' in spec_params:
                params.mel_min_hz = float(spec_params['f_min'])
            if 'f_max' in spec_params:
                f_max_val = float(spec_params['f_max'])
                params.mel_max_hz = f_max_val if f_max_val > 0 else params.sample_rate / 2.0
                
        # 確保 mel_max_hz 不會超過 Nyquist
        params.mel_max_hz = min(params.mel_max_hz, params.sample_rate / 2.0)
        
        linear_mel_spectrogram = waveform_to_linear_mel_spectrogram_patches(y, params)
        data_to_plot = np.array(linear_mel_spectrogram).T

        fig = Figure(figsize=(9.69, 3.7))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        hop_length = int(params.sample_rate * params.stft_hop_seconds)
        
        img = librosa.display.specshow(
            data_to_plot, 
            sr=params.sample_rate, 
            hop_length=hop_length, 
            x_axis='time', 
            y_axis='mel', 
            fmin=params.mel_min_hz, 
            fmax=params.mel_max_hz,
            ax=ax,
            cmap='viridis'
        )
        
        time_str = ""
        if spec_params and 'time_start' in spec_params and 'time_end' in spec_params:
            time_str = f" ({spec_params['time_start']:.1f}s - {spec_params['time_end']:.1f}s)"
        ax.set_title(f"Linear Mel Spectrogram{time_str}")
        fig.colorbar(img, ax=ax)
        fig.tight_layout()
        fig.savefig(out_path_display, dpi=100)
        
        try:
            import cv2
            import matplotlib.cm as cm
            d_min = data_to_plot.min()
            d_max = data_to_plot.max()
            norm_data = (data_to_plot - d_min) / (d_max - d_min + 1e-8)
            norm_data = np.flipud(norm_data)
            colored = cm.viridis(norm_data)
            img_bgr = (colored[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(out_path_training, img_bgr)
        except Exception as fast_save_err:
            print(f"快速儲存 Linear Mel 失敗: {fast_save_err}")
    except Exception as e:
        print(f"繪製 Linear Mel 頻譜圖時發生錯誤: {e}")
    finally:
        if fig: fig.clf()

def save_classic_demon_plot(segment, sr, out_path_display, out_path_training, spec_params=None):
    fig = None
    try:
        params = CLASSIC_DEMON_PARAMS
        nyquist = sr / 2
        bandpass_high = min(params['BANDPASS_HIGH'], nyquist * 0.99)
        if params['BANDPASS_LOW'] >= bandpass_high: return
        filtered = _bandpass_filter(segment, sr, params['BANDPASS_LOW'], bandpass_high)
        demodulated = _square_law_demodulate(filtered)
        decimation_factor = max(1, int(sr / params['DOWNSAMPLE_RATE']))
        decimated_signal = decimate(demodulated, decimation_factor)
        processed_signal = decimated_signal - np.mean(decimated_signal)
        fs_demo = sr // decimation_factor
        window = get_window(params['WINDOW_TYPE'], params['WINDOW_SIZE'])
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        S, freqs, times, _ = ax.specgram(processed_signal, NFFT=params['WINDOW_SIZE'], Fs=fs_demo, window=window, noverlap=int(params['WINDOW_SIZE'] * params['WINDOW_OVERLAP_RATIO']))
        S_db = 10 * np.log10(S + 1e-9)

        ax.pcolormesh(times, freqs, S_db, cmap='viridis', shading='auto')
        ax.set_ylim(0, params['FREQ_YLIM'])
        ax.set_ylabel('Modulation Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        
        time_str = ""
        if spec_params and 'time_start' in spec_params and 'time_end' in spec_params:
            time_str = f" ({spec_params['time_start']:.1f}s - {spec_params['time_end']:.1f}s)"
        ax.set_title(f"Classic DEMON Spectrogram (2D){time_str}")
        fig.colorbar(ax.collections[0], ax=ax, label='Amplitude (dB)')
        fig.tight_layout()
        fig.savefig(out_path_display, dpi=100)
        
        try:
            import cv2
            import matplotlib.cm as cm
            freq_mask = freqs <= params['FREQ_YLIM']
            train_data = S_db[freq_mask, :]
            d_min, d_max = train_data.min(), train_data.max()
            norm_data = (train_data - d_min) / (d_max - d_min + 1e-8)
            norm_data = np.flipud(norm_data)
            colored = cm.viridis(norm_data)
            img_bgr = (colored[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(out_path_training, img_bgr)
        except Exception as fast_save_err:
            print(f"快速儲存 DEMON 失敗: {fast_save_err}")
    finally:
        if fig: fig.clf()

def save_envelope_spectrum_plot(segment, sr, out_path_display, out_path_training, spec_params=None):
    fig = None
    try:
        nyquist = sr / 2
        bp_low, bp_high = 2000, min(20000, nyquist * 0.99)
        if bp_low >= bp_high: return
        
        b, a = butter(4, [bp_low, bp_high], btype='band', fs=sr)
        segment_filt = lfilter(b, a, segment)
        envelope = np.abs(hilbert(segment_filt))
        
        N = len(envelope)
        if N == 0: return
        
        yf = fft(envelope - np.mean(envelope))
        xf = fftfreq(N, 1 / sr)
        half_N = N // 2
        xf_positive, yf_positive = xf[:half_N], 2.0/N * np.abs(yf[:half_N])
        
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(xf_positive, yf_positive)
        
        time_str = ""
        if spec_params and 'time_start' in spec_params and 'time_end' in spec_params:
            time_str = f" ({spec_params['time_start']:.1f}s - {spec_params['time_end']:.1f}s)"
        ax.set_title(f'Envelope Spectrum (DEMON 1D){time_str}')
        ax.set_xlabel('Modulation Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True)
        ax.set_xlim(0, 300)
        fig.tight_layout()
        fig.savefig(out_path_display, dpi=100)

        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(xf_positive, yf_positive)
        ax.set_xlim(0, 300)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(out_path_training, bbox_inches='tight', pad_inches=0, dpi=100)
    except Exception as e:
        print(f"處理包絡線頻譜時發生錯誤: {e}")
    finally:
        if fig: fig.clf()

# --- 記憶體優化處理流程 ---

def _process_single_segment(i, start_s, y_segment, sr, basename, result_dir, spec_type, spec_params, training_spec_path, is_mono):
    import os
    import numpy as np
    from scipy.io import wavfile
    from .ai_model import run_inference
    
    audio_filename = f"{basename}_part{i}.wav"
    display_spec_filename = f"{basename}_spec_display_{i}.png"
    training_spec_filename = f"{basename}_spec_training_{i}.png"
    
    audio_path = os.path.join(result_dir, audio_filename)
    display_spec_path = os.path.join(result_dir, display_spec_filename)

    # 儲存切割音檔 (確保正確處理多聲道)
    if y_segment.ndim > 1:
        y_mono = librosa.to_mono(y_segment)
        if np.max(np.abs(y_mono)) < 1e-4 and np.max(np.abs(y_segment)) > 1e-3:
            y_segment = y_segment[0]
        else:
            y_segment = y_mono
            
    audio_int16 = (y_segment * 32767).astype(np.int16)
    wavfile.write(audio_path, sr, audio_int16)
    
    mono_segment = y_segment
    current_spec_params = {} if spec_params is None else spec_params.copy()
    current_spec_params['time_start'] = start_s
    current_spec_params['time_end'] = start_s + (len(y_segment) / sr)
    
    save_spectrogram(mono_segment, sr, display_spec_path, training_spec_path, spec_type, current_spec_params)
    
    return {
        'audio': audio_filename,
        'display_spectrogram': display_spec_filename,
        'training_spectrogram': training_spec_filename,
        'detections': run_inference(training_spec_path)
    }

def process_large_audio(filepath, result_dir, spec_type, segment_duration=2.0, overlap_ratio=0.5, target_sr=None, is_mono=True, progress_callback=None, spec_params=None):
    """
    以一次性完整載入並切分的方式處理大型音訊檔案，搭配 ThreadPool 平行處理加速。
    """
    all_results = []
    basename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # 新增容錯讀取機制
        try:
            info = sf.info(filepath)
            original_sr = info.samplerate
            total_samples = info.frames
        except Exception as e:
            print(f"SoundFile 無法讀取 {filepath}，嘗試使用 Librosa Fallback。錯誤: {e}")
            original_sr = librosa.get_samplerate(filepath)
            total_duration = librosa.get_duration(path=filepath)
            total_samples = int(total_duration * original_sr)

        sr = target_sr if target_sr else original_sr

        # 一次性完整載入音訊，大幅減少 I/O 等待 (O(N^2) seek issues in MP3)
        print(f"正在完整載入音訊: {filepath} ({total_samples} samples)")
        full_audio, _ = librosa.load(filepath, sr=sr, mono=is_mono)
        print("音訊載入完成。")

        frame_length = int(segment_duration * sr)
        actual_total_samples = full_audio.shape[-1]

        if actual_total_samples < frame_length:
            print("警告：音訊檔案總長度小於設定的單一片段長度。")
            y_segment = full_audio
            pad_width = [(0, 0)] * y_segment.ndim
            pad_width[-1] = (0, frame_length - actual_total_samples)
            y_segment = np.pad(y_segment, pad_width)
            segments_to_process = [(0, 0.0, y_segment)]
        else:
            step_samples = int(frame_length * (1 - overlap_ratio))
            segments_to_process = []
            start_sample = 0
            idx = 0
            while start_sample <= actual_total_samples - frame_length:
                start_s = start_sample / sr
                y_segment = full_audio[..., start_sample:start_sample + frame_length]
                segments_to_process.append((idx, start_s, y_segment))
                start_sample += step_samples
                idx += 1

        total_segments = len(segments_to_process)
        completed_tasks = 0
        
        # 準備結果的 list (保持順序)
        all_results = [None] * total_segments
        
        print(f"開始平行處理 {total_segments} 個音訊片段...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
            futures = {}
            for idx, start_s, y_seg in segments_to_process:
                training_spec_path = os.path.join(result_dir, f"{basename}_spec_training_{idx}.png")
                fut = executor.submit(
                    _process_single_segment, 
                    idx, start_s, y_seg, sr, basename, result_dir, spec_type, spec_params, training_spec_path, is_mono
                )
                futures[fut] = idx
                
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                    all_results[idx] = res
                    completed_tasks += 1
                    if progress_callback:
                        progress_callback(completed_tasks, total_segments)
                except Exception as exc:
                    print(f"片段 {idx} 處理發生錯誤: {exc}")
                    
        # 過濾異常片段
        all_results = [r for r in all_results if r is not None]
        del full_audio
        gc.collect()
        
    except Exception as e:
        print(f"處理大型音訊檔案時發生錯誤: {e}")
        raise e
        
    return all_results

def _process_segment_group(i, start_s, y_segment, sr, basename, group_defs, is_mono):
    import os
    import numpy as np
    from scipy.io import wavfile
    from .ai_model import run_inference
    
    audio_filename = f"{basename}_part{i}.wav"
    display_spec_filename = f"{basename}_spec_display_{i}.png"
    training_spec_filename = f"{basename}_spec_training_{i}.png"
    
    # 儲存切割音檔 (確保正確處理多聲道)
    if y_segment.ndim > 1:
        y_mono = librosa.to_mono(y_segment)
        if np.max(np.abs(y_mono)) < 1e-4 and np.max(np.abs(y_segment)) > 1e-3:
            y_segment = y_segment[0]
        else:
            y_segment = y_mono
            
    audio_int16 = (y_segment * 32767).astype(np.int16)
    
    # 寫入 wav 檔案到所有的 result_dir
    for g in group_defs:
        audio_path = os.path.join(g['result_dir'], audio_filename)
        wavfile.write(audio_path, sr, audio_int16)
        
    results = {}
    mono_segment = y_segment
    
    for g in group_defs:
        r_dir = g['result_dir']
        spec_type = g['spec_type']
        spec_params = g['spec_params']
        
        training_spec_path = os.path.join(r_dir, training_spec_filename)
        display_spec_path = os.path.join(r_dir, display_spec_filename)
        
        current_spec_params = spec_params.copy() if spec_params else {}
        current_spec_params['time_start'] = start_s
        current_spec_params['time_end'] = start_s + (len(mono_segment) / sr)
        
        save_spectrogram(mono_segment, sr, display_spec_path, training_spec_path, spec_type, current_spec_params)
        
        results[g['id']] = {
            'audio': audio_filename,
            'display_spectrogram': display_spec_filename,
            'training_spectrogram': training_spec_filename,
            'detections': run_inference(training_spec_path)
        }
        
    return results

def process_large_audio_group(filepath, group_defs, segment_duration=2.0, overlap_ratio=0.5, target_sr=None, is_mono=True, progress_callback=None):
    """
    以一次性完整載入並切分的方式處理大型音訊檔案，並在每個切片同時產出多種頻譜圖。
    """
    import soundfile as sf
    all_results = {g['id']: [] for g in group_defs}
    basename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        try:
            info = sf.info(filepath)
            original_sr = info.samplerate
            total_samples = info.frames
        except Exception as e:
            print(f"SoundFile 無法讀取 {filepath}，嘗試使用 Librosa Fallback。錯誤: {e}")
            original_sr = librosa.get_samplerate(filepath)
            total_duration = librosa.get_duration(path=filepath)
            total_samples = int(total_duration * original_sr)

        sr = target_sr if target_sr else original_sr

        print(f"[群組處理] 正在完整載入音訊: {filepath} ({total_samples} samples)")
        full_audio, _ = librosa.load(filepath, sr=sr, mono=is_mono)
        print("[群組處理] 音訊載入完成。")

        frame_length = int(segment_duration * sr)
        actual_total_samples = full_audio.shape[-1]

        if actual_total_samples < frame_length:
            print("警告：音訊檔案總長度小於設定的單一片段長度。")
            y_segment = full_audio
            pad_width = [(0, 0)] * y_segment.ndim
            pad_width[-1] = (0, frame_length - actual_total_samples)
            y_segment = np.pad(y_segment, pad_width)
            segments_to_process = [(0, 0.0, y_segment)]
        else:
            step_samples = int(frame_length * (1 - overlap_ratio))
            segments_to_process = []
            start_sample = 0
            idx = 0
            while start_sample <= actual_total_samples - frame_length:
                start_s = start_sample / sr
                y_segment = full_audio[..., start_sample:start_sample + frame_length]
                segments_to_process.append((idx, start_s, y_segment))
                start_sample += step_samples
                idx += 1

        total_segments = len(segments_to_process)
        completed_tasks = 0
        
        print(f"[群組處理] 開始平行處理 {total_segments} 個音訊片段...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
            futures = {}
            for idx, start_s, y_seg in segments_to_process:
                fut = executor.submit(
                    _process_segment_group, 
                    idx, start_s, y_seg, sr, basename, group_defs, is_mono
                )
                futures[fut] = idx
                
            # 依序回收結果，使用 list 來固定順序
            # 建立一個佔位用的字典
            ordered_results = {g['id']: [None] * total_segments for g in group_defs}
                
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    res_dict = fut.result()
                    for gid, res in res_dict.items():
                        ordered_results[gid][idx] = res
                    completed_tasks += 1
                    if progress_callback:
                        progress_callback(completed_tasks, total_segments)
                except Exception as exc:
                    print(f"片段 {idx} 處理發生錯誤: {exc}")
                    
        # 移除 None
        for gid in ordered_results:
            all_results[gid] = [r for r in ordered_results[gid] if r is not None]
            
        del full_audio
        gc.collect()
        
    except Exception as e:
        print(f"處理大型音訊檔案群組時發生錯誤: {e}")
        raise e
        
    return all_results