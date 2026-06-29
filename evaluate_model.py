# -*- coding: utf-8 -*-

import os
import csv
import argparse
import sys
from collections import defaultdict

# 嘗試載入 PyTorch 相關套件
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 嘗試載入 YOLO 相關套件
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# 嘗試載入 ONNX 相關套件
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# 檢查必備的基礎分析套件
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    HAS_ANALYSIS_LIBS = True
except ImportError:
    print("提示：缺少 pandas, numpy, scikit-learn 或 matplotlib 等分析庫。")
    print("系統將以基本模式運行（不生成混淆矩陣圖表與詳細分類報告）。")
    print("建議安裝完整套件以獲得最佳體驗：")
    print("    pip install pandas numpy scikit-learn matplotlib seaborn Pillow")
    HAS_ANALYSIS_LIBS = False


# =====================================================================
# U-Net 架構定義 (若訓練模型為 UNetClassifier 時載入用)
# =====================================================================
if HAS_TORCH:
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        def forward(self, x):
            return self.double_conv(x)

    class Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        def forward(self, x):
            return self.maxpool_conv(x)

    class Up(nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)
        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class UNetClassifier(nn.Module):
        def __init__(self, n_channels=3, n_classes=1000, bilinear=False):
            super(UNetClassifier, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            
            self.up1 = Up(1024, 512 // factor, bilinear)
            self.up2 = Up(512, 256 // factor, bilinear)
            self.up3 = Up(256, 128 // factor, bilinear)
            self.up4 = Up(128, 64, bilinear)
            
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes)
            )

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            return logits

    class AttentionBlock(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super(AttentionBlock, self).__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)
            
        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class AttentionUp(nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)
            self.att = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
            
        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x2 = self.att(g=x1, x=x2)
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

    class AttentionUNetClassifier(nn.Module):
        def __init__(self, n_channels=3, n_classes=1000, bilinear=False):
            super(AttentionUNetClassifier, self).__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.bilinear = bilinear

            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)
            
            self.up1 = AttentionUp(1024, 512 // factor, bilinear)
            self.up2 = AttentionUp(512, 256 // factor, bilinear)
            self.up3 = AttentionUp(256, 128 // factor, bilinear)
            self.up4 = AttentionUp(128, 64, bilinear)
            
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, n_classes)
            )

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            logits = self.classifier(x)
            return logits


def find_image_path(csv_filename, base_dir):
    """
    尋找圖片在本地的實際路徑，處理可能存在的路徑差異。
    """
    path1 = os.path.join(base_dir, csv_filename)
    if os.path.exists(path1):
        return path1
    
    filename_only = os.path.basename(csv_filename)
    path2 = os.path.join(base_dir, filename_only)
    if os.path.exists(path2):
        return path2
        
    path3 = os.path.join(base_dir, "images", filename_only)
    if os.path.exists(path3):
        return path3

    for root, _, files in os.walk(base_dir):
        if filename_only in files:
            return os.path.join(root, filename_only)

    return None


def run_evaluation(model_path, csv_path, images_dir, output_dir, skip_unlabeled=True):
    print("=" * 60)
    print(" 鯨豚聲音頻譜圖分類模型評估與測試系統")
    print("=" * 60)
    print(f"載入權重檔: {model_path}")
    print(f"載入 CSV 標記檔: {csv_path}")
    print(f"圖片搜尋目錄: {images_dir}")
    print("-" * 60)

    # 1. 偵測與載入模型
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型權重檔 {model_path}。")
        return

    is_cnn = False
    cnn_arch = None
    cnn_classes = None
    cnn_model = None
    device = "cpu"
    cnn_transforms = None
    model_classes = {}
    yolo_model = None
    
    is_onnx = False
    onnx_session = None
    onnx_input_name = None

    # A0. 嘗試載入為 ONNX 模型
    if model_path.endswith('.onnx'):
        if HAS_ONNX:
            try:
                print("嘗試載入為 ONNX 模型...")
                onnx_session = ort.InferenceSession(model_path)
                onnx_input_name = onnx_session.get_inputs()[0].name
                is_onnx = True
                
                # 嘗試從目錄讀取 classes.txt 來取得類別名稱
                classes_txt = os.path.join(os.path.dirname(model_path), 'classes.txt')
                if os.path.exists(classes_txt):
                    with open(classes_txt, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        model_classes = {i: name for i, name in enumerate(lines)}
                        cnn_classes = lines
                    print(f" 從 {classes_txt} 載入類別名稱成功！")
                else:
                    print(" [警告] 找不到 classes.txt，將使用預設數字作為類別。如果預測標籤是文字，可能無法匹配。")
                print(" ONNX 模型載入成功！")
                
                if HAS_TORCH:
                    cnn_transforms = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                else:
                    print(" [警告] 需要 PyTorch 與 torchvision 來進行影像前處理。")
                    is_onnx = False
            except Exception as e:
                print(f"ONNX 模型載入失敗: {e}")
                is_onnx = False
        else:
            print("警告：未安裝 onnxruntime，無法載入 .onnx 模型，嘗試使用 YOLO 推論...")

    # A. 優先嘗試當成 PyTorch CNN 模型載入
    if not is_onnx and HAS_TORCH:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                is_cnn = True
                cnn_arch = checkpoint.get('arch', 'resnet18')
                cnn_classes = checkpoint.get('classes', [])
                state_dict = checkpoint['model_state_dict']
                
                num_classes = len(cnn_classes)
                print(f"偵測到系統訓練的 PyTorch CNN 模型！")
                print(f"  - 模型架構: {cnn_arch}")
                print(f"  - 輸出類別數: {num_classes}")
                print(f"  - 類別名稱列表: {cnn_classes}")
                
                # 初始化模型架構
                if cnn_arch == 'resnet18':
                    cnn_model = models.resnet18(weights=None)
                    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)
                elif cnn_arch == 'efficientnet_b0':
                    cnn_model = models.efficientnet_b0(weights=None)
                    cnn_model.classifier[1] = nn.Linear(cnn_model.classifier[1].in_features, num_classes)
                elif cnn_arch == 'unet':
                    cnn_model = UNetClassifier(n_channels=3, n_classes=num_classes)
                elif cnn_arch == 'attention_unet':
                    cnn_model = AttentionUNetClassifier(n_channels=3, n_classes=num_classes)
                else:
                    print(f"  [提示] 未知架構 {cnn_arch}，預設使用 resnet18 初始化。")
                    cnn_model = models.resnet18(weights=None)
                    cnn_model.fc = nn.Linear(cnn_model.fc.in_features, num_classes)
                
                # 清除 module. 前綴 (多卡訓練可能產生)
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k[7:] if k.startswith('module.') else k] = v
                
                cnn_model.load_state_dict(new_state_dict)
                cnn_model.to(device)
                cnn_model.eval()
                
                # 初始化影像前處理
                cnn_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                model_classes = {i: name for i, name in enumerate(cnn_classes)}
                print(" PyTorch CNN 模型載入成功！")
        except Exception as e:
            # 如果載入失敗，表示可能不是 PyTorch CNN 自訂模型，我們後續會嘗試用 YOLOv8 載入
            is_cnn = False

    # B. 若不是自訂 CNN 模型且不是 ONNX，嘗試當作 YOLO 模型載入
    if not is_cnn and not is_onnx:
        if not HAS_YOLO:
            print("錯誤：無法將模型載入為 PyTorch CNN，且本地未安裝 'ultralytics' 套件，無法嘗試載入為 YOLO。")
            print("請先執行: pip install ultralytics")
            return
        
        try:
            print("嘗試載入為 YOLOv8 模型...")
            yolo_model = YOLO(model_path)
            model_classes = yolo_model.names
            print(" YOLOv8 模型載入成功！")
            print(f"  - 訓練類別 ({len(model_classes)} 個): {model_classes}")
        except Exception as yolo_err:
            print(f"錯誤：無法載入模型檔 {model_path}。")
            print(f"  - 嘗試 PyTorch CNN 載入錯誤原因: {locals().get('e', '無')}")
            print(f"  - 嘗試 YOLOv8 載入錯誤原因: {yolo_err}")
            return

    # 2. 讀取 labels.csv 標記資料
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到 CSV 檔案 {csv_path}。")
        return

    samples = []
    unlabeled_count = 0
    missing_images_count = 0

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        required_fields = ['filename', 'label_name']
        for field in required_fields:
            if field not in reader.fieldnames:
                print(f"錯誤：CSV 檔案缺少必要欄位 '{field}'。")
                print(f"現有欄位: {reader.fieldnames}")
                return

        for row in reader:
            filename = row['filename']
            true_label = row['label_name'].strip()
            
            is_unlabeled = (
                true_label.lower() in ['unlabeled', 'unknown', 'none', ''] or 
                row.get('event_type') in ['0', 0]
            )
            
            if is_unlabeled and skip_unlabeled:
                unlabeled_count += 1
                continue

            img_real_path = find_image_path(filename, images_dir)
            if not img_real_path:
                missing_images_count += 1
                continue

            samples.append({
                'csv_filename': filename,
                'real_path': img_real_path,
                'true_label': true_label,
                'event_type': row.get('event_type', '')
            })

    print(f"CSV 讀取完成:")
    print(f"  - 找到有效標記樣本數: {len(samples)} 筆")
    if skip_unlabeled and unlabeled_count > 0:
        print(f"  - 已自動跳過未標記 (Unlabeled) 樣本: {unlabeled_count} 筆")
    if missing_images_count > 0:
        print(f"  - 警告：有 {missing_images_count} 筆圖片在 {images_dir} 中找不到，已跳過。")

    if not samples:
        print("錯誤：沒有可供評估的有效樣本。請檢查圖片路徑或 CSV 標記內容。")
        return

    # 3. 開始進行推論與預測
    print("\n 開始進行批次推論...")
    results_records = []
    correct_predictions = 0

    for i, sample in enumerate(samples, 1):
        img_path = sample['real_path']
        true_label = sample['true_label']
        
        try:
            if is_onnx:
                # ONNX 推論
                img = Image.open(img_path).convert('RGB')
                img_t = cnn_transforms(img).unsqueeze(0).numpy()
                
                outputs = onnx_session.run(None, {onnx_input_name: img_t})[0][0]
                
                # 計算 Softmax
                exp_outputs = np.exp(outputs - np.max(outputs))
                probs = exp_outputs / exp_outputs.sum()
                pred_idx = int(np.argmax(probs))
                
                if cnn_classes and pred_idx < len(cnn_classes):
                    pred_label = cnn_classes[pred_idx]
                else:
                    pred_label = str(pred_idx)
                confidence = float(probs[pred_idx])
            elif is_cnn:
                # CNN 推論
                img = Image.open(img_path).convert('RGB')
                img_t = cnn_transforms(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = cnn_model(img_t)
                    probs = torch.softmax(outputs, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    pred_label = cnn_classes[pred_idx] if pred_idx < len(cnn_classes) else "Unknown"
                    confidence = probs[pred_idx].item()
            else:
                # YOLOv8 推論
                pred_results = yolo_model.predict(source=img_path, verbose=False)
                if pred_results and len(pred_results) > 0 and pred_results[0].probs is not None:
                    probs = pred_results[0].probs
                    pred_idx = probs.top1
                    pred_label = model_classes[pred_idx]
                    confidence = float(probs.top1conf)
                else:
                    pred_label = "Prediction_Failed"
                    confidence = 0.0
        except Exception as pred_err:
            print(f"  [警告] 預測圖片失敗: {sample['csv_filename']}, 原因: {pred_err}")
            pred_label = "Error"
            confidence = 0.0

        is_correct = (pred_label.lower().strip() == true_label.lower().strip())
        if is_correct:
            correct_predictions += 1

        results_records.append({
            'filename': sample['csv_filename'],
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': round(confidence, 4),
            'correct': 'Yes' if is_correct else 'No'
        })

        if i % max(1, len(samples) // 10) == 0 or i == len(samples):
            print(f"  進度: {i}/{len(samples)} ({int(i/len(samples)*100)}%)")

    # 4. 計算並呈現結果
    accuracy = correct_predictions / len(samples)
    print("-" * 60)
    print(" 評估結果摘要")
    print("-" * 60)
    print(f"總測試樣本數: {len(samples)} 筆")
    print(f"預測正確數:   {correct_predictions} 筆")
    print(f"預測錯誤數:   {len(samples) - correct_predictions} 筆")
    print(f"整體準確率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 5. 輸出結果至 CSV
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, 'test_results.csv')
    
    with open(output_csv_path, mode='w', encoding='utf-8', newline='') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['filename', 'true_label', 'predicted_label', 'confidence', 'correct'])
        writer.writeheader()
        writer.writerows(results_records)
    print(f"\n 預測明細已儲存至: {output_csv_path}")

    # 6. 使用高階分析庫生成詳細報告與混淆矩陣
    if HAS_ANALYSIS_LIBS:
        df = pd.DataFrame(results_records)
        
        all_unique_labels = sorted(list(set(df['true_label'].unique()) | set(df['predicted_label'].unique())))
        
        # 確保模型的所有類別名稱都有加入 (包含沒在資料集出現的)
        for name in model_classes.values():
            if name not in all_unique_labels and name != "Prediction_Failed" and name != "Error":
                all_unique_labels.append(name)
        all_unique_labels = sorted(list(set(all_unique_labels)))

        # 生成分類報告
        print("\n 詳細分類報告 (Classification Report):")
        report = classification_report(
            df['true_label'], 
            df['predicted_label'], 
            labels=all_unique_labels,
            zero_division=0
        )
        print(report)

        # 儲存文字版報告
        report_txt_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_txt_path, 'w', encoding='utf-8') as rep_f:
            rep_f.write("=== 模型測試集評估報告 ===\n")
            rep_f.write(f"模型檔案: {model_path}\n")
            if is_onnx:
                rep_f.write("模型類型: ONNX\n")
            else:
                rep_f.write(f"模型類型: {'PyTorch CNN (' + cnn_arch + ')' if is_cnn else 'YOLOv8'}\n")
            rep_f.write(f"測試樣本總數: {len(samples)}\n")
            rep_f.write(f"整體準確率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            rep_f.write("詳細指標:\n")
            rep_f.write(report)
        print(f" 詳細評估報告已儲存至: {report_txt_path}")

        # 生成並儲存混淆矩陣
        try:
            cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=all_unique_labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=all_unique_labels, 
                yticklabels=all_unique_labels
            )
            title_str = "ONNX" if is_onnx else ('CNN (' + cnn_arch + ')' if is_cnn else 'YOLOv8')
            plt.title(f"Confusion Matrix - {title_str} Test Results")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            cm_img_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_img_path, dpi=300)
            plt.close()
            print(f" 混淆矩陣圖已儲存至: {cm_img_path}")
        except Exception as plt_err:
            print(f" 警告：繪製混淆矩陣圖表失敗: {plt_err}")
    else:
        # 基本模式下的文字報告
        report_txt_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_txt_path, 'w', encoding='utf-8') as rep_f:
            rep_f.write("=== 模型測試集評估報告 (基本模式) ===\n")
            rep_f.write(f"模型檔案: {model_path}\n")
            if is_onnx:
                rep_f.write("模型類型: ONNX\n")
            else:
                rep_f.write(f"模型類型: {'PyTorch CNN (' + cnn_arch + ')' if is_cnn else 'YOLOv8'}\n")
            rep_f.write(f"測試樣本總數: {len(samples)}\n")
            rep_f.write(f"整體準確率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        print(f" 基本評估報告已儲存至: {report_txt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="評估下載的 CNN/YOLOv8 模型在測試集上的表現")
    parser.add_argument('--model', type=str, default='best.pt', help='權重檔路徑 (例如 best.pt)')
    parser.add_argument('--csv', type=str, default='labels.csv', help='資料集標記 CSV 檔案路徑 (例如 labels.csv)')
    parser.add_argument('--images_dir', type=str, default='.', help='圖片檔案存放的根目錄路徑')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='評估結果輸出目錄')
    parser.add_argument('--include_unlabeled', action='store_true', help='是否也要測試未標記的樣本 (預設為跳過)')

    if len(sys.argv) == 1:
        print("=" * 60)
        print(" 進入互動式設定模式 (您也可以使用命令列參數執行本工具)")
        print(" 例如: python evaluate_model.py --model best.pt --csv labels.csv --images_dir .")
        print("=" * 60)
        
        model_input = input("請輸入模型權重檔 (best.pt) 路徑 [預設: best.pt]: ").strip()
        model_path = model_input if model_input else 'best.pt'
        
        csv_input = input("請輸入 labels.csv 檔案路徑 [預設: labels.csv]: ").strip()
        csv_path = csv_input if csv_input else 'labels.csv'
        
        images_input = input("請輸入圖片存放目錄 [預設: 當前目錄 .]: ").strip()
        images_dir = images_input if images_input else '.'
        
        output_dir = './test_results'
        skip_unlabeled = True
    else:
        args = parser.parse_args()
        model_path = args.model
        csv_path = args.csv
        images_dir = args.images_dir
        output_dir = args.output_dir
        skip_unlabeled = not args.include_unlabeled

    run_evaluation(
        model_path=model_path, 
        csv_path=csv_path, 
        images_dir=images_dir, 
        output_dir=output_dir, 
        skip_unlabeled=skip_unlabeled
    )
