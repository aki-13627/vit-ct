# scripts/evaluate.py

import os
import torch
from torchvision import transforms
from PIL import Image
import glob
from collections import Counter
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vit_model import create_vit_model

MODEL_PATH = 'output/【ここに学習結果のタイムスタンプフォルダ名】/models/best_model.pth' 
TEST_DATA_DIR = 'data/test'
NUM_CLASSES = 2
IMG_SIZE = 224
CLASS_NAMES = ['normal', 'pneumonia']

def evaluate_cases():
    device = torch.device("mps")
    model = create_vit_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"モデルをロードしました: {MODEL_PATH}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    y_true = []
    y_pred = []

    print("\n--- 症例ごとの評価を開始します ---")
    for true_label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        case_dirs = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
        
        for case_id in tqdm(case_dirs, desc=f"Processing {class_name} cases"):
            case_path = os.path.join(class_dir, case_id)
            image_paths = glob.glob(os.path.join(case_path, '*.png'))
            
            if not image_paths:
                continue

            slice_predictions = []
            with torch.no_grad():
                for img_path in image_paths:
                    image = Image.open(img_path)
                    image = transform(image).unsqueeze(0).to(device) # バッチ次元を追加
                    
                    outputs = model(image).logits
                    _, predicted_idx = torch.max(outputs, 1)
                    slice_predictions.append(predicted_idx.item())
            
            # 多数決で症例の最終予測を決定
            final_prediction = Counter(slice_predictions).most_common(1)[0][0]
            
            y_true.append(true_label_idx)
            y_pred.append(final_prediction)

    if not y_true:
        print("評価データが見つかりませんでした。")
        return

    correct_predictions = sum(p == t for p, t in zip(y_pred, y_true))
    total_cases = len(y_true)
    accuracy = correct_predictions / total_cases

    # 混同行列（Confusion Matrix）を手動で計算
    # TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative
    # (クラス1をPositive、クラス0をNegativeとする)
    tp, tn, fp, fn = 0, 0, 0, 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1: tp += 1
        elif true == 0 and pred == 0: tn += 1
        elif true == 0 and pred == 1: fp += 1
        elif true == 1 and pred == 0: fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n--- 評価結果 (症例単位) ---")
    print(f"評価した総症例数: {total_cases}")
    print(f"Accuracy (正解率): {accuracy:.4f}")
    print(f"Sensitivity (感度): {sensitivity:.4f}")
    print(f"Specificity (特異度): {specificity:.4f}")
    print("----------------------------")

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません。パスを確認してください: {MODEL_PATH}")
    elif not os.path.exists(TEST_DATA_DIR):
        print(f"エラー: テストデータディレクトリが見つかりません。パスを確認してください: {TEST_DATA_DIR}")
    else:
        evaluate_cases()