import os
import pydicom
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
IMG_SIZE = 224
LUNG_WINDOW_LEVEL = -600
LUNG_WINDOW_WIDTH = 1500

def apply_ct_window(image):
    min_val = LUNG_WINDOW_LEVEL - LUNG_WINDOW_WIDTH / 2
    max_val = LUNG_WINDOW_LEVEL + LUNG_WINDOW_WIDTH / 2
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    return image

def preprocess_data():
    """
    rowデータを読み込み、前処理を行なってprocessedディレクトリに保存する
    """
    print(f"---前処理を開始します---")
    print(f"---{RAW_DATA_DIR}ディレクトリからデータを読み込みます---")
    print(f"---{PROCESSED_DATA_DIR}ディレクトリに保存します---")

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    for file in tqdm(glob.glob(os.path.join(RAW_DATA_DIR, '*.dcm'))):
        ds = pydicom.dcmread(file)
        img = ds.pixel_array
        img = apply_ct_window(img)
        class_names = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
        for class_name in class_names:
            print(f"---{class_name}ディレクトリからデータを読み込みます---")
            input_dir = os.path.join(RAW_DATA_DIR, class_name)
            output_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
            os.makedirs(output_dir, exist_ok=True)
            dcm_files = glob.glob(os.path.join(input_dir, '*.dcm'))
            for dcm_path in tqdm(dcm_files, desc=f"Processing {class_name}..."):
                try:
                    dicom_data = pydicom.dcmread(dcm_path)
                    image_array = dicom_data.pixel_array.astype(np.float32)
                    windowed_image = apply_ct_window(image_array)
                    
                    image = Image.fromarray(windowed_image)
                    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                    
                    base_filename = os.path.splitext(os.path.basename(dcm_path))[0]
                    output_path = os.path.join(output_dir, f"{base_filename}.png")
                    image.save(output_path)
                    
                except Exception as e:
                    print(f"---{dcm_path}の処理中にエラーが発生しました: {e}---")
                    continue
    print(f"---前処理が完了しました---")

if __name__ == "__main__":
    preprocess_data()