import os
import cv2
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np

def summarize_imagefolder(image_root):
    """
    ImageFolder形式のデータセットを要約する。

    Parameters:
    - image_root (str): ImageFolder形式のルートディレクトリ
    """
    image_root = Path(image_root)
    class_counts = defaultdict(int)
    image_sizes = []
    failed_images = []
    
    for class_dir in sorted(image_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in class_dir.glob("*"):
            if not img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                failed_images.append(str(img_path))
                continue

            h, w = img.shape[:2]
            image_sizes.append((w, h))
            class_counts[class_name] += 1

    # 結果出力
    print("ImageFolder Summary")
    print("--------------------")
    total_images = sum(class_counts.values())
    total_classes = len(class_counts)
    avg_per_class = total_images / total_classes if total_classes > 0 else 0

    print(f"Total images   : {total_images}")
    print(f"Total classes  : {total_classes}")
    print(f"Avg/class      : {avg_per_class:.2f}")

    # サイズ確認
    unique_sizes = set(image_sizes)
    print(f"Unique sizes   : {unique_sizes}")
    if (112, 112) not in unique_sizes:
        print("WARNING: Not all images are 112x112")

    # カラーチャネル確認（BとRの平均ピクセル値の比較）
    sample_imgs = min(10, len(image_sizes))
    if sample_imgs > 0:
        print("Checking color channels (BGR vs RGB)...")
        bgr_diffs = []
        count = 0
        for class_dir in image_root.iterdir():
            for img_path in class_dir.glob("*"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                b, g, r = cv2.split(img)
                bgr_diffs.append(np.mean(r) - np.mean(b))
                count += 1
                if count >= sample_imgs:
                    break
            if count >= sample_imgs:
                break
        avg_diff = np.mean(bgr_diffs)
        print(f"Avg(R - B) pixel diff (should be ~0 for gray images): {avg_diff:.2f}")
        if abs(avg_diff) > 10:
            print("Likely BGR format (OpenCV default)")
        else:
            print("Possibly RGB or grayscale")

    # エラー画像表示
    if failed_images:
        print(f"\n[Warning] Failed to load {len(failed_images)} images:")
        for f in failed_images:
            print(f"  - {f}")
