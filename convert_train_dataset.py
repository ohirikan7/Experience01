import os
import shutil
import pandas as pd

def convert_to_imagefolder(csv_path, src_image_root, dest_root):
    """
    CSVと画像ディレクトリを元にImageFolder形式でコピーする関数

    Parameters:
    - csv_path (str): CSVファイルへのパス
    - src_image_root (str): 元画像があるルートディレクトリ
    - dest_root (str): ImageFolder形式で保存するルートディレクトリ
    """
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        file_path = row['File Path']
        person_id = str(row['Person ID'])

        src_path = os.path.join(src_image_root, file_path)
        dest_dir = os.path.join(dest_root, person_id)
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        # print(f"Copied {src_path} to {dest_path}")
