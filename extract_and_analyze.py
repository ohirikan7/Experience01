import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from AdaFace.net import build_model
from AdaFace.dataset.image_folder_dataset import CustomImageFolderDataset


def extract_features(model, dataloader, device):
    model.eval()
    features = []

    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            embeddings, _ = model(images)  # embeddings: (B, D)
            features.append(embeddings.cpu())

    all_features = torch.cat(features, dim=0)  # (N, D)
    return all_features


def main(args):
    # モデルの読み込み
    model = build_model('ir_101')
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items() if 'model.' in k})
    model.to(args.device)

    # データセット読み込み
    transform = transforms.Compose([
        transforms.Resize((112, 112)), # 112x112にリサイズ
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = CustomImageFolderDataset(root=args.img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # 特徴抽出
    features = extract_features(model, dataloader, args.device)

    # 分散計算（全体の分散：Var(X)）
    mean = features.mean(dim=0)                # (D,)
    centered = features - mean                 # (N, D)
    norm_sq = (centered ** 2).sum(dim=1)       # 各ベクトルのL2ノルム2乗 (N,)
    var_x = norm_sq.mean().item()              # 分散 Var(X)

    print(f"Total number of features: {features.shape[0]}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Variance Var(X): {var_x:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpoint (.ckpt)')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory to image folder dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    main(args)
