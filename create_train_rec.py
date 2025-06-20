#!/usr/bin/env python3
"""Utility to create MXNet record files from an ImageFolder dataset.

The script scans an image directory with subdirectories per class and
produces ``train.rec``, ``train.idx`` and ``train.lst`` compatible with
``dataset.record_dataset.AugmentRecordDataset``.

Example
-------
python create_rec.py --img_dir path/to/imgs

By default the output files are written to the parent directory of
``img_dir``. This matches the expected layout in ``README_TRAIN.md``.
"""

from pathlib import Path
import argparse
import os

import cv2
import mxnet as mx
from tqdm import tqdm
from torchvision.datasets import ImageFolder


def build_rec(img_dir: Path, output_dir: Path, quality: int = 95) -> None:
    """Create record files from images under ``img_dir``.

    Parameters
    ----------
    img_dir : Path
        Directory containing class subfolders with images.
    output_dir : Path
        Directory to store ``train.rec``, ``train.idx`` and ``train.lst``.
    quality : int, optional
        JPEG quality for encoding, by default 95.
    """
    dataset = ImageFolder(str(img_dir))
    num_images = len(dataset.samples)
    num_classes = len(dataset.classes)

    idx_path = output_dir / "train.idx"
    rec_path = output_dir / "train.rec"
    lst_path = output_dir / "train.lst"

    record = mx.recordio.MXIndexedRecordIO(str(idx_path), str(rec_path), "w")
    header0 = mx.recordio.IRHeader(1, [num_images + 1, num_classes], 0, 0)
    record.write_idx(0, mx.recordio.pack(header0, b""))

    with open(lst_path, "w") as fout:
        for idx, (path, label) in enumerate(tqdm(dataset.samples), start=1):
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            header = mx.recordio.IRHeader(0, label, idx, 0)
            packed = mx.recordio.pack_img(header, img, quality=quality, img_fmt=".jpg")
            record.write_idx(idx, packed)
            rel_path = os.path.relpath(path, img_dir)
            fout.write(f"{idx}\t{label}\t{rel_path}\n")
    record.close()
    print(f"Wrote {num_images} images belonging to {num_classes} classes")
    print(f"Record file : {rec_path}")
    print(f"Index file  : {idx_path}")
    print(f"List file   : {lst_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create MXNet rec from ImageFolder dataset")
    parser.add_argument("--img_dir", required=True, help="Directory with class subfolders")
    parser.add_argument("--output_dir", help="Where to write rec files. Defaults to parent of img_dir")
    parser.add_argument("--quality", type=int, default=95, help="JPEG encoding quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise ValueError(f"img_dir {img_dir} does not exist")

    output_dir = Path(args.output_dir) if args.output_dir else img_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    build_rec(img_dir, output_dir, args.quality)


if __name__ == "__main__":
    main()