# src/evaluate.py
"""
Batch evaluation script for Sea Ice Segmentation
Reads parameters from YAML config and processes all PNGs in a directory.
Usage:
  python evaluate.py --config configs/default.yaml --img_dir data/processed/ssha_png
"""
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import ToTensor

from model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Batch Evaluate Sea Ice Segmentation')
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'),
                        help='Path to YAML config file')
    parser.add_argument('--img_dir', type=Path, required=True,
                        help='Directory containing grayscale PNG images for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    # load config
    cfg = yaml.safe_load(open(args.config))
    eval_cfg = cfg.get('evaluation', {})
    model_cfg = cfg.get('model', {})

    # parameters from config
    weights_path = Path(eval_cfg.get('weights', 'results/best_model.pth'))
    result_dir   = Path(eval_cfg.get('result_dir', 'results'))
    num_classes  = int(model_cfg.get('num_classes', 3))
    colors       = eval_cfg.get('colors', [[0,0,255], [255,255,255], [0,255,0]])
    device       = eval_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    result_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = get_model(num_classes=num_classes,
                      pretrained_backbone=model_cfg.get('pretrained_backbone', False))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    # process all images in directory
    img_paths = sorted(args.img_dir.glob('*.png'))
    for img_path in img_paths:
        # read and preprocess image
        img = np.array(Image.open(img_path), dtype=np.uint8)
        img3 = np.stack([img, img, img], axis=-1)
        x = ToTensor()(img3).unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            out = model(x)['out'].softmax(1).argmax(1).cpu().numpy()[0]

        # prepare base name
        stem = img_path.stem

        # save one-hot
        h, w = out.shape
        onehot = np.zeros((num_classes, h, w), dtype=np.uint8)
        for c in range(num_classes):
            onehot[c] = (out == c).astype(np.uint8)
        np.save(result_dir / f'onehot_{stem}.npy', onehot)

        # color segmentation
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for c, col in enumerate(colors):
            vis[out == c] = col
        cv2.imwrite(str(result_dir / f'segmentation_{stem}.png'), vis)

        # extract and overlay contours for ice class (label 1)
        ice_mask = (out == 1).astype(np.uint8)
        contours, _ = cv2.findContours(
            ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, contours, -1, tuple(colors[-1]), 2)
        cv2.imwrite(str(result_dir / f'overlay_{stem}.png'), overlay)

if __name__ == '__main__':
    main()
