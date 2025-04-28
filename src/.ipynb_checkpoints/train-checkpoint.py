# src/train.py
"""
Training script for Sea Ice Semantic Segmentation
Reads hyperparameters from YAML config, including checkpoint interval.
Usage:
  python train.py --config configs/default.yaml
"""
import argparse
import yaml
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import SeaIceDataset
from model import get_model
from utils import iou_metric


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sea Ice Segmentation')
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'),
                        help='Path to YAML config file')
    return parser.parse_args()


def freeze_bn_layers(model):
    """Freeze BatchNorm layers to inference mode to avoid single-sample issues"""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            # disable gradient
            for p in m.parameters(): p.requires_grad = False


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # extract config
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    model_cfg = cfg['model']

    # directories
    img_dir = Path(data_cfg['img_dir'])
    lbl_dir = Path(data_cfg['lbl_dir'])
    out_dir = Path(train_cfg.get('out_dir', 'results'))
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / 'tb_logs'
    tb_dir.mkdir(exist_ok=True)

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir/'train.log')]
    )
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir=tb_dir)

    # dataset
    imgs = sorted(img_dir.glob('*.png'))
    lbls = sorted(lbl_dir.glob('*.png'))
    dataset = SeaIceDataset(imgs, lbls, transform=None)
    train_size = int(len(dataset) * train_cfg['train_ratio'])
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg.get('seed', 42))
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=train_cfg.get('pin_memory', True),
        drop_last=True        # 丢弃最后一个样本数 < batch_size 的 batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg['batch_size'], shuffle=False,
        num_workers=train_cfg.get('num_workers', 4), pin_memory=train_cfg.get('pin_memory', True)
    )

    # model
    device = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_cfg['num_classes'], pretrained_backbone=model_cfg.get('pretrained_backbone', False))
    model.to(device)
    # freeze BN if needed
    if train_cfg.get('freeze_bn', False):
        freeze_bn_layers(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg['lr']), weight_decay=float(train_cfg['weight_decay'])
    )

    # checkpoint & curve interval from config
    ckpt_epoch_interval = train_cfg.get('save_interval_epochs', 1)

    best_miou = 0.0
    history = {'loss': [], 'miou': []}

    for epoch in range(1, train_cfg['epochs'] + 1):
        # train
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch} Train'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)['out']
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        epoch_loss = total_loss / len(train_ds)
        history['loss'].append(epoch_loss)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # validate
        model.eval()
        total_iou = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f'Epoch {epoch} Val'):
                x, y = x.to(device), y.to(device)
                out = model(x)['out']
                total_iou += iou_metric(out, y, model_cfg['num_classes']).item() * x.size(0)
        epoch_miou = total_iou / len(val_ds)
        history['miou'].append(epoch_miou)
        writer.add_scalar('mIoU/val', epoch_miou, epoch)

        logger.info(f'Epoch {epoch:02d}: train_loss={epoch_loss:.4f}, val_mIoU={epoch_miou:.4f}')

        # save per epoch interval
        if epoch % ckpt_epoch_interval == 0:
            # save model
            ckpt_path = out_dir / f'model_epoch{epoch}.pth'
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f'Saved checkpoint: {ckpt_path}')
            # save loss curve
            import matplotlib.pyplot as plt
            epochs = list(range(1, epoch+1))
            plt.figure()
            plt.plot(epochs, history['loss'], label='Train Loss')
            plt.plot(epochs, history['miou'], label='Val mIoU')
            plt.xlabel('Epoch'); plt.ylabel('Value')
            plt.legend(); plt.tight_layout()
            curve_path = out_dir / f'curve_epoch{epoch}.png'
            plt.savefig(curve_path, dpi=150)
            plt.close()
            logger.info(f'Saved curve: {curve_path}')

        # save best
        if epoch_miou > best_miou:
            best_miou = epoch_miou
            best_path = out_dir / 'best_model.pth'
            torch.save(model.state_dict(), best_path)
            logger.info(f'Saved best model: {best_path} (mIoU={best_miou:.4f})')

    writer.close()
    logger.info('Training completed.')


if __name__ == '__main__':
    main()

