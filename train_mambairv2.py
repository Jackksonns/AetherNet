# -*- coding: utf-8 -*-
"""
Training script for pure MambaIRv2 model
"""

import os
import math
import random
import glob
from io import BytesIO
from typing import Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Repro & small utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _make_img_size_divisible(patch_size, window_size):
    if patch_size % window_size == 0:
        return patch_size
    return int(np.ceil(patch_size / window_size) * window_size)

# -------------------------
# Dataset (same as reference)
# -------------------------
class SRDataset(Dataset):
    def __init__(self, root_hr=None, root_lq=None, root_gt=None,
                 task='classical_sr', scale=4, patch=48, noise=0, jpeg=0,
                 split='train', gray=False, window_size=8):
        super().__init__()
        self.task = task
        self.scale = int(scale)
        self.patch = int(patch)
        self.noise = int(noise)
        self.jpeg = int(jpeg)
        self.gray = bool(gray)
        self.split = split
        self.window = int(window_size)

        self.root_hr = root_hr
        self.root_lq = root_lq
        self.root_gt = root_gt

        if split == 'train':
            # 检查root_hr/GT目录是否存在，如果存在则使用它
            gt_dir = os.path.join(self.root_hr, 'GT') if self.root_hr else None
            if gt_dir and os.path.isdir(gt_dir):
                self.hr_files = _imagelist_from_dir(gt_dir)
            elif self.root_hr and os.path.isdir(self.root_hr):
                self.hr_files = _imagelist_from_dir(self.root_hr)
            else:
                raise RuntimeError(f"train HR dir invalid: {self.root_hr}")
            # 添加调试信息
            print(f"训练集路径: {root_hr}")
            print(f"找到的训练图像数量: {len(self.hr_files)}")
        else:
            if self.root_lq and self.root_gt and os.path.isdir(self.root_lq) and os.path.isdir(self.root_gt):
                self.lq_files = _imagelist_from_dir(self.root_lq)
                self.gt_files = _imagelist_from_dir(self.root_gt)
                self.pairs = []
                gt_map = {os.path.basename(p): p for p in self.gt_files}
                for p in self.lq_files:
                    name = os.path.basename(p)
                    if name in gt_map:
                        self.pairs.append((p, gt_map[name]))
            else:
                self.hr_files = _imagelist_from_dir(self.root_hr)

    def __len__(self):
        if self.split == 'train':
            return len(self.hr_files) * 50  # <-- mimic reference repeat
        return len(self.pairs) if hasattr(self, 'pairs') else len(self.hr_files)

    def __getitem__(self, index):
        if self.split == 'train':
            hr_path = self.hr_files[index % len(self.hr_files)]
        elif hasattr(self, 'pairs'):
            lr_path, hr_path = self.pairs[index]
        else:
            hr_path = self.hr_files[index]

        hr = Image.open(hr_path).convert('L' if self.gray else 'RGB')
        if self.split == 'train':
            # Train: generate LR on-the-fly
            lr, hr = _generate_lr(hr, self.task, self.scale, self.noise, self.jpeg, self.patch)
        else:
            # Val: load LR image if exists
            if hasattr(self, 'pairs'):
                lr = Image.open(lr_path).convert('L' if self.gray else 'RGB')
            else:
                lr, hr = _generate_lr(hr, self.task, self.scale, self.noise, self.jpeg, 0)

        # to tensor [0,1]
        if self.gray:
            lr = np.array(lr).astype(np.float32) / 255.0
            hr = np.array(hr).astype(np.float32) / 255.0
            lr = torch.from_numpy(lr).unsqueeze(0)
            hr = torch.from_numpy(hr).unsqueeze(0)
        else:
            lr = torch.from_numpy(np.array(lr).astype(np.float32).transpose(2, 0, 1)) / 255.0
            hr = torch.from_numpy(np.array(hr).astype(np.float32).transpose(2, 0, 1)) / 255.0
        return lr, hr

def _imagelist_from_dir(root):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f'**/*{e}'), recursive=True)
    files = sorted(files)
    return files

def _generate_lr(hr, task, scale, noise, jpeg, patch_size):
    if patch_size > 0:
        hr = _extract_patch(hr, patch_size, scale)
    lr = _apply_degradation(hr, task, scale, noise, jpeg)
    return lr, hr

def _extract_patch(im: Image.Image, patch: int, scale: int) -> Image.Image:
    w, h = im.size
    ww, hh = patch * scale, patch * scale
    if w < ww or h < hh:
        im = im.resize((max(w, ww), max(h, hh)), resample=Image.BICUBIC)
        w, h = im.size
    l, t = random.randint(0, w - ww), random.randint(0, h - hh)
    im = im.crop((l, t, l + ww, t + hh))
    return im

def _apply_degradation(im: Image.Image, task, scale, noise, jpeg):
    # bicubic | gaussian | jpeg
    if task in ('classical_sr', 'lightweight_sr', 'real_sr'):
        # Resize: HR -> LR (scale)
        w, h = im.size
        lr = im.resize((w // scale, h // scale), resample=Image.BICUBIC)
    elif task in ('gray_dn', 'color_dn'):
        lr = im
    elif task in ('jpeg_car', 'color_jpeg_car'):
        lr = im
    else:
        raise ValueError(f"Unknown task: {task}")

    # jpeg compression
    if jpeg > 0:
        with BytesIO() as f:
            lr.save(f, format='JPEG', quality=jpeg)
            f.seek(0)
            lr = Image.open(f).copy()

    # noise injection
    if noise > 0:
        lr = np.array(lr)
        lr = lr + np.random.normal(0, noise, lr.shape)
        lr = np.clip(lr, 0, 255).astype(np.uint8)
        lr = Image.fromarray(lr)

    # final downscale (for SR tasks)
    if task in ('classical_sr', 'lightweight_sr', 'real_sr'):
        lr = lr.resize((w // scale, h // scale), resample=Image.BICUBIC)

    return lr

# -------------------------
# Dataloaders (same as reference)
# -------------------------
def build_train_loader(args, window_size):
    gray = (args.task in ('gray_dn', 'jpeg_car'))
    ds = SRDataset(
        root_hr=args.train_dir, task=args.task, scale=args.scale,
        patch=getattr(args, 'training_patch_size', 64),
        noise=getattr(args, 'noise', 0), jpeg=getattr(args, 'jpeg', 0),
        split='train', gray=gray, window_size=window_size
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    return loader

def build_val_loader(args, window_size):
    gray = (args.task in ('gray_dn', 'jpeg_car'))
    use_pairs = bool(args.folder_val_lq and args.folder_val_gt and
                     os.path.isdir(args.folder_val_lq) and os.path.isdir(args.folder_val_gt))
    if use_pairs:
        ds = SRDataset(root_lq=args.folder_val_lq, root_gt=args.folder_val_gt,
                       task=args.task, scale=args.scale, noise=getattr(args, 'noise', 0),
                       jpeg=getattr(args, 'jpeg', 0), split='val', gray=gray, window_size=window_size)
    else:
        ds = SRDataset(root_hr=args.val_dir, task=args.task, scale=args.scale,
                       noise=getattr(args, 'noise', 0), jpeg=getattr(args, 'jpeg', 0),
                       split='val', gray=gray, window_size=window_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return loader

# -------------------------
# Train / Validate loop (same logic as reference)
# -------------------------
def train(args, define_model_fn, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(getattr(args, 'seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sanitize by task
    if args.task == 'classical_sr':
        args.noise = 0; args.jpeg = 0
        args.degradation = getattr(args, 'degradation', 'bicubic')
    elif args.task in ('gray_dn', 'color_dn'):
        args.scale = 1; args.degradation = 'gaussian'; args.jpeg = 0
    elif args.task in ('jpeg_car', 'color_jpeg_car'):
        args.scale = 1; args.degradation = 'jpeg'; args.noise = 0

    model, window_size = define_model_fn(args)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    train_loader = build_train_loader(args, window_size)
    val_loader = build_val_loader(args, window_size)

    best_psnr = -1.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", dynamic_ncols=True, leave=False)
        for i, (lq, gt) in enumerate(pbar, 1):
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            loss = criterion(pred, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{(epoch_loss/i):.4f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        tqdm.write(f"Epoch [{epoch}/{args.epochs}]  AvgLoss: {avg_loss:.6f}")

        # validate
        if val_loader is not None:
            model.eval()
            psnr_total, seen = 0.0, 0
            with torch.no_grad():
                for lr, hr in tqdm(val_loader, desc="Validate", leave=False, dynamic_ncols=True):
                    lr, hr = lr.to(device), hr.to(device)
                    pred = model(lr)
                    psnr = _calculate_psnr(pred, hr)
                    psnr_total += psnr
                    seen += 1
            avg_psnr = psnr_total / max(1, seen)
            tqdm.write(f"Epoch [{epoch}/{args.epochs}]  ValPSNR: {avg_psnr:.4f}")

            # save best
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
                tqdm.write(f"Saved best model with PSNR: {best_psnr:.4f}")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    tqdm.write(f"Early stopping at epoch {epoch}")
                    break

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final.pth"))
    tqdm.write(f"Training finished. Best PSNR: {best_psnr:.4f}")

def _calculate_psnr(pred, gt):
    """Calculate PSNR between predicted and ground truth images"""
    # 处理尺寸不匹配问题，取较小的尺寸
    if pred.shape != gt.shape:
        min_h = min(pred.shape[2], gt.shape[2])
        min_w = min(pred.shape[3], gt.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        gt = gt[:, :, :min_h, :min_w]
    
    mse = F.mse_loss(pred, gt, reduction='none').view(pred.shape[0], -1).mean(dim=1)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean().item()

# -------------------------
# Model definition (Pure MambaIRv2)
# -------------------------
def define_model(args):
    # 从项目中导入MambaIRv2模型
    from models.mambairv2_standalone.basicsr.archs.mambairv2_arch import MambaIRv2
    
    # Task-specific configs
    jpeg_tasks = {'jpeg_car', 'color_jpeg_car'}
    task = getattr(args, 'task', '')
    window_size = 7 if task in jpeg_tasks else 8
    training_patch = int(getattr(args, 'training_patch_size', 64))
    model_img_size = _make_img_size_divisible(training_patch, window_size)

    # Model configs (following MambaIRv2 reference)
    model_configs = {
        'MambaIRv2-S': dict(embed_dim=48,  upsampler='pixelshuffledirect', resi_connection='1conv'),
        'MambaIRv2-M': dict(embed_dim=96,  upsampler='pixelshuffle',       resi_connection='1conv'),
        'MambaIRv2-L': dict(embed_dim=128, upsampler='pixelshuffle',       resi_connection='1conv'),
    }
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model_size {args.model_size}")
    cfg = model_configs[args.model_size]

    use_sr_upsampler = (task in ('classical_sr', 'lightweight_sr', 'real_sr')) and (args.scale > 1)
    upsampler = cfg['upsampler'] if use_sr_upsampler else ''

    # Depths/heads follow reference (uniform stacks)
    depths = [6]*6  # 6 layers, 6 blocks each
    num_heads = [6]*6
    mlp_ratio = 2

    # For denoise/jpeg tasks, use different img_range; keep consistent
    img_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0

    model = MambaIRv2(
        upscale=args.scale,
        in_chans=3 if 'gray' not in task else 1,
        img_size=model_img_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=cfg['embed_dim'],
        num_heads=tuple(num_heads),
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=cfg['resi_connection'] if task in ('classical_sr','lightweight_sr') else (
            '3conv' if (task=='real_sr' and getattr(args,'large_model',False)) else '1conv'
        ),
        d_state=16,        # Default d_state for Mamba
        inner_rank=32,     # Default inner_rank for Mamba
        num_tokens=64,     # Default num_tokens for Mamba
        convffn_kernel_size=5,  # Default convffn_kernel_size
    )

    # sanity check SR upsampler geometry like reference
    if use_sr_upsampler:
        with torch.no_grad():
            # 确保模型在验证前已经在正确的设备上
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            C = 1 if ('gray' in task) else 3
            dummy = torch.randn(1, C, 16, 16)
            # 确保dummy张量在GPU上，如果可用的话
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = model(dummy)
            assert out.shape[-2:] == (16*args.scale, 16*args.scale), "SR head must scale correctly"
    return model, window_size

# -------------------------
# CLI
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train pure MambaIRv2 model")
    parser.add_argument("--task", type=str, default="classical_sr",
                        choices=["classical_sr","lightweight_sr","real_sr","gray_dn","color_dn","jpeg_car","color_jpeg_car"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="MambaIRv2-M", choices=["MambaIRv2-S","MambaIRv2-M","MambaIRv2-L"])
    parser.add_argument("--large_model", action="store_true")

    parser.add_argument("--training_patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    
    # 数据集路径参数
    parser.add_argument("--train_dir", type=str, default="datasets/train/GT", help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="datasets/val", help="Validation data directory")
    parser.add_argument("--folder_val_lq", type=str, default="", help="Paired validation LQ directory")
    parser.add_argument("--folder_val_gt", type=str, default="", help="Paired validation GT directory")
    parser.add_argument("--noise", type=int, default=0, help="Noise level for dn tasks")
    parser.add_argument("--jpeg", type=int, default=0, help="JPEG quality for jpeg tasks")
    parser.add_argument("--degradation", type=str, default="bicubic", help="Degradation type for SR tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="model_zoo/MambaIRv2", help="Save directory")
    args = parser.parse_args()
    
    # Print configurations
    print("====== Training Pure MambaIRv2 ======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    train(args, define_model, args.save_dir)

if __name__ == "__main__":
    main()