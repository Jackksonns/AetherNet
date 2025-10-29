# -*- coding: utf-8 -*-
"""
Testing script for EDSR model with self-ensemble support

This script provides:
1. Standard testing on benchmark datasets
2. Self-ensemble (x8 geometric augmentation)
3. Forward chopping for large images
4. Comprehensive evaluation metrics (PSNR, SSIM, RMSE)

Usage example:
python test_edsr.py --task classical_sr --scale 4 \
  --model_size EDSR-baseline --model_path model_zoo/EDSR/best.pth \
  --test_lq_dir datasets/Set5/LR --test_gt_dir datasets/Set5/HR \
  --self_ensemble --save_images
"""

import os
import math
import glob
import argparse
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Model definition (same as training)
# -------------------------
from models.edsr import EDSR

def define_model(args):
    """
    Map CLI args -> EDSR config.
    """
    task = getattr(args, 'task', '')
    
    # Model configs
    model_configs = {
        'EDSR-baseline': dict(n_resblocks=16, n_feats=64, res_scale=1.0),
        'EDSR': dict(n_resblocks=32, n_feats=256, res_scale=0.1),
    }
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model_size {args.model_size}")
    cfg = model_configs[args.model_size]

    # RGB range
    rgb_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0
    
    # Number of colors
    n_colors = 1 if 'gray' in task else 3

    model = EDSR(
        n_resblocks=cfg['n_resblocks'],
        n_feats=cfg['n_feats'],
        scale=args.scale,
        res_scale=cfg['res_scale'],
        n_colors=n_colors,
        rgb_range=rgb_range
    )

    return model

# -------------------------
# Self-Ensemble Functions
# -------------------------
def forward_chop(model, x, scale, shave=10, min_size=160000):
    """
    Forward function with image chopping for large images.
    Splits image into patches if too large, processes separately, then merges.
    """
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, scale, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def forward_x8(model, x, forward_function=None):
    """
    Self-ensemble with 8 geometric augmentations.
    Applies 8 transformations, forwards each, then averages the inverse-transformed outputs.
    """
    if forward_function is None:
        forward_function = lambda m, inp: m(inp)
    
    def _transform(v, op):
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(v.device)
        return ret

    lr_list = [x]
    for tf in 'v', 'h', 't':
        lr_list.extend([_transform(t, tf) for t in lr_list])

    sr_list = []
    for aug in lr_list:
        sr = forward_function(model, aug)
        sr_list.append(sr)

    for i in range(len(sr_list)):
        if i > 3:
            sr_list[i] = _transform(sr_list[i], 't')
        if i % 4 > 1:
            sr_list[i] = _transform(sr_list[i], 'h')
        if (i % 4) % 2 == 1:
            sr_list[i] = _transform(sr_list[i], 'v')

    output_cat = torch.cat(sr_list, dim=0)
    output = output_cat.mean(dim=0, keepdim=True)
    return output

# -------------------------
# Test Dataset
# -------------------------
def _imagelist_from_dir(root):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f'**/*{e}'), recursive=True)
    files = sorted(files)
    return files

class TestDataset(Dataset):
    def __init__(self, lq_dir=None, gt_dir=None, task='classical_sr', scale=4, gray=False):
        super().__init__()
        self.task = task
        self.scale = int(scale)
        self.gray = bool(gray)
        
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir

        if self.lq_dir and self.gt_dir and os.path.isdir(self.lq_dir) and os.path.isdir(self.gt_dir):
            self.lq_files = _imagelist_from_dir(self.lq_dir)
            self.gt_files = _imagelist_from_dir(self.gt_dir)
            self.pairs = []
            gt_map = {os.path.basename(p): p for p in self.gt_files}
            for p in self.lq_files:
                name = os.path.basename(p)
                if name in gt_map:
                    self.pairs.append((p, gt_map[name]))
        else:
            raise RuntimeError(f"Test dirs invalid: lq={self.lq_dir}, gt={self.gt_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]
        lr = Image.open(lr_path).convert('L' if self.gray else 'RGB')
        hr = Image.open(hr_path).convert('L' if self.gray else 'RGB')

        # to tensor [0,1]
        if self.gray:
            lr = np.array(lr).astype(np.float32) / 255.0
            hr = np.array(hr).astype(np.float32) / 255.0
            lr = torch.from_numpy(lr).unsqueeze(0)
            hr = torch.from_numpy(hr).unsqueeze(0)
        else:
            lr = torch.from_numpy(np.array(lr).astype(np.float32).transpose(2, 0, 1)) / 255.0
            hr = torch.from_numpy(np.array(hr).astype(np.float32).transpose(2, 0, 1)) / 255.0
        return lr, hr, os.path.basename(lr_path)

# -------------------------
# Metrics
# -------------------------
def calculate_psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 99.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM using OpenCV"""
    if img1.ndim == 2:
        return cv2.quality.QualitySSIM_compute(img1, img2)[0]
    elif img1.ndim == 3:
        ssim_sum = 0
        for i in range(img1.shape[2]):
            ssim_sum += cv2.quality.QualitySSIM_compute(img1[:,:,i], img2[:,:,i])[0]
        return ssim_sum / img1.shape[2]
    else:
        raise ValueError('Input images must be 2D or 3D arrays')

def calculate_rmse(pred, gt):
    """Calculate RMSE between predicted and ground truth images"""
    if pred.shape != gt.shape:
        min_h = min(pred.shape[2], gt.shape[2])
        min_w = min(pred.shape[3], gt.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        gt = gt[:, :, :min_h, :min_w]
    
    mse = F.mse_loss(pred, gt, reduction='none').view(pred.shape[0], -1).mean(dim=1)
    rmse = torch.sqrt(mse)
    return rmse.mean().item()

# -------------------------
# Test function
# -------------------------
def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = define_model(args)
    model = model.to(device)

    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Test dataset
    test_dataset = TestDataset(
        lq_dir=args.test_lq_dir, gt_dir=args.test_gt_dir,
        task=args.task, scale=args.scale, 
        gray=(args.task in ('gray_dn', 'jpeg_car'))
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    psnr_total, ssim_total, rmse_total = 0.0, 0.0, 0.0
    seen = 0
    
    # Create output directory
    output_dir = os.path.join(args.save_dir, 'output_images')
    if args.save_images:
        os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    with torch.no_grad():
        for lr, hr, filename in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            lr, hr = lr.to(device), hr.to(device)
            
            # Forward pass
            if args.self_ensemble:
                if args.chop_forward:
                    # Self-ensemble with chopping
                    pred = forward_x8(model, lr, 
                                     forward_function=lambda m, x: forward_chop(m, x, args.scale))
                else:
                    # Self-ensemble without chopping
                    pred = forward_x8(model, lr)
            else:
                if args.chop_forward:
                    # Standard forward with chopping
                    pred = forward_chop(model, lr, args.scale)
                else:
                    # Standard forward
                    pred = model(lr)
            
            # Calculate metrics
            # For PSNR: convert to numpy uint8
            pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            hr_np = hr.squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)
            pred_np = (pred_np * 255.0).round().astype(np.uint8)
            hr_np = (hr_np * 255.0).round().astype(np.uint8)
            
            # PSNR with crop_border
            border = args.scale if 'sr' in args.task else 0
            psnr = calculate_psnr(pred_np, hr_np, crop_border=border, input_order='HWC')
            
            # SSIM
            if pred_np.ndim == 2:
                ssim = calculate_ssim(pred_np, hr_np)
            else:
                ssim = calculate_ssim(pred_np, hr_np)
            
            # RMSE on tensors
            rmse = calculate_rmse(pred, hr)
            
            psnr_total += psnr
            ssim_total += ssim
            rmse_total += rmse
            seen += 1
            
            # Save results
            results.append({
                'filename': filename[0],
                'psnr': psnr,
                'ssim': ssim,
                'rmse': rmse
            })
            
            # Save output image
            if args.save_images:
                output_path = os.path.join(output_dir, filename[0])
                if pred_np.ndim == 2:
                    Image.fromarray(pred_np).save(output_path)
                else:
                    Image.fromarray(pred_np).save(output_path)
            
            print(f"{filename[0]}: PSNR={psnr:.2f}dB, SSIM={ssim:.4f}, RMSE={rmse:.4f}")
    
    # Calculate averages
    avg_psnr = psnr_total / max(1, seen)
    avg_ssim = ssim_total / max(1, seen)
    avg_rmse = rmse_total / max(1, seen)
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    
    # Save detailed results
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
        f.write("EDSR Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Scale: {args.scale}\n")
        f.write(f"Model: {args.model_size}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Self-Ensemble: {args.self_ensemble}\n")
        f.write(f"Chop Forward: {args.chop_forward}\n")
        f.write(f"Test Images: {seen}\n")
        f.write("\n")
        f.write("Per-image Results:\n")
        f.write("-" * 30 + "\n")
        for r in results:
            f.write(f"{r['filename']}: PSNR={r['psnr']:.4f}dB, SSIM={r['ssim']:.4f}, RMSE={r['rmse']:.4f}\n")
        f.write("\n")
        f.write("Average Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")
    
    # Save simplified summary
    with open(os.path.join(args.save_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")
    
    print(f"\nResults saved to {args.save_dir}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Test EDSR model with self-ensemble support")
    
    # Task and Model Configs
    parser.add_argument("--task", type=str, default="classical_sr",
                        choices=["classical_sr","lightweight_sr","real_sr","gray_dn","color_dn","jpeg_car","color_jpeg_car"])
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--model_size", type=str, default="EDSR-baseline", choices=["EDSR-baseline","EDSR"])
    
    # Test Dataset
    parser.add_argument("--test_lq_dir", type=str, required=True, help="Test LQ (low-quality) images directory")
    parser.add_argument("--test_gt_dir", type=str, required=True, help="Test GT (ground-truth) images directory")
    
    # Model Path
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    
    # Test Options
    parser.add_argument("--self_ensemble", action="store_true", help="Use self-ensemble (x8) for testing")
    parser.add_argument("--chop_forward", action="store_true", help="Use forward chopping for large images")
    parser.add_argument("--save_images", action="store_true", help="Save output SR images")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="test_results/EDSR", help="Directory to save test results")

    args = parser.parse_args()
    
    print("====== Testing EDSR ======")
    for arg in vars(args):
        print(f"{arg.ljust(25)}: {getattr(args, arg)}")
    print("=" * 50)
    
    test(args)

if __name__ == "__main__":
    main()

