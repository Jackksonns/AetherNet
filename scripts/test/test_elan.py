# -*- coding: utf-8 -*-
"""
Testing script for ELAN model
"""

import os
import math
import random
import glob
from io import BytesIO
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

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
class TestDataset(Dataset):
    def __init__(self, root_lq=None, root_gt=None, task='classical_sr', scale=4, noise=0, jpeg=0, gray=False, window_size=8):
        super().__init__()
        self.task = task
        self.scale = int(scale)
        self.noise = int(noise)
        self.jpeg = int(jpeg)
        self.gray = bool(gray)
        self.window = int(window_size)

        self.root_lq = root_lq
        self.root_gt = root_gt

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
            raise RuntimeError(f"Test dirs invalid: lq={self.root_lq}, gt={self.root_gt}")

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

def _imagelist_from_dir(root):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f'**/*{e}'), recursive=True)
    files = sorted(files)
    return files

# -------------------------
# Metrics (same as reference)
# -------------------------
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

def calculate_ssim(img1, img2):
    """计算SSIM指标"""
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if img1.ndim == 3:
        ssims = []
        for i in range(img1.shape[0]):
            ssims.append(ssim(img1[i], img2[i]))
        return np.array(ssims).mean()
    elif img1.ndim == 2:
        return ssim(img1, img2)
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_rmse(pred, gt):
    """计算RMSE指标"""
    if pred.shape != gt.shape:
        min_h = min(pred.shape[2], gt.shape[2])
        min_w = min(pred.shape[3], gt.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        gt = gt[:, :, :min_h, :min_w]
    
    mse = F.mse_loss(pred, gt, reduction='none').view(pred.shape[0], -1).mean(dim=1)
    rmse = torch.sqrt(mse)
    return rmse.mean().item()

# -------------------------
# Test function (same logic as reference)
# -------------------------
def test(args, define_model_fn, save_dir):
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

    # Load model weights
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    test_dataset = TestDataset(
        root_lq=args.test_lq_dir, root_gt=args.test_gt_dir,
        task=args.task, scale=args.scale, noise=getattr(args, 'noise', 0),
        jpeg=getattr(args, 'jpeg', 0), gray=(args.task in ('gray_dn', 'jpeg_car')),
        window_size=window_size
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    psnr_total, ssim_total, rmse_total = 0.0, 0.0, 0.0
    seen = 0
    
    # 创建输出目录
    output_dir = os.path.join(save_dir, 'output_images')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    with torch.no_grad():
        for lr, hr, filename in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            lr, hr = lr.to(device), hr.to(device)
            pred = model(lr)
            
            # 计算指标
            psnr = _calculate_psnr(pred, hr)
            rmse = calculate_rmse(pred, hr)
            
            # 转换为numpy计算SSIM
            pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            hr_np = hr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            if pred_np.shape[2] == 1:  # 灰度图
                pred_np = pred_np.squeeze(2)
                hr_np = hr_np.squeeze(2)
            
            pred_np = (pred_np * 255).astype(np.uint8)
            hr_np = (hr_np * 255).astype(np.uint8)
            ssim = calculate_ssim(pred_np, hr_np)
            
            psnr_total += psnr
            ssim_total += ssim
            rmse_total += rmse
            seen += 1
            
            # 保存结果
            results.append({
                'filename': filename[0],
                'psnr': psnr,
                'ssim': ssim,
                'rmse': rmse
            })
            
            # 保存输出图像
            pred_img = (pred.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            if pred_img.shape[2] == 1:
                pred_img = pred_img.squeeze(2)
            Image.fromarray(pred_img).save(os.path.join(output_dir, filename[0]))
            
            print(f"{filename[0]}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, RMSE={rmse:.4f}")

    # 计算平均指标
    avg_psnr = psnr_total / max(1, seen)
    avg_ssim = ssim_total / max(1, seen)
    avg_rmse = rmse_total / max(1, seen)
    
    print(f"\n=== Test Results ===")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    
    # 保存详细结果
    with open(os.path.join(save_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
        f.write("ELAN Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Scale: {args.scale}\n")
        f.write(f"Model: {args.model_size}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Test Images: {seen}\n")
        f.write("\n")
        f.write("Per-image Results:\n")
        f.write("-" * 30 + "\n")
        for r in results:
            f.write(f"{r['filename']}: PSNR={r['psnr']:.4f}, SSIM={r['ssim']:.4f}, RMSE={r['rmse']:.4f}\n")
        f.write("\n")
        f.write("Average Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")
    
    # 保存简化摘要
    with open(os.path.join(save_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"RMSE: {avg_rmse:.4f}\n")

# -------------------------
# Model definition (ELAN)
# -------------------------
def define_model(args):
    # 导入ELAN模型
    from models.elan import ELAN
    
    # Task-specific configs
    jpeg_tasks = {'jpeg_car', 'color_jpeg_car'}
    task = getattr(args, 'task', '')
    window_size = 7 if task in jpeg_tasks else 8
    training_patch = int(getattr(args, 'training_patch_size', 64))
    model_img_size = _make_img_size_divisible(training_patch, window_size)

    # Model configs (ELAN specific)
    model_configs = {
        'ELAN-Light': dict(embed_dim=36, depths=[6]*4, upsampler='pixelshuffledirect', resi_connection='1conv'),
        'ELAN-Standard': dict(embed_dim=48, depths=[6]*6, upsampler='pixelshuffle', resi_connection='1conv'),
        'ELAN-Large': dict(embed_dim=64, depths=[6]*6, upsampler='pixelshuffle', resi_connection='1conv'),
    }
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model_size {args.model_size}")
    cfg = model_configs[args.model_size]

    use_sr_upsampler = (task in ('classical_sr', 'lightweight_sr', 'real_sr')) and (args.scale > 1)
    upsampler = cfg['upsampler'] if use_sr_upsampler else ''

    # Depths/heads follow ELAN reference
    depths = cfg['depths']
    num_heads = [6] * len(depths)  # ELAN使用固定的6个注意力头
    mlp_ratio = 4  # ELAN使用4倍MLP比率

    # For denoise/jpeg tasks, use different img_range; keep consistent
    img_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0

    model = ELAN(
        img_size=model_img_size,
        patch_size=1,
        in_chans=3 if 'gray' not in task else 1,
        embed_dim=cfg['embed_dim'],
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        upscale=args.scale,
        img_range=img_range,
        upsampler=upsampler,
        resi_connection=cfg['resi_connection'] if task in ('classical_sr','lightweight_sr') else (
            '3conv' if (task=='real_sr' and getattr(args,'large_model',False)) else '1conv'
        ),
        group_size=4,  # ELAN特有的分组大小
    )

    return model, window_size

# -------------------------
# CLI
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test ELAN model")
    parser.add_argument("--task", type=str, default="classical_sr",
                        choices=["classical_sr","lightweight_sr","real_sr","gray_dn","color_dn","jpeg_car","color_jpeg_car"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="ELAN-Standard", choices=["ELAN-Light","ELAN-Standard","ELAN-Large"])
    parser.add_argument("--large_model", action="store_true")

    parser.add_argument("--training_patch_size", type=int, default=64)
    parser.add_argument("--noise", type=int, default=0, help="Noise level for dn tasks")
    parser.add_argument("--jpeg", type=int, default=0, help="JPEG quality for jpeg tasks")
    parser.add_argument("--degradation", type=str, default="bicubic", help="Degradation type for SR tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 测试数据路径参数
    parser.add_argument("--test_lq_dir", type=str, required=True, help="Test LQ data directory")
    parser.add_argument("--test_gt_dir", type=str, required=True, help="Test GT data directory")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory for results")
    
    args = parser.parse_args()
    
    # Print configurations
    print("====== Testing ELAN ======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    test(args, define_model, args.save_dir)

if __name__ == "__main__":
    main()
