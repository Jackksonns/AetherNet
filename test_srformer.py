# -*- coding: utf-8 -*-
"""
Testing script for SRFormer model
"""

import os
import math
import glob
from io import BytesIO
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
def _make_img_size_divisible(patch_size, window_size):
    if patch_size % window_size == 0:
        return patch_size
    return int(np.ceil(patch_size / window_size) * window_size)

def define_model(args):
    # 从项目中导入SRFormer模型
    from models.srformer import SRFormer
    
    # Task-specific configs
    jpeg_tasks = {'jpeg_car', 'color_jpeg_car'}
    task = getattr(args, 'task', '')
    window_size = 7 if task in jpeg_tasks else 8
    training_patch = int(getattr(args, 'training_patch_size', 64))
    model_img_size = _make_img_size_divisible(training_patch, window_size)

    # Model configs (SRFormer specific)
    model_configs = {
        'SRFormer-S': dict(embed_dim=60, depths=[6]*4, upsampler='pixelshuffledirect', resi_connection='1conv'),
        'SRFormer-M': dict(embed_dim=180, depths=[6]*6, upsampler='pixelshuffle', resi_connection='1conv'),
        'SRFormer-L': dict(embed_dim=240, depths=[6]*6, upsampler='pixelshuffle', resi_connection='1conv'),
    }
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model_size {args.model_size}")
    cfg = model_configs[args.model_size]

    use_sr_upsampler = (task in ('classical_sr', 'lightweight_sr', 'real_sr')) and (args.scale > 1)
    upsampler = cfg['upsampler'] if use_sr_upsampler else ''

    # Depths/heads follow SRFormer reference
    depths = cfg['depths']
    num_heads = [6] * len(depths)  # SRFormer使用固定的6个注意力头
    mlp_ratio = 2

    # For denoise/jpeg tasks, use different img_range; keep consistent
    img_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0

    model = SRFormer(
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
    )
    return model, window_size

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
        
        # Load LQ and GT file pairs
        if lq_dir and gt_dir and os.path.isdir(lq_dir) and os.path.isdir(gt_dir):
            self.lq_files = _imagelist_from_dir(lq_dir)
            self.gt_files = _imagelist_from_dir(gt_dir)
            self.pairs = []
            gt_map = {os.path.basename(p): p for p in self.gt_files}
            for p in self.lq_files:
                name = os.path.basename(p)
                if name in gt_map:
                    self.pairs.append((p, gt_map[name]))
            print(f"Found {len(self.pairs)} paired test images")
        else:
            raise ValueError("Both LQ and GT directories must be provided for testing")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lq_path, gt_path = self.pairs[index]
        
        # Load images
        lq = Image.open(lq_path).convert('L' if self.gray else 'RGB')
        gt = Image.open(gt_path).convert('L' if self.gray else 'RGB')
        
        # Convert to tensor [0,1]
        if self.gray:
            lq_tensor = torch.from_numpy(np.array(lq).astype(np.float32) / 255.0).unsqueeze(0)
            gt_tensor = torch.from_numpy(np.array(gt).astype(np.float32) / 255.0).unsqueeze(0)
        else:
            lq_tensor = torch.from_numpy(np.array(lq).astype(np.float32).transpose(2, 0, 1)) / 255.0
            gt_tensor = torch.from_numpy(np.array(gt).astype(np.float32).transpose(2, 0, 1)) / 255.0
            
        filename = os.path.basename(lq_path)
        return lq_tensor, gt_tensor, filename

# -------------------------
# Metrics
# -------------------------
def _calculate_psnr(pred, gt):
    """Calculate PSNR between predicted and ground truth images"""
    if pred.shape != gt.shape:
        min_h = min(pred.shape[2], gt.shape[2])
        min_w = min(pred.shape[3], gt.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        gt = gt[:, :, :min_h, :min_w]
    
    mse = F.mse_loss(pred, gt, reduction='none').view(pred.shape[0], -1).mean(dim=1)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.mean().item()

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Convert to numpy for OpenCV
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.squeeze().cpu().numpy()
    
    # Handle grayscale vs RGB
    if len(img1.shape) == 3:
        img1 = img1.transpose(1, 2, 0)
    if len(img2.shape) == 3:
        img2 = img2.transpose(1, 2, 0)
        
    # Ensure same size
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]
    
    # Calculate SSIM using OpenCV
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    if len(img1.shape) == 2:  # Grayscale
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
    else:  # RGB
        ssim_values = []
        for i in range(3):
            channel1 = img1[:, :, i]
            channel2 = img2[:, :, i]
            
            mu1 = cv2.GaussianBlur(channel1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(channel2, (11, 11), 1.5)
            
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(channel1 ** 2, (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(channel2 ** 2, (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(channel1 * channel2, (11, 11), 1.5) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_values.append(ssim_map.mean())
        return np.mean(ssim_values)

def calculate_rmse(pred, gt):
    """Calculate RMSE between predicted and ground truth images"""
    if pred.shape != gt.shape:
        min_h = min(pred.shape[2], gt.shape[2])
        min_w = min(pred.shape[3], gt.shape[3])
        pred = pred[:, :, :min_h, :min_w]
        gt = gt[:, :, :min_h, :min_w]
    
    # Calculate MSE first, then take square root
    mse = F.mse_loss(pred, gt, reduction='mean').item()
    rmse = math.sqrt(mse)
    return rmse

# -------------------------
# Test Function
# -------------------------
def test(args, model_path, save_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, window_size = define_model(args)
    
    # Load trained weights
    if os.path.exists(model_path):
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Create test dataset and loader
    gray = (args.task in ('gray_dn', 'jpeg_car'))
    test_dataset = TestDataset(
        lq_dir=args.test_lq_dir,
        gt_dir=args.test_gt_dir,
        task=args.task,
        scale=args.scale,
        gray=gray
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "output_images"), exist_ok=True)
    
    # Test loop
    psnr_total = 0.0
    ssim_total = 0.0
    rmse_total = 0.0
    results = []
    
    with torch.no_grad():
        for lq, gt, filename in tqdm(test_loader, desc="Testing"):
            lq, gt = lq.to(device), gt.to(device)
            
            # Forward pass
            pred = model(lq)
            
            # Calculate metrics
            psnr = _calculate_psnr(pred, gt)
            ssim = calculate_ssim(pred, gt)
            rmse = calculate_rmse(pred, gt)
            
            psnr_total += psnr
            ssim_total += ssim
            rmse_total += rmse
            
            results.append({
                'filename': filename[0],
                'psnr': psnr,
                'ssim': ssim,
                'rmse': rmse
            })
            
            # Save result images if requested
            if save_dir:
                # Convert tensor to PIL Image
                pred_np = pred.squeeze().cpu().numpy()
                if len(pred_np.shape) == 3:
                    pred_np = pred_np.transpose(1, 2, 0)
                pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
                
                if gray:
                    pred_img = Image.fromarray(pred_np.squeeze(), mode='L')
                else:
                    pred_img = Image.fromarray(pred_np)
                
                save_path = os.path.join(save_dir, "output_images", filename[0])
                pred_img.save(save_path)
    
    # Calculate average metrics
    num_samples = len(test_loader)
    avg_psnr = psnr_total / num_samples
    avg_ssim = ssim_total / num_samples
    avg_rmse = rmse_total / num_samples
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Average PSNR:  {avg_psnr:.4f} dB")
    print(f"Average SSIM:  {avg_ssim:.4f}")
    print(f"Average RMSE:  {avg_rmse:.6f}")
    print(f"Tested on {num_samples} images")
    
    # Print individual results
    print("\nIndividual Results:")
    print("-" * 80)
    print(f"{'Filename':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'RMSE':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['filename']:<30} {result['psnr']:<12.4f} {result['ssim']:<10.4f} {result['rmse']:<10.6f}")
    
    # Save results to file
    if save_dir:
        result_file = os.path.join(save_dir, "test_results.txt")
        with open(result_file, 'w') as f:
            f.write("TEST RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {args.model_size}\n")
            f.write(f"Task: {args.task}\n")
            f.write(f"Scale: {args.scale}\n")
            f.write(f"Model weights: {model_path}\n")
            f.write(f"Test LQ dir: {args.test_lq_dir}\n")
            f.write(f"Test GT dir: {args.test_gt_dir}\n")
            f.write("="*60 + "\n")
            f.write(f"Average PSNR:  {avg_psnr:.4f} dB\n")
            f.write(f"Average SSIM:  {avg_ssim:.4f}\n")
            f.write(f"Average RMSE:  {avg_rmse:.6f}\n")
            f.write(f"Tested on {num_samples} images\n\n")
            f.write("Individual Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Filename':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'RMSE':<10}\n")
            f.write("-" * 80 + "\n")
            for result in results:
                f.write(f"{result['filename']:<30} {result['psnr']:<12.4f} {result['ssim']:<10.4f} {result['rmse']:<10.6f}\n")
        print(f"\nDetailed results saved to: {result_file}")
        
        # Save a summary file
        summary_file = os.path.join(save_dir, "results_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"PSNR: {avg_psnr:.4f}\n")
            f.write(f"SSIM: {avg_ssim:.4f}\n")
            f.write(f"RMSE: {avg_rmse:.6f}\n")
        print(f"Results summary saved to: {summary_file}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Test SRFormer model")
    parser.add_argument("--task", type=str, default="classical_sr",
                        choices=["classical_sr","lightweight_sr","real_sr","gray_dn","color_dn","jpeg_car","color_jpeg_car"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="SRFormer-M", 
                        choices=["SRFormer-S","SRFormer-M","SRFormer-L"])
    parser.add_argument("--large_model", action="store_true")
    parser.add_argument("--training_patch_size", type=int, default=64)
    
    # Test data paths
    parser.add_argument("--test_lq_dir", type=str, required=True, 
                        help="Test LQ (low quality) images directory")
    parser.add_argument("--test_gt_dir", type=str, required=True, 
                        help="Test GT (ground truth) images directory")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model weights (.pth file)")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="test_results", 
                        help="Directory to save test results and output images")
    
    args = parser.parse_args()
    
    # Print configurations
    print("====== Testing SRFormer ======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    # Run test
    test(args, args.model_path, args.save_dir)

if __name__ == "__main__":
    main()
