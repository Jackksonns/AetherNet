# -*- coding: utf-8 -*-
"""
Testing script for JkNet model (modified to only test scale=4)
"""

import os
import math
import glob
from io import BytesIO
import argparse
import numpy as np
from PIL import Image
import cv2
import datetime

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
    # 从项目中导入JkNet模型
    from models.jknet import JkNet
    
    # Task-specific configs
    jpeg_tasks = {'jpeg_car', 'color_jpeg_car'}
    task = getattr(args, 'task', '')
    window_size = 7 if task in jpeg_tasks else 8
    training_patch = int(getattr(args, 'training_patch_size', 64))
    model_img_size = _make_img_size_divisible(training_patch, window_size)

    # Model configs (following SwinIR reference for fairness)
    model_configs = {
        'SwinIR-S': dict(embed_dim=60,  upsampler='pixelshuffledirect', resi_connection='1conv'),
        'SwinIR-M': dict(embed_dim=180, upsampler='pixelshuffle',       resi_connection='1conv'),
        'SwinIR-L': dict(embed_dim=240, upsampler='pixelshuffle',       resi_connection='1conv'),
        'MambaIRv2-S': dict(embed_dim=60,  upsampler='pixelshuffledirect', resi_connection='1conv'),
        'MambaIRv2-M': dict(embed_dim=180, upsampler='pixelshuffle',       resi_connection='1conv'),
        'MambaIRv2-L': dict(embed_dim=240, upsampler='pixelshuffle',       resi_connection='1conv'),
    }
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model_size {args.model_size}")
    cfg = model_configs[args.model_size]

    use_sr_upsampler = (task in ('classical_sr', 'lightweight_sr', 'real_sr')) and (args.scale > 1)
    upsampler = cfg['upsampler'] if use_sr_upsampler else ''

    # Heads/depths mimic the reference (uniform stacks)
    depths = [6]*6
    num_heads = [6]*6
    mlp_ratio = 2

    # For denoise/jpeg tasks, SwinIR uses different img_range; we keep consistency
    img_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0

    # JkNet-specific knobs: wavelet hint, FPA, and RSTB tail
    model = JkNet(
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
        # --- JkNet extras ---
        use_wavelet_hint=True,
        wavelet_gate=True,
        use_fpa=True,
        num_tail_rstb=1,
        tail_depth=2,
        tail_num_heads=None,      # default -> use first of num_heads
        tail_window_size=None     # default -> window_size
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
                
                if args.task in ('gray_dn', 'jpeg_car'):
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
    
    # Return results instead of printing
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_rmse': avg_rmse,
        'num_samples': num_samples,
        'results': results
    }

# -------------------------
# Auto Test All Datasets (modified to only test scale=4)
# -------------------------
def discover_test_datasets(data_root):
    """Discover all available test datasets and scales"""
    datasets = {}
    
    # Common test dataset names
    dataset_names = ['Set5', 'Set14', 'BSD100', 'Urban100', 'Manga109']
    
    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_root, dataset_name)
        if os.path.isdir(dataset_path):
            datasets[dataset_name] = {}
            
            # Only check scale=4 (modified from original to only test scale=4)
            scale = 4
            scale_dir = os.path.join(dataset_path, f"image_SRF_{scale}")
            if os.path.isdir(scale_dir):
                hr_dir = os.path.join(scale_dir, "HR")
                lr_dir = os.path.join(scale_dir, "LR")
                if os.path.isdir(hr_dir) and os.path.isdir(lr_dir):
                    datasets[dataset_name][scale] = {
                        'hr_dir': hr_dir,
                        'lr_dir': lr_dir
                    }
    
    return datasets

def auto_test_all_datasets(args, model_path, output_dir="auto_test_results"):
    """Automatically test all available datasets with scale=4 only"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Discover all available datasets
    data_root = os.path.dirname(model_path) if not args.data_root else args.data_root
    datasets = discover_test_datasets(data_root)
    
    if not datasets:
        print("No test datasets found!")
        return
    
    print(f"Found {len(datasets)} test datasets (only scale=4 will be tested):")
    for name, scales in datasets.items():
        print(f"  {name}: scales {list(scales.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for sample images
    sample_images_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    # Load model once
    model, window_size = define_model(args)
    
    if os.path.exists(model_path):
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
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
    
    # Store all results
    all_results = {}
    global_psnr_sum = 0.0
    global_ssim_sum = 0.0
    global_rmse_sum = 0.0
    global_count = 0
    
    # Test each dataset and scale (only scale=4)
    for dataset_name, scales in datasets.items():
        all_results[dataset_name] = {}
        
        for scale, paths in scales.items():
            # Only test scale=4
            if scale != 4:
                print(f"Skipping scale {scale}x, only scale=4 is tested")
                continue
                
            print(f"\nTesting {dataset_name} with scale factor {scale}x...")
            
            # Update args for this specific test
            test_args = argparse.Namespace(**vars(args))
            test_args.test_lq_dir = paths['lr_dir']
            test_args.test_gt_dir = paths['hr_dir']
            test_args.scale = scale
            
            # Create test dataset and loader
            gray = (test_args.task in ('gray_dn', 'jpeg_car'))
            test_dataset = TestDataset(
                lq_dir=test_args.test_lq_dir,
                gt_dir=test_args.test_gt_dir,
                task=test_args.task,
                scale=test_args.scale,
                gray=gray
            )
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
            
            # Create directory for this dataset and scale
            dataset_scale_dir = os.path.join(sample_images_dir, f"{dataset_name}_x{scale}")
            os.makedirs(dataset_scale_dir, exist_ok=True)
            
            # Test loop
            psnr_total = 0.0
            ssim_total = 0.0
            rmse_total = 0.0
            results = []
            
            with torch.no_grad():
                for i, (lq, gt, filename) in enumerate(tqdm(test_loader, desc=f"Testing {dataset_name}x{scale}")):
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
                    
                    # Save first 2 images (LQ, GT, and restored)
                    if i < 4:
                        # Convert tensors to PIL Images
                        # LQ image
                        lq_np = lq.squeeze().cpu().numpy()
                        if len(lq_np.shape) == 3:
                            lq_np = lq_np.transpose(1, 2, 0)
                        lq_np = np.clip(lq_np * 255.0, 0, 255).astype(np.uint8)
                        
                        if gray:
                            lq_img = Image.fromarray(lq_np.squeeze(), mode='L')
                        else:
                            lq_img = Image.fromarray(lq_np)
                        
                        # GT image
                        gt_np = gt.squeeze().cpu().numpy()
                        if len(gt_np.shape) == 3:
                            gt_np = gt_np.transpose(1, 2, 0)
                        gt_np = np.clip(gt_np * 255.0, 0, 255).astype(np.uint8)
                        
                        if gray:
                            gt_img = Image.fromarray(gt_np.squeeze(), mode='L')
                        else:
                            gt_img = Image.fromarray(gt_np)
                        
                        # Restored image
                        pred_np = pred.squeeze().cpu().numpy()
                        if len(pred_np.shape) == 3:
                            pred_np = pred_np.transpose(1, 2, 0)
                        pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
                        
                        if gray:
                            pred_img = Image.fromarray(pred_np.squeeze(), mode='L')
                        else:
                            pred_img = Image.fromarray(pred_np)
                        
                        # Save images with proper naming
                        base_name = os.path.splitext(filename[0])[0]
                        
                        lq_path = os.path.join(dataset_scale_dir, f"{base_name}_LQ.png")
                        gt_path = os.path.join(dataset_scale_dir, f"{base_name}_GT.png")
                        pred_path = os.path.join(dataset_scale_dir, f"{base_name}_Restored.png")
                        
                        lq_img.save(lq_path)
                        gt_img.save(gt_path)
                        pred_img.save(pred_path)
                        
                        print(f"Saved sample images for {filename[0]}")
            
            # Calculate average metrics
            num_samples = len(test_loader)
            avg_psnr = psnr_total / num_samples
            avg_ssim = ssim_total / num_samples
            avg_rmse = rmse_total / num_samples
            
            # Store results
            all_results[dataset_name][scale] = {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'avg_rmse': avg_rmse,
                'num_samples': num_samples,
                'results': results
            }
            
            # Update global sums
            global_psnr_sum += avg_psnr * num_samples
            global_ssim_sum += avg_ssim * num_samples
            global_rmse_sum += avg_rmse * num_samples
            global_count += num_samples
            
            # Print results for this dataset/scale
            print(f"  PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, RMSE: {avg_rmse:.6f}")
    
    # Calculate global averages
    global_avg_psnr = global_psnr_sum / global_count
    global_avg_ssim = global_ssim_sum / global_count
    global_avg_rmse = global_rmse_sum / global_count
    
    # Generate summary report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"test_summary_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("JkNet Model Test Results Summary (Scale=4 only)\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_size}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Model weights: {model_path}\n")
        f.write(f"Testing scale: 4x only\n")
        f.write("="*60 + "\n\n")
        
        # Global averages
        f.write("GLOBAL AVERAGES (all datasets, scale=4 only)\n")
        f.write("-"*40 + "\n")
        f.write(f"Average PSNR:  {global_avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM:  {global_avg_ssim:.4f}\n")
        f.write(f"Average RMSE:  {global_avg_rmse:.6f}\n")
        f.write(f"Total images tested: {global_count}\n\n")
        
        # Results by dataset
        f.write("RESULTS BY DATASET (Scale=4 only)\n")
        f.write("="*60 + "\n")
        f.write(f"{'Dataset':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'RMSE':<10} {'Images':<8}\n")
        f.write("-"*60 + "\n")
        
        for dataset_name in sorted(all_results.keys()):
            for scale in sorted(all_results[dataset_name].keys()):
                if scale == 4:  # Only show scale=4 results
                    result = all_results[dataset_name][scale]
                    f.write(f"{dataset_name:<12} {result['avg_psnr']:<12.4f} {result['avg_ssim']:<10.4f} {result['avg_rmse']:<10.6f} {result['num_samples']:<8}\n")
        
        # Note about sample images
        f.write("\nSAMPLE IMAGES\n")
        f.write("="*60 + "\n")
        f.write(f"Sample images (LQ, GT, and Restored) have been saved to:\n")
        f.write(f"{sample_images_dir}\n")
        f.write("Each dataset has its own subdirectory with the first 4 test images (scale=4 only).\n")
    
    print(f"\nTest completed! Results saved to {output_dir}")
    print(f"Sample images saved to {sample_images_dir}")
    print(f"Summary report saved to {report_file}")
    
    return all_results

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Test JkNet model (Scale=4 only)")
    parser.add_argument("--task", type=str, default="classical_sr",
                        choices=["classical_sr","lightweight_sr","real_sr","gray_dn","color_dn","jpeg_car","color_jpeg_car"])
    parser.add_argument("--scale", type=int, default=4,  # Changed from 2 to 4
                        help="Scale factor for super-resolution (fixed to 4 for this test)")
    parser.add_argument("--model_size", type=str, default="SwinIR-M", 
                        choices=["SwinIR-S","SwinIR-M","SwinIR-L","MambaIRv2-S","MambaIRv2-M","MambaIRv2-L"])
    parser.add_argument("--large_model", action="store_true")
    parser.add_argument("--training_patch_size", type=int, default=64)
    
    # Test data paths
    parser.add_argument("--test_lq_dir", type=str, 
                        help="Test LQ (low quality) images directory")
    parser.add_argument("--test_gt_dir", type=str, 
                        help="Test GT (ground truth) images directory")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained model weights (.pth file)")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="Jknetx4_testResults", 
                        help="Directory to save test results and output images")
    
    # Auto test options
    parser.add_argument("--auto_test", action="store_true",
                        help="Automatically test all available datasets (scale=4 only)")
    parser.add_argument("--data_root", type=str,
                        help="Root directory containing test datasets")
    
    args = parser.parse_args()
    
    # Print configurations
    print("====== Testing JkNet (Scale=4 only) ======")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    # Run test
    if args.auto_test:
        auto_test_all_datasets(args, args.model_path, args.save_dir)
    else:
        if not args.test_lq_dir or not args.test_gt_dir:
            print("Error: Both --test_lq_dir and --test_gt_dir must be provided for single dataset testing")
            return
            
        results = test(args, args.model_path, args.save_dir)
        
        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS (Scale=4 only)")
        print("="*60)
        print(f"Average PSNR:  {results['avg_psnr']:.4f} dB")
        print(f"Average SSIM:  {results['avg_ssim']:.4f}")
        print(f"Average RMSE:  {results['avg_rmse']:.6f}")
        print(f"Tested on {results['num_samples']} images")
        
        # Print individual results
        print("\nIndividual Results:")
        print("-" * 80)
        print(f"{'Filename':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'RMSE':<10}")
        print("-" * 80)
        for result in results['results']:
            print(f"{result['filename']:<30} {result['psnr']:<12.4f} {result['ssim']:<10.4f} {result['rmse']:<10.6f}")
        
        # Save results to file
        if args.save_dir:
            result_file = os.path.join(args.save_dir, "test_results.txt")
            with open(result_file, 'w') as f:
                f.write("TEST RESULTS (Scale=4 only)\n")
                f.write("="*60 + "\n")
                f.write(f"Model: {args.model_size}\n")
                f.write(f"Task: {args.task}\n")
                f.write(f"Scale: {args.scale}\n")
                f.write(f"Model weights: {args.model_path}\n")
                f.write(f"Test LQ dir: {args.test_lq_dir}\n")
                f.write(f"Test GT dir: {args.test_gt_dir}\n")
                f.write("="*60 + "\n")
                f.write(f"Average PSNR:  {results['avg_psnr']:.4f} dB\n")
                f.write(f"Average SSIM:  {results['avg_ssim']:.4f}\n")
                f.write(f"Average RMSE:  {results['avg_rmse']:.6f}\n")
                f.write(f"Tested on {results['num_samples']} images\n\n")
                f.write("Individual Results:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Filename':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'RMSE':<10}\n")
                f.write("-" * 80 + "\n")
                for result in results['results']:
                    f.write(f"{result['filename']:<30} {result['psnr']:<12.4f} {result['ssim']:<10.4f} {result['rmse']:<10.6f}\n")
            print(f"\nDetailed results saved to: {result_file}")
            
            # Save a summary file
            summary_file = os.path.join(args.save_dir, "results_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"PSNR: {results['avg_psnr']:.4f}\n")
                f.write(f"SSIM: {results['avg_ssim']:.4f}\n")
                f.write(f"RMSE: {results['avg_rmse']:.6f}\n")
                f.write(f"Scale: 4\n")
            print(f"Results summary saved to: {summary_file}")

if __name__ == "__main__":
    main()